# Reward specification — v15 and v16

## 0. Design principles inherited from v15, and not broken

These were established before the work in this repository and are treated as constraints.

1. **All terminal potentials are zero**, and shaping is
   `shaping = γ·Φ(next) − Φ(prev)`. The discounted sum of shaping therefore telescopes to
   **exactly zero**, so "stand still" and "march in place" earn nothing from shaping.
   Asserted in every evaluation: `discounted_shaping_max_abs < 2e-5` (measured 1.7e-7).
2. **Task credit is paid only for a new personal-best credible glide.** Not for surviving,
   not for posture.
3. **Failure is a terminal event worth `−2`, and nothing else.** No structure in which
   dying early lets the policy escape an accumulating penalty.
4. **No new negative continuing reward.** If a cost is needed, it must be a *discount* on
   future credit, because a discount cannot be escaped by terminating.
5. Authority over the policy is an actor-side concern and is never changed through the
   reward.

The reward is `task + shaping + failure` in v15 and `task + shaping + failure + handoff` in
v16. `cfg.rewards` is `('task','shaping','failure','handoff')`.

## 1. B — the handoff speed bonus (new in v16)

Define **`entry_speed`** as the COM forward speed at the first frame at which single support
is established (the first frame with `valid`, i.e. left-only support).

* **Paid at most once per episode.**
* The condition is that a valid **and credible** consecutive glide reaches
  `HANDOFF_HOLD_S = 0.25 s`. A throw-and-fall does not get paid.
* Amount:

```python
BRIDGE_HANDOFF_REF_SPEED = .38   # frozen public expert's measured speed ceiling
BRIDGE_HANDOFF_SPAN      = .30   # full credit at .68 m/s
BRIDGE_HANDOFF_GAIN      = 1.2   # below failure_cost, so a reckless lift cannot pay
BRIDGE_HANDOFF_HOLD_S    = .25

handoff = HANDOFF_GAIN * clamp((entry_speed - REF) / SPAN, 0, 1)
```

Every constant has a reason:

* **`REF = .38` m/s is the frozen public expert's measured ceiling** (8 s acceleration
  window). The bonus is **exactly zero** below it. The intended optimum is therefore
  unchanged for existing behaviour; only *new* capability is rewarded.
* **Gain < failure cost.** Full credit (1.2) is less than the 2.0 cost of a fall, so
  flinging the robot into a one-foot stance cannot pay.
* The **0.25 s hold** requirement is what separates "established a glide" from "was briefly
  airborne on one foot".

**Calibration history.** The span was initially `0.15` (full credit at 0.53 m/s) with
`GAIN = 1.0`. Measured entry speeds then clustered such that the gradient vanished at
0.53 m/s and the policy pinned itself there, so the transition
`records/actor-transition-reward-v16b.json` widened the span to `0.30` and the gain to
`1.2`.

**Measured effect.** Entry speed rose from **0.393 m/s** (v15 u1000) to **0.465 m/s**
(v16 u1000), with a maximum of 0.534 m/s — past the frozen expert's 0.38 ceiling, which was
the entire point. The discounted handoff return is reported per evaluation
(`discounted_handoff_return`).

## 2. E — two-axis balance debt (new in v16)

`quality = 1 / (1 + smooth_debt)`, with `smooth_debt += cost · dt`. v16 adds to `cost`, but
**only while the robot is in left-only support**:

```python
BRIDGE_LATERAL_DEADBAND = .008
BRIDGE_LATERAL_WIDTH    = .025
BRIDGE_FOREAFT_DEADBAND = .005
BRIDGE_FOREAFT_WIDTH    = .020
BRIDGE_BALANCE_RATE     = 1.

excess_lat  = max(|lateral_capture_error|  - LATERAL_DEADBAND, 0) / LATERAL_WIDTH
excess_fore = max(|fore_aft_capture_error| - FOREAFT_DEADBAND, 0) / FOREAFT_WIDTH
balance_cost = BALANCE_RATE * (excess_lat**2 + excess_fore**2)
```

| Constant | Value | Basis (measured) |
|---|---:|---|
| `LATERAL_DEADBAND` | 0.008 m | observed p50 of \|lateral capture error\| |
| `LATERAL_WIDTH` | 0.025 m | p90 of 0.031 should give excess ≈ 0.9 |
| `FOREAFT_DEADBAND` | 0.005 m | observed p50 |
| `FOREAFT_WIDTH` | 0.020 m | p90 of 0.0136 should give excess ≈ 0.43 |
| `BALANCE_RATE` | 1.0 | at the p90 of both axes the debt is ≈ 1/s, so `quality ≈ 0.5` |

Why a debt and not a penalty:

* It **accumulates**, so collapsing early discounts all subsequent credit.
* Terminating does not clear it — being a discount, it cannot be escaped.
* Normal gliding (near the p50) accrues essentially nothing, because of the deadbands.

The fore/aft axis is included because credible glides showed the robot drifting out of
balance **fore and aft as well as laterally**: |capture error| moves 0.0058 → 0.0251 m
fore/aft and 0.0072 → 0.0203 m laterally over a glide. A signed fore/aft error means
different things (COM ahead of or behind the support line) and a 3 cm forward escape is a
fall, but v16 treats it symmetrically in absolute value for now. Long glides drift from
negative to positive.

**Measured effect.** Lateral capture error fell from **0.0203 m → 0.0065 m**. It has not
yet produced a 2.0 s glide, because the residual failure is the lateral collapse at
t ≈ 1.5 s described in `GLIDE-PHYSICS.md`.

## 3. E′ — fore/aft in the potential (new in v16)

`credit_bridge_potential(com_speed, skate_speed, capture_error, fore_aft_error, ...)`
gained a second capture factor:

```python
capture_lat  = exp(-(capture_error  / .05)**2)     # pre-existing (lateral)
BRIDGE_FOREAFT_PHI_WIDTH = .02
capture_fore = exp(-(fore_aft_error / .02)**2)     # new (fore/aft)

phi = readiness * moving * posture * straight * capture_lat * capture_fore
```

* Terminals still have `Φ = 0`, so the telescoping argument is unchanged.
* Measured fore/aft |error| is p50 0.005 / p90 0.0136 m. With width 0.02, a p90 episode
  keeps 0.63× of its potential.

## 4. Why these two levers, together

The requirement for a `T`-second credible glide is `v₀ ≥ 0.10 + T·|a|`. B attacks `v₀`; E
attacks `|a|`. Either alone closes the gap:

| Assumed deceleration | Required entry speed | Reachable at the time? |
|---|---:|---|
| a = −0.17 (whole-glide mean then) | 0.44 | no — frozen expert caps at 0.38 |
| a = −0.13 (best quartile) | 0.36 | **yes** |
| a = −0.04 (first 0.25 s of a glide) | 0.18 | yes, comfortably |

This is why the plan was B **and** E rather than a single change. Today B has succeeded
(entry 0.465) and the binding constraint has moved to the lateral balance failure, which is
what E is aimed at.

## 5. Implementation and verification

Changed sources: `tasks/mdp.py` (v15 region only — the legacy prefix is byte-identical),
`tasks/microduck_onefoot_credit_bridge_env_cfg.py`, `bridge_recovery.py`,
`scripts/train-credit-bridge.py`, `scripts/audit-credit-bridge-reward.py`,
`scripts/write-actor-transition.py`, and the two test modules.

Invariants pinned by tests (`pytest tests/` — 434 passed, 1 skipped):

1. `handoff` is **exactly zero** for `entry_speed ≤ 0.38`.
2. `handoff` is paid **once per episode** and requires a 0.25 s credible glide.
3. `balance_debt` does not grow inside the deadband.
4. The discounted failure path equals `−2·γ^(L−1)`.
5. `discounted_shaping_max_abs < 2e-5`.
6. The first 351 782 bytes of `mdp.py` are unchanged.
7. A `reward_forward_transition` may not move the actor, and an
   `actor_forward_transition` may not move the reward.

### Transition records

| Record | Kind | Moved |
|---|---|---|
| `actor-transition-hold-authority.json` | `actor_forward_transition` | the actor only (`HOLD_AUTHORITY`) |
| `actor-transition-reward-v16.json` | `reward_forward_transition` | `mdp.py` + env cfg + tooling; `reward_terms_before ['failure','shaping','task']` → `after [..., 'handoff']` |
| `actor-transition-reward-v16b.json` | `reward_forward_transition` | span/gain widening |
| `actor-transition-glide-stage.json` | `task_contract_transition` | env cfg (owner), poses, `bridge_recovery.py`; `reward_terms_before == after` |

Each `reward_forward_transition` was validated to leave the actor's 39 parameter keys
identical and the frozen-expert digest unchanged.

## 6. Measured result of the v16 block

| Metric | v15 u1000 | v16 u1000 |
|---|---:|---:|
| credible 0.5 s | 90.6 % | **93.8 %** |
| credible 1.0 s | 51.6 % | **64.1 %** |
| credible 2.0 s | 0 % | 0 % |
| longest / mean credible | 1.36 / 0.957 s | **1.56 / 1.069 s** |
| entry speed | 0.393 m/s | **0.465 m/s** |
| mean episode | 4.34 s | **4.73 s** |
| `stalled_transfer` | 27 | **8** |
| `body_ground` | 26 | 51 |

Checkpoint: `checkpoints/full-v16/…` (`model_6899.pt`, the u1000 checkpoint of the
`full-v16-training` block).

**B worked** (entry 0.393 → 0.465, handoff discount +0.107, max entry 0.534 — into the
full-credit region). **E worked** (lateral capture error 0.0203 → 0.0065 m). **But the
deceleration did not improve**, reading −0.168 → −0.233 m/s².

That last number is a **selection effect, not a regression**. Binning by speed shows the
deceleration improving monotonically; low-speed frames are exactly the frames of a glide
that is already collapsing at the end. Above 0.40 m/s the measured deceleration is
−0.170 m/s², and below 0.12 m/s it is −0.326 m/s². The policy did not learn to brake — the
*end* of each failed glide migrated into the low-speed bin.

Using the healthy-regime figure, the condition becomes `0.44 ≤ 0.465`, which **v16
satisfies**. The remaining barrier is therefore not deceleration as such, but that the
robot **falls sideways at around 1.5 s**.
