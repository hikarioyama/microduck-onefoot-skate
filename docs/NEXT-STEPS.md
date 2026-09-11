# Next steps and open questions

The repository is handed over in a state the harness itself flags: the trainer emits
`CURRICULUM_REVIEW_REQUIRED review_required stage=glide (saved; do not cold-restart)`. The
`glide` block finished its 1000 updates and did not pass the 2.0 s gate, so a decision is
outstanding rather than a run being interrupted.

## Where the problem now stands

One sentence: **the glide deceleration must fall by 28 %, and nothing else will do.**

| | value | source |
|---|---:|---|
| mean entry speed, best block | 0.592 m/s | `evidence/milestones.json` |
| mean glide deceleration | 0.343 m/s² | derived from mean credible 1.43 s |
| mean credible glide | 1.43 s | `evidence/milestones.json` |
| best credible glide | 2.14 s | `evidence/milestones.json` |
| four-wheel coast deceleration, for reference | 0.058 m/s² | `records/accel-probe-long.json` |
| deceleration needed for a 2.0 s mean glide | **0.246 m/s²** | derived |
| entry speed needed for a 2.0 s mean glide | 0.786 m/s | derived |

The formal gate agrees and quantifies it: credible 0.5 s 99.6 %, credible 1.0 s 95.3 %,
credible 2.0 s **0.0 % / 0.39 %** on 512 first episodes — a 95 % skill at one second against
a 0.2 % skill at two seconds.

## Routes already closed, with the measurement that closed them

Do not re-litigate these without new evidence.

| Route | Closed by |
|---|---|
| Widen the glide entry band | Implemented (`.50,.76`): credible 1.0 s 0.891 → 0.969, longest 1.62 → 2.14 s, and the first ever 2.0 s glides. Capped: the valid predicate rejects root speed ≥ 0.80 m/s and 80 % would need ~0.94 m/s. |
| Fix the recorded-wheel-speed contradiction | Implemented; it is what made the band change work at all (entry gain 0.054 → 0.184 m/s for the same change in injection). |
| Raise `HOLD_AUTHORITY` | **Rejected by measurement.** A = 0.25 → 1.46 s mean, 1 fall; A = 0.50 → 0.028 s, 72 falls. The correction is a trained equilibrium at A = 0.25, so it cannot be re-scaled post-hoc. |
| Blame the frozen expert for the drag | **Rejected.** Driving it alone gives a 0.022 s glide and 59 / 64 falls — it is not resisting, it is simply not contributing. |
| Rolling resistance / skid / lateral scrub / blade yaw as the drag | **All measured and rejected** (`REPORT.md` §8.2, `GLIDE-PHYSICS.md` §2b). |
| Lower the declared goal | Proposed once from a diagnosis later found wrong. **The 2.0 s criterion is unchanged.** |

## Open directions, none measured yet

Ordered by how directly they attack the deceleration. This is a genuinely open problem: the
loss is real, reproducible and ~6× a rolling coast, and has no identified mechanism after a
systematic elimination.

1. **Make the support skate steer.** Blade yaw sits at a steady ~11° and never moves, so the
   steering degree of freedom is unused. Blade yaw correlated *negatively* with deceleration,
   so this is not a simple scrub story — but a steady 11° yaw on two in-line wheels is not
   obviously a free-rolling configuration either, and nothing has yet asked the policy to
   hold the blade aligned with its travel.
2. **Separate posture drag from balance drag in the frozen expert.** It falls over at once at
   a fast spawn, which masks whether its *posture* also drags. Holding the pose open-loop
   would separate the two.
3. **Look for internal-motion loss.** The swing leg's joints move at up to 7.6 rad/s and the
   deceleration has no dominant frame-level correlate, which is consistent with a distributed
   loss. Instrumenting joint work / power flow would test it.
4. **Retrain the correction at a higher authority** — deliberately, with its own
   `actor_forward_transition`, never as a post-hoc scale factor.
5. **Gate the `full` stage.** It still has no gate result and its screening has never reached
   the 0.8 trigger. `glide` at least now has a measured failure.

## Open decisions

### 1. Continue the assisted `glide` stage, or go back to `full`?

| | Argument |
|---|---|
| **Continue `glide`** | Every episode is 100 % inside the learnable region; the block already reached credible 0.5 s 98.4 % / 1.0 s 89.1 % with only 2+6 falls per 64. It is the cheapest place to work on deceleration. |
| **Return to `full`** | Only `full` can certify a from-rest result, and `full` has not been trained since the `glide` stage was introduced. The v16b `full` block was interrupted at u600 (the mid-run source edit, `REPORT.md` §9) and could be resumed from `full-v16b-training/state-recovered.json`. |
| **Both, in sequence** | Use `glide` to work the deceleration lever, then re-enter `full` and check whether the from-rest path inherits it. This is what the curriculum was designed for, and it is the option not yet tried. |

The untried option is the last one: **no `full` block has ever run with a produced `glide`
stage behind it.**

### 2. Which lever — deceleration or entry speed?

**Settled: deceleration.** The entry-speed lever has been worked (§ above) and is capped by
the predicate; the deceleration lever is the only one left. It needs 0.343 → 0.246 m/s².
Reward term E did move the lateral capture error a long way (0.0203 → 0.0065 m) without
converting into glide duration, so an incentive aimed at how the glide *ends* rather than at
lateral capture is the obvious next reward experiment. See the "Open directions" list for
what has not been tried.

### 3. The fall-versus-replant asymmetry — an undecided judgement call

Falls currently dominate `full`-stage terminations, and the reward treats a fall and a
replanted swing foot asymmetrically. The asymmetry has never been justified from
measurement; it is a design assumption that was inherited. It remains open, and changing it
would change the reward weights, which is why it has not been touched unilaterally.

### 4. Lower the declared goal?

Proposed once, rejected once. The proposal came from a diagnosis (§ "A correction that
matters" in `docs/GLIDE-PHYSICS.md`) that turned out to be wrong. **The 2.0 s criterion is
unchanged** and no lower figure has been adopted. This option exists in the record only so
that adopting it later would be a visible decision rather than a silent drift.

### 5. Gate the `full` stage properly

`glide` has now been gated directly (`records/glide-glide-02-gate.json`) and failed on the
2.0 s duration alone. `full` still has no gate result, because the trainer only attempts the
gate when 64-episode screening reaches 80 % on `credible_200` and it never has. Run
`scripts/run-bridge-stage-gate.py` on the next `full` checkpoint that gets close, so the
result is comparable with the stages that have been gated.

## Constraints that any continuation must respect

1. **Never edit a pinned source while a run is active.** This has already cost one run
   (`records/incident-v16b-01.json`).
2. **The first 351 782 bytes of `tasks/mdp.py` must stay byte-identical.** Anything that
   needs new behaviour goes after that boundary, or into the bridge env cfg.
3. **Do not loosen the valid-glide predicate or the 256 × 2 × 80 % gate**, including when
   they fail.
4. **A source change needs a declared transition kind and a record.** A plain continuation
   refuses any change; `actor_forward_transition`, `reward_forward_transition` and
   `task_contract_transition` each name an owner that must move, and non-actor kinds may
   never move the actor.
5. **Do not promote these results into the main curriculum state.** The main branch is
   untouched at waist-balance-100 / iteration 16399.
6. **Assisted stages never certify a from-rest result.** The `initial_assistance` flag is
   checked by the gate for exactly this reason.

## What has not been done at all

* No full-sequence visual review of a complete from-rest → glide rollout.
* No real-robot validation of anything in this repository.
* No promotion of any policy to the deployable set; the project's publish path only accepts
  constant-command episodic/perpetual policies, and this is a phase-conditioned policy.
