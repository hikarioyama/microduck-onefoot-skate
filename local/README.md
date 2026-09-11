# `local/` — the tools and data the tests load

This directory looks unusual, and it exists for a specific reason.

## Why it is here

Thirteen of the 24 test modules do not `import` a module — they load a **file by path**:

```python
spec = importlib.util.spec_from_file_location('bridge_transfer_timeline_test',
                                              root/'local/record-retained-bridge-transfer.py')
```

That is upstream's own convention, not something introduced here: upstream's
`tests/test_onefoot_knee.py` loads `local/run-onefoot-knee-pilot.py` the same way. The tools
are loaded as source, never imported as a package, so the file has to be at that path.

Upstream maintains `local/` as a **working directory that git does not track**. At the commit
this work derives from it holds 281 files and `git ls-files local` returns nothing, so
upstream does not publish any of it. Two consequences:

1. The tools below are **not available anywhere else**. This repository is currently their
   only public home.
2. Cloning this repository without them makes the suite fail during collection, not at some
   convenient later point. That is the 42-failure number quoted in the top-level README.

So the needed files are vendored here in full, at the exact paths the tests expect, rather
than being stubbed or replaced by a description.

## What is here

**Tools loaded by the test suite** (19 files: 18 `.py` and one `.html`, ~185 KB total):

| File | Loaded by |
|---|---|
| `train-onefoot-curriculum.py` | `test_onefoot_supervision.py` (19 call sites) |
| `watch-onefoot-training.py` | `test_onefoot_supervision.py` |
| `wait-onefoot-review.py` | `test_onefoot_supervision.py` |
| `run-history-comparison.py` | `test_onefoot_supervision.py` |
| `tensorboard-evaluations.py` | `test_onefoot_supervision.py`, `test_tensorboard_evaluations.py` |
| `record-onefoot-timeline.py` | `test_onefoot_timeline.py` |
| `summarize-onefoot-knee.py` | `test_onefoot_timeline.py` |
| `record-retained-bridge-transfer.py` | `test_bridge_transfer_timeline.py` |
| `evaluate-public-roller-connection.py` | `test_credit_bridge_rewards.py`, `test_public_roller_policy.py` |
| `train-dynamic-onefoot.py` | `test_onefoot_dynamic.py` |
| `play-standard-loopback.py` | `test_onefoot_dynamic.py` |
| `onefoot-dashboard.py`, `onefoot-dashboard.html` | `test_onefoot_dynamic.py` |
| `run-onefoot-knee-pilot.py` | `test_onefoot_knee.py` |
| `preserve-onefoot-branch-review.py` | `test_onefoot_branch_review.py` |
| `play-latest-onefoot.py` | `test_onefoot_performance.py` |
| `onefoot_live_source.py` | `test_live_onefoot_viewer.py` |
| `verify-retained-skill-bridge.py` | loaded by `record-retained-bridge-transfer.py` |

**Measured data**, read directly by tests rather than hard-coded:

* `onefoot-bridge-curriculum/glide-pose-measurement.json` — the glide-entry pose, captured
  across 64 real rollouts. `test_credit_bridge_rewards.py` and `test_onefoot_curriculum.py`
  assert the spawn pose against it, so the pose and the measurement cannot drift apart.
* `onefoot-reward-review/source-before.json` — the SHA256 and byte length of the first
  351 782 bytes of `tasks/mdp.py` and of the pinned registry, taken before the v15 reward
  work. `test_credit_bridge_rewards.py::test_old_mdp_and_registry_bytes_are_preserved`
  hashes the shipped files against it to prove the legacy region is byte-identical. Without
  this file the test skips and that invariant goes unchecked.
* `onefoot-public-acceleration/assets/088524a64e2557dc453256b6071dbb9d23888802/roller.onnx` —
  the pinned public roller expert, at the content-addressed path the code records. It is
  **byte-identical** to `checkpoints/frozen/public-roller-expert.onnx`; both are needed, one
  because `public_residual_policy.py` hashes that exact path, the other because it is the
  readable copy in `checkpoints/`.

**Duplicates are intentional.** `onefoot_live_source.py` and
`record-retained-bridge-transfer.py` also exist in `scripts/`, which is the flat copy of the
tools developed here. The copies in `local/` are what the tests load; both are byte-identical
(`sha256sum local/onefoot_live_source.py scripts/onefoot_live_source.py`).

**Not here.** The 526 MB `onefoot-bridge-curriculum/` experiment tree, the 103 MB
`performance/` traces and the rest of upstream's `local/` are not vendored: they are training
output, not tooling, and the parts that matter are already extracted into `records/` and
`trajectories/` as machine-readable JSON and CSV.

## What these tools do

They are the experiment harness, not the environment. The environment lives in
`src/mjlab_microduck/tasks/`.

* `train-onefoot-curriculum.py` — the bounded, chunked curriculum trainer. It writes a state
  file, refuses to exceed its declared budget, and stops at a checkpoint on a stop-file
  request rather than mid-chunk.
* `train-dynamic-onefoot.py`, `run-onefoot-knee-pilot.py`, `run-history-comparison.py` —
  recorded pilots that reuse the bounded recipe for a different stage or recipe.
* `watch-onefoot-training.py`, `wait-onefoot-review.py` — supervision around the trainer:
  health monitoring, review gating, and a notification pass that never promotes a candidate
  on its own.
* `tensorboard-evaluations.py` — backfills evaluation scalars into TensorBoard, refusing to
  publish mismatched evidence.
* `record-onefoot-timeline.py`, `record-retained-bridge-transfer.py`,
  `summarize-onefoot-knee.py` — reduce rollouts to JSON summaries with explicit censoring, so
  an early termination is never counted as a success or as a zero.
* `preserve-onefoot-branch-review.py` — the branch-boundary and stop-request protocol.
* `evaluate-public-roller-connection.py` — the reference roller-only baseline; the bridge's
  physics, rewards and zero-velocity start are asserted against the environment it builds.
* `measure-glide-pose.py`, `add-glide-pose.py`, `solve-onefoot-curriculum-poses.py` — the pose
  pipeline that produced the measured glide entry.
* `verify-retained-skill-bridge.py` — the original verification timeline the credit-bridge
  work reuses.
* `onefoot_live_source.py` — resolves a live viewer to the newest training state without
  reading hardware telemetry.
* `play-standard-loopback.py`, `onefoot-dashboard.py` — launch helpers bound to
  `127.0.0.1`; the dashboard only redirects to the standard viewer. No unpublished
  dashboard is part of this work.

## Provenance

Copied verbatim from `pollen-robotics/microduck_rl` at
`53b8971b61baf5b7f3c16d135dd7cac37623de4b`, from the untracked working directory. No file
here has been edited for publication, with one exception recorded below.

**Edit:** three test modules gained a `pytest.skip` guard for the robot XML
(`test_onefoot_knee.py`, `test_onefoot_curriculum.py`, `test_onefoot_lean.py`). Those are in
`tests/`, not here, and the guard only fires when upstream's `src/` is absent — it never
changes a result when the overlay is complete. See `docs/REPRODUCING.md`.
