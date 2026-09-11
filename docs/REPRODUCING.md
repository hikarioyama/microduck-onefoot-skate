# Reproducing this work

This repository is an **overlay on the upstream working tree**, not a standalone package.
That is a deliberate choice, and it is the single most important thing to understand before
you try to run anything.

* `src/` here contains only the **24 files this work added or changed**. The other ~230
  files of the `mjlab_microduck` package — including the robot model and its 23 MB of
  meshes, which three tests load — come from upstream.
* `tests/` here contains **24 new test modules**. Upstream ships 20 more. The upstream ones
  are **not** duplicated here; they still run in the overlay.
* `scripts/` here is a flat copy of the tools that were developed in the upstream `local/`
  directory. `local/` in this repository is the minimal data and tool set those tests need.
* `logs/` holds exactly one checkpoint. It is the frozen one-foot expert, at the path the
  pinned code records.

**Consequence:** running `pytest` in a bare clone of this repository does not work. Upstream
must be present underneath it. Use the recipe below.

## Verified recipe

```bash
# 1. Upstream, at the commit this work is derived from.
git clone https://github.com/pollen-robotics/microduck_rl.git microduck_rl
cd microduck_rl
git checkout 53b8971b61baf5b7f3c16d135dd7cac37623de4b

# 2. Overlay this repository's files on top. src/ is merged, not replaced: the
#    deliverable provides its 24 files, upstream provides everything else.
DELIVERABLE=/path/to/microduck-onefoot-skate
cp -r "$DELIVERABLE/src/."  src/
cp -r "$DELIVERABLE/tests/." tests/
cp -r "$DELIVERABLE/local" local
cp -r "$DELIVERABLE/logs"  logs

# 3. Run. `uv run --locked` builds the venv from upstream's pyproject.toml / uv.lock,
#    so the interpreter and the dependencies are the reviewed ones.
CUDA_VISIBLE_DEVICES='' uv run --locked --with pytest python -m pytest tests/ -q
```

Measured on a clean clone of upstream `53b8971` with this overlay:

```
434 passed, 1 skipped, 15 warnings in 23.60s
```

The single skip is `tests/test_aarch64_cuda_torch.py`, which is upstream's and is
conditional on running on `linux-aarch64` with a GPU. Nothing in this work is skipped when
the overlay is present: the robot model is found, the frozen expert is found, and the
measured data files are found.

### Without the overlay

For contrast, and because it is the number that matters for a reader who only clones this
repository: a bare clone of this repository alone gives

```
42 failed, 180 passed, 3 skipped, 1 error
```

The failures are not subtle. 39 of them are `FileNotFoundError` on
`local/*.py` tools, and the rest are `ValueError: ParseXML: Error opening
.../scene_rollers.xml`. Both are cured by the overlay, and nothing else is.

## Environment

| Item | Value |
|---|---|
| Python | 3.12 |
| `mjlab` | 1.3.0 |
| `mujoco` | 3.10.0 |
| `mujoco-warp` | 3.8.1 |
| `rsl-rl-lib` | 5.0.1 |
| `torch` | 2.9.1+cu129 |
| TensorBoard | 2.20.0 (only for the trajectories in `trajectories/`) |

The suite is **CPU-only**. Training is not: every recorded run used one RTX 5070 Ti, and
`docs/METHOD.md` §6 lists the settings.

## Checkpoints are git-lfs

`.pt` and `.onnx` are stored with LFS. A fresh clone gets 132-byte pointer files until LFS
is active, and the failure mode is a confusing `torch.load` error rather than an obvious
one:

```bash
git lfs install      # once per machine
git lfs pull         # in an existing clone that was made before installing LFS
```

Verify before trusting any number:

```bash
sha256sum checkpoints/glide/model_10499.pt
# 8b46d6fa28c55df1a722a92dc35d176113438612633665c5f52291f6fecc69a1
```

## Running the shipped policies

`checkpoints/*.onnx` need neither mjlab nor MuJoCo — only `numpy` and `onnxruntime`. See
[ONNX-INFERENCE.md](ONNX-INFERENCE.md) for the exact 61-dim observation layout, the action
convention, and a working script.

```bash
uv run --with onnxruntime --with numpy --with onnx \
    python scripts/policy_inference.py checkpoints/glide/policy-glide-02.onnx
```

## Training trajectories

`trajectories/` holds the learning curves of all 15 runs. They are plain CSV, one row per
update, and the screening evaluations the gate and the report judge are in
`trajectories/screening.json`. `trajectories/README.md` has the column glossary.

They are exported from the raw TensorBoard logs (377 event files, 155 MB) by
`scripts/export-training-trajectories.py`, which needs those logs, i.e. the upstream working
tree with its `logs/` directory intact. The exported CSV is what makes the curves readable
without them.

## Why the tests need `local/`

Thirteen of the 24 test modules load a tool by file path,
`importlib.util.spec_from_file_location(..., root/'local/<tool>.py')`, and two read a
measurement file. This is upstream's own convention — its `tests/test_onefoot_knee.py` does
the same — so `local/` is loaded as source, not imported as a package.

Upstream's `local/` directory is **not tracked by git** (281 files, `git ls-files local` is
empty), so upstream does not ship these tools at all. This repository is currently the only
place they are published, which is why `local/` is included in full rather than as a stub.
