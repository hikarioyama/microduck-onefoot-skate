#!/usr/bin/env bash
# Generic chained curriculum block for the retained-skill bridge.
#   usage: run-bridge-extension-03.sh <source-state> <tag> <stage> [<stage>...]
# Each stage gets its own verified same-stage 64x5 smoke from the current source
# checkpoint, then a bounded 1000-update block on the assigned GPU. The chain
# stops on the first stage that does not pass the formal two-seed 256-episode
# gate, and records the result in course.json. Main state is never promoted.
set -euo pipefail
# This work is an overlay on the upstream working tree, so the script runs from
# the repository root. Point MICRODUCK_ROOT at your checkout if you are not there.
cd "${MICRODUCK_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${MICRODUCK_CUDA:-2} OMP_NUM_THREADS=4
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
export BRIDGE_COURSE_PID=$$
if [ "$#" -lt 3 ]; then echo 'usage: <source-state> <tag> <stage>...' >&2; exit 2; fi
base=local/onefoot-bridge-curriculum
source_state="$1"; shift
tag="$1"; shift
stages=("$@")
budget=1000

COURSE_OWNED=0
finish_course() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$COURSE_OWNED" = 1 ]; then
    python3 - "$rc" <<'PY'
import json,sys
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json')
s=json.loads(p.read_text());s.update(status='error',exit_code=int(sys.argv[1]))
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY
  fi
}
trap finish_course EXIT

for stage in "${stages[@]}"; do
  smoke="$base/${stage}-${tag}-smoke"
  run="$base/${stage}-${tag}-training"
  smoke_log="$base/${stage}-${tag}-smoke.log"
  train_log="$base/${stage}-${tag}-training.log"
  transition=()
  if [ -n "${BRIDGE_ACTOR_TRANSITION:-}" ]; then
    transition=(--actor-transition "$BRIDGE_ACTOR_TRANSITION")
  fi
  if [ ! -d "$smoke" ]; then
    uv run --locked python local/train-credit-bridge.py --resume-state "$source_state" \
      "${transition[@]}" \
      --stage "$stage" --smoke --num-envs 64 --updates 5 --chunk 5 --eval-episodes 64 \
      --output "$smoke" > "$smoke_log" 2>&1 || { tail -60 "$smoke_log"; exit 1; }
    tail -1 "$smoke_log"
  fi
  python3 - "$stage" "$source_state" "$run/state.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json');s=json.loads(p.read_text())
s.update(status='training',stage=sys.argv[1],source_state=sys.argv[2],active_state=sys.argv[3],
         coordinator_pid=int(os.environ['BRIDGE_COURSE_PID']),chain_tag=os.environ.get('BRIDGE_CHAIN_TAG'))
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY
  COURSE_OWNED=1
  printf 'CURRICULUM_STAGE_START %s source=%s budget=%s\n' "$stage" "$source_state" "$budget"
  uv run --locked python local/train-credit-bridge.py --resume-state "$source_state" \
    "${transition[@]}" \
    --stage "$stage" --num-envs 4096 --updates "$budget" --chunk 100 --eval-episodes 64 \
    --exit-on-gate --smoke-proof "$smoke/state.json" --output "$run" > "$train_log" 2>&1 \
    || { tail -100 "$train_log"; exit 1; }
  tail -2 "$train_log"
  # A transition is only valid against the source run it was recorded for.
  unset BRIDGE_ACTOR_TRANSITION

  CUDA_VISIBLE_DEVICES='' uv run --locked python - "$run/state.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
from mjlab_microduck.bridge_recovery import bridge_stage_gate
p=Path('local/onefoot-bridge-curriculum/course.json');course=json.loads(p.read_text())
state_path=Path(sys.argv[1]);s=json.loads(state_path.read_text())
digest=hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()
passed=s['status']=='stage_passed' and bridge_stage_gate(s['stage_gate_reports'],s['stage'],digest)
if s['stage_gate_passed']!=passed:raise RuntimeError('Stage proof mismatch')
course['stage_runs'].append({'stage':s['stage'],'state_path':str(state_path),'status':s['status'],
    'updates':s['updates'],'iteration':s['iteration'],'checkpoint':s['checkpoint'],
    'checkpoint_sha256':digest,'stage_gate_passed':passed})
course['status']='stage_passed' if passed else ('interrupted' if s['status']=='interrupted' else 'review_required')
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(course,indent=2)+'\n');t.replace(p)
print('CURRICULUM_STAGE_FINISHED',json.dumps(course['stage_runs'][-1]),flush=True)
PY
  status=$(python3 -c "import json; print(json.load(open('$base/course.json'))['status'])")
  if [ "$status" != stage_passed ]; then
    printf 'CURRICULUM_REVIEW_REQUIRED %s stage=%s (saved; do not cold-restart)\n' "$status" "$stage"
    exit 0
  fi
  source_state="$run/state.json"
done
python3 - <<'PY'
import json
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json');s=json.loads(p.read_text())
s['status']='full_numeric_gate_passed_visual_review_required'
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
print('CURRICULUM_NUMERIC_GATES_FINISHED; perform full-sequence visual review before declaring success')
PY
