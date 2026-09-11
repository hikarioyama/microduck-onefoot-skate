#!/usr/bin/env bash
# Reviewed finite stage blocks; preserve every parent checkpoint and optimizer.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
export BRIDGE_COURSE_PID=$$
BRIDGE_COURSE_OWNED=0
mark_course_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$BRIDGE_COURSE_OWNED" = 1 ]; then
    python3 - "$rc" <<'PY'
import json,sys
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json')
if p.exists():
    s=json.loads(p.read_text());s.update(status='error',exit_code=int(sys.argv[1]))
    t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY
  fi
}
trap mark_course_exit EXIT
base='local/onefoot-bridge-curriculum'
source_state='local/onefoot-bridge-curriculum/unload-training/state.json'
python3 - <<'PY'
from datetime import datetime,timezone
import json,os
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json')
s=json.loads(p.read_text())
assert s['status']=='review_required' and s['stage']=='unload'
assert not Path(f"/proc/{s['coordinator_pid']}").exists()
s.update(status='starting',coordinator_pid=int(os.environ['BRIDGE_COURSE_PID']),
    resumed_at=datetime.now(timezone.utc).isoformat(),extension_budget=500,
    extension_reason='At 1000 updates credible mean reached .517 s versus .486 s at 600; continue the saved improving branch for a bounded 500-update review.')
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY
BRIDGE_COURSE_OWNED=1
for stage in unload transfer-near transfer full; do
  if [ "$stage" = unload ]; then budget=500; else budget=1000; fi
  if [ "$stage" = near ]; then
    smoke="$base/near-smoke/state.json"
  else
    smoke="$base/${stage}-extension-01-smoke/state.json"
    uv run --locked python local/train-credit-bridge.py --resume-state "$source_state" \
      --stage "$stage" --smoke --num-envs 64 --updates 5 --chunk 5 --eval-episodes 64 \
      --output "$base/${stage}-extension-01-smoke" > "$base/${stage}-extension-01-smoke.log" 2>&1 || { tail -80 "$base/${stage}-extension-01-smoke.log"; exit 1; }
    tail -1 "$base/${stage}-extension-01-smoke.log"
  fi
  run="$base/${stage}-extension-01-training"
  python3 - "$stage" "$source_state" "$run/state.json" <<'PY'
import json,sys
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json');s=json.loads(p.read_text())
s.update(status='training',stage=sys.argv[1],source_state=sys.argv[2],active_state=sys.argv[3])
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY
  printf '\nCURRICULUM_STAGE_START %s source=%s\n' "$stage" "$source_state"
  uv run --locked python local/train-credit-bridge.py --resume-state "$source_state" \
    --stage "$stage" --num-envs 4096 --updates "$budget" --chunk 100 --eval-episodes 64 \
    --exit-on-gate --smoke-proof "$smoke" --output "$run" > "$run.log" 2>&1 || { tail -100 "$run.log"; exit 1; }
  tail -2 "$run.log"
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
