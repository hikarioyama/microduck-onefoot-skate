#!/usr/bin/env bash
# Bounded unload continuation 02.
# Source: unload-extension-01-training (model_1599, interrupted by an external
# shell restart, not a numeric failure). Purpose: measure whether the credible
# 0.5 s rate is still improving before changing the curriculum granularity.
# Preserves every earlier checkpoint and never promotes into the main state.
set -euo pipefail
# This work is an overlay on the upstream working tree, so the script runs from
# the repository root. Point MICRODUCK_ROOT at your checkout if you are not there.
cd "${MICRODUCK_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${MICRODUCK_CUDA:-2} OMP_NUM_THREADS=4
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
export BRIDGE_COURSE_PID=$$

base=local/onefoot-bridge-curriculum
source_state=$base/unload-extension-01-training/state.json
smoke=$base/unload-extension-02-smoke
run=$base/unload-extension-02-training
budget=1000
stage=unload

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

# Record the true state of the killed extension-01 block before touching anything.
python3 - "$source_state" <<'PY'
import json,sys
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json')
s=json.loads(p.read_text())
if not any(r.get('stage_path')=='unload-extension-01' for r in s.get('stage_runs',[])+s.get('interrupted_runs',[])):
    s.setdefault('interrupted_runs',[]).append({
        'stage_path':'unload-extension-01',
        'state_path':sys.argv[1],
        'status':'interrupted',
        'cause':'external_shell_restart_killed_trainer',
        'incident_record':'local/onefoot-bridge-curriculum/incident-unload-extension-01.json',
        'updates':300,'iteration':1599,
        'checkpoint':str(Path('logs/rsl_rl/onefoot/2026-09-10_17-28-14_credit-bridge-unload-pilot/model_1599.pt').resolve()),
        'checkpoint_sha256':'119a25723ca49187349299d223abdc829f6a9c763c715e4cc32432eae8b4622b',
        'credible_050_trend':[0.5469,0.5469,0.5938,0.6094],'not_a_numeric_failure':True})
s.update(status='starting_extension_02',coordinator_pid=int(__import__('os').environ['BRIDGE_COURSE_PID']))
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY

# 1) Verified same-stage, same-source 64x5 smoke for the new continuation.
smoke_log=$base/unload-extension-02-smoke.log
uv run --locked python local/train-credit-bridge.py --resume-state "$source_state" \
  --stage "$stage" --smoke --num-envs 64 --updates 5 --chunk 5 --eval-episodes 64 \
  --output "$smoke" > "$smoke_log" 2>&1 || { tail -60 "$smoke_log"; exit 1; }
tail -1 "$smoke_log"
python3 - "$smoke/state.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['status']=='completed' and s['exports_verified'] and s['stage']=='unload', s['status']
print('SMOKE_OK',s['source_checkpoint_sha256'][:16])
PY

# 2) Publish the new active state so the browser follows the running trainer.
python3 - "$source_state" "$run/state.json" <<'PY'
import json,sys
from pathlib import Path
p=Path('local/onefoot-bridge-curriculum/course.json');s=json.loads(p.read_text())
s.update(status='training',stage='unload',source_state=sys.argv[1],active_state=sys.argv[2],
         extension_budget=1000,
         extension_reason='Credible 0.5 s rate rose 26.6%->60.9% over 1300 updates while the mean credible glide stayed ~0.51 s. Continue the saved branch for a bounded 1000-update block to test whether the rate is still improving.')
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PY
COURSE_OWNED=1
printf 'CURRICULUM_STAGE_START %s source=%s budget=%s\n' "$stage" "$source_state" "$budget"

# 3) Bounded training block on the assigned GPU.
train_log=$base/unload-extension-02-training.log
uv run --locked python local/train-credit-bridge.py --resume-state "$source_state" \
  --stage "$stage" --num-envs 4096 --updates "$budget" --chunk 100 --eval-episodes 64 \
  --exit-on-gate --smoke-proof "$smoke/state.json" --output "$run" > "$train_log" 2>&1 || { tail -100 "$train_log"; exit 1; }
tail -2 "$train_log"

# 4) Record the stage result with the verified gate proof.
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
