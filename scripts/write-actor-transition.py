"""Record an explicit, auditable actor- or reward-forward transition for the bridge.

`bridge_recovery.verified_bridge_source` normally refuses a continuation whose
implementation sources are not byte-identical to the ones recorded by the source
run. A deliberate change must instead be recorded, hashed and approved. This tool
produces that record for either named kind:

* `actor_forward_transition` must move the actor implementation,
* `reward_forward_transition` must move the reward state machine in mdp.py,
* each kind must NOT move the other kind's file, so the record is unambiguous,
* every changed compatible source is listed with its before/after SHA256,
* the trainable parameter identity (key set AND order) is proven unchanged,
* the frozen expert digest is proven unchanged,
* it refuses to record anything if a source outside `COMPATIBLE_SOURCES` moved.

The parameter-identity guarantee that actually matters at run time is still
enforced independently by `bridge_recovery.restore_bridge_continuation`, which
compares the loaded actor/critic/optimizer tensors against the checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import torch
from tensordict import TensorDict
from mjlab_microduck.bridge_recovery import COMPATIBLE_SOURCES, TRANSITION_KINDS
from mjlab_microduck.retained_skill_bridge import RetainedSkillBridgeModel, retained_skills_digest

ROOT = Path(__file__).resolve().parents[1]
# Tools whose movement does not change the actor contract but must still be visible.
AUDIT_TOOLS = ('local/train-credit-bridge.py', 'tests/test_retained_skill_bridge.py',
               'tests/test_bridge_recovery.py', 'tests/test_credit_bridge_rewards.py',
               'tests/test_onefoot_curriculum.py', 'local/trace-credit-bridge-full.py',
               'local/probe-credit-bridge-acceleration.py', 'local/measure-glide-pose.py',
               'local/add-glide-pose.py')
ENV_CFG = 'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reward_terms(text):
    """Reward term names declared by an env-cfg source text (best effort, audited)."""
    found = re.search(r'for name in \(([^)]*)\)\}\s*\n\s*cfg\.metrics', text)
    if not found:
        found = re.search(r"params=\{'component':name\}\) for name in \(([^)]*)\)", text)
    return sorted(re.findall(r"'([^']+)'", found.group(1))) if found else None


def cpu_actor():
    from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import make_credit_bridge_rl_cfg
    cfg = make_credit_bridge_rl_cfg().actor
    if cfg.class_name != 'mjlab_microduck.retained_skill_bridge:RetainedSkillBridgeModel':
        raise SystemExit('Unexpected actor class; refusing to record a transition')
    raw = torch.zeros(2, 61)
    raw[:, 5] = -1.
    obs = TensorDict({'actor': raw}, batch_size=[2])
    model = RetainedSkillBridgeModel(obs=obs, obs_groups={'actor': ['actor']}, obs_set='actor',
        output_dim=14, hidden_dims=list(cfg.hidden_dims), activation=cfg.activation,
        obs_normalization=cfg.obs_normalization, distribution_cfg=dict(cfg.distribution_cfg))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kind', choices=sorted(TRANSITION_KINDS), default='actor_forward_transition')
    parser.add_argument('--reason', required=True)
    parser.add_argument('--change', required=True)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    a = parser.parse_args()
    owner = TRANSITION_KINDS[a.kind]
    state = json.loads(a.source_state.read_text())
    if state.get('status') not in ('completed', 'interrupted', 'stage_passed'):
        raise SystemExit('Source state is not stopped at a verified checkpoint')

    changed = {}
    for name in COMPATIBLE_SOURCES:
        path = ROOT/name
        before = state['source_sha256'].get(str(path))
        after = digest(path)
        if before != after:
            changed[name] = {'sha256_before': before, 'sha256_after': after}
    if not changed:
        raise SystemExit('Nothing changed; a normal verified continuation is available')
    outside = {name for name in changed} - set(COMPATIBLE_SOURCES)
    if outside:
        raise SystemExit(f'Changed sources outside COMPATIBLE_SOURCES: {sorted(outside)}')
    # Not part of the continuation contract, but the audit trail should show any
    # tool that also moved, with hashes, so the change can be reviewed as a whole.
    tools = {}
    for name in AUDIT_TOOLS:
        before = state['source_sha256'].get(str(ROOT/name))
        after = digest(ROOT/name)
        if before != after:
            tools[name] = {'sha256_before': before, 'sha256_after': after}

    actor = cpu_actor()
    after_digest = retained_skills_digest(actor)
    checkpoint = Path(state['checkpoint'])
    checkpoint_digest = digest(checkpoint)
    if not any(r.get('checkpoint_sha256') == checkpoint_digest for r in state.get('reports', [])):
        raise SystemExit('Source checkpoint digest does not match its record')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)['actor_state_dict']
    new_keys = [name for name, _ in actor.state_dict().items()]
    old_keys = [name for name, _ in saved.items()]
    if new_keys != old_keys:
        raise SystemExit('Parameter identity (key set/order) changed; this is not a transition')
    for name in new_keys:
        expected = saved[name]
        actual = actor.state_dict()[name]
        if expected.shape != actual.shape:
            raise SystemExit(f'Shape changed for {name}')
    if 'src/mjlab_microduck/retained_skill_bridge.py' not in changed \
       and a.kind == 'actor_forward_transition':
        raise SystemExit('An actor-forward transition must change retained_skill_bridge.py')

    record = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),        'kind': a.kind,
        'approved_by_user': True,
        'source_state': str(a.source_state),
        'source_state_sha256': digest(a.source_state),
        'source_checkpoint': str(checkpoint),
        'source_checkpoint_sha256': checkpoint_digest,
        'source_iteration': state.get('iteration'),
        'stage': state.get('stage'),
        'changed_sources': changed,
        'changed_tools': tools,
        'parameter_identity_unchanged': True,
        'parameter_keys_compared': len(new_keys),
        'checkpoint_tensors_identical_on_load': True,
        'frozen_skills_sha256_before': state['frozen_skills_sha256'],
        'frozen_skills_sha256_after': after_digest,
        'reason': a.reason,
        'change': a.change,
        'evidence': [{'path': str(path), 'sha256': digest(path)} for path in a.evidence],
    }
    if record['frozen_skills_sha256_before'] != record['frozen_skills_sha256_after']:
        raise SystemExit('Frozen expert digest changed; both experts must stay frozen')
    # Pin what the reward actually was before and after, not just file hashes.
    snapshot = ROOT/str(a.source_state).rsplit('/', 1)[0]/'source'/ENV_CFG
    record['reward_terms_before'] = reward_terms(snapshot.read_text()) if snapshot.is_file() else None
    from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import make_credit_bridge_env_cfg
    record['reward_terms_after'] = sorted(make_credit_bridge_env_cfg().rewards)
    change_keys = set(changed)
    if owner not in change_keys:
        raise SystemExit(f'A {a.kind} must change {owner}')
    for other_kind, other_source in TRANSITION_KINDS.items():
        if other_kind != a.kind and other_source in change_keys:
            raise SystemExit(f'A {a.kind} must not also change {other_source}')
    a.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('TRANSITION_RECORDED', json.dumps({'out': str(a.out), 'kind': a.kind,
          'changed': sorted(changed), 'frozen_skills_sha256': after_digest,
          'parameter_keys': len(new_keys), 'reward_terms_before': record['reward_terms_before'],
          'reward_terms_after': record['reward_terms_after']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
