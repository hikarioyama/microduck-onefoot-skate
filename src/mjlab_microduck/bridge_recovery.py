"""Audited continuation for the retained-skill bridge, not main-stage promotion."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import torch
from .checkpoint_safety import safe_runner_load
from .retained_skill_bridge import retained_skills_digest

# These define parameter identity/order, the transition policy itself and the
# reward state machine, so all of them are pinned to the source run's bytes.
# Curriculum cfg/trainer code may change with an explicitly recorded transition.
COMPATIBLE_SOURCES = (
    'src/mjlab_microduck/retained_skill_bridge.py',
    'src/mjlab_microduck/public_residual_policy.py',
    'src/mjlab_microduck/checkpoint_safety.py',
    'src/mjlab_microduck/bridge_recovery.py',
    'src/mjlab_microduck/tasks/mdp.py',
    'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py',
    'src/mjlab_microduck/tasks/microduck_onefoot_curriculum_poses.py',
)

# A deliberate, recorded change to those sources is allowed in exactly three named
# forms. Each form must move its own owner file and must NOT move another form's
# owner, so a record always states unambiguously what changed: the actor's forward
# semantics, the reward/objective state machine, or the task contract (which stages
# exist, where they spawn, and how long they are certified for). In every case the
# parameter identity and the frozen expert digest must be unchanged, and the real
# load-time guarantee is still `restore_bridge_continuation`.
TRANSITION_KINDS = {
    'actor_forward_transition': 'src/mjlab_microduck/retained_skill_bridge.py',
    'reward_forward_transition': 'src/mjlab_microduck/tasks/mdp.py',
    'task_contract_transition': 'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py',
}


def _verified_state(state, state_path, root):
    if state.get('status') not in ('completed','interrupted','stage_passed'):
        raise ValueError('Source training must have stopped at a verified checkpoint')
    if state.get('learning_rate_schedule')!='fixed' or state.get('learning_rate')!=1e-4:
        raise ValueError('Expected the reviewed fixed learning rate')
    checkpoint=Path(state['checkpoint']).resolve()
    if not checkpoint.is_relative_to(root/'logs/rsl_rl/onefoot') or not checkpoint.is_file():
        raise ValueError('Checkpoint is outside the onefoot experiment tree')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reports=[r for r in state.get('reports',[]) if Path(r['checkpoint']).resolve()==checkpoint
             and r.get('checkpoint_sha256')==digest and r.get('nan_episodes')==0]
    if not reports:
        raise ValueError('Checkpoint lacks a matching finite evaluation/hash')
    if hashlib.sha256((root/'local/onefoot-curriculum-state.json').read_bytes()).hexdigest()!=state['main_state_sha256']:
        raise ValueError('Main state changed since the source run')
    for name in COMPATIBLE_SOURCES:
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=state['source_sha256'].get(str(path)):
            raise ValueError(f'Incompatible continuation source: {name}')
    return dict(state_path=str(state_path),state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                checkpoint=str(checkpoint),checkpoint_sha256=digest,stage=state['stage'],
                frozen_skills_sha256=state['frozen_skills_sha256'],learning_rate=state['learning_rate'])


def verified_bridge_source(state_path, root):
    root=Path(root).resolve();state_path=Path(state_path).resolve()
    if not state_path.is_relative_to(root/'local'):
        raise ValueError('Bridge source state must be a local experiment record')
    return _verified_state(json.loads(state_path.read_text()),state_path,root)


def transition_violation(kind, changed):
    """Which tracked change violates the named-kind contract, or None.

    The actor gets the strictest treatment, because parameter identity and the
    frozen-expert digest are the guarantees a continuation actually relies on: a
    record may change the actor ONLY if it is declared an actor-forward transition,
    and an actor-forward transition may change nothing else. Every other file is
    environment/task machinery that the record must list, with hashes, and explain
    in its change text. Any non-actor kind still has to move its own owner file, so
    the kind label can never be vacuous.
    """
    owner=TRANSITION_KINDS[kind]
    actor=TRANSITION_KINDS['actor_forward_transition']
    if owner not in changed:
        return f'A {kind} must change {owner}'
    if kind=='actor_forward_transition':
        others=sorted(set(changed)-{actor})
        if others:
            return f'An actor-forward transition must not also change {others}'
    elif actor in changed:
        return f'A {kind} must not change the actor implementation'
    return None


def verified_bridge_transition(state_path, root, transition_path):
    """Continue across a recorded, approved actor- or reward-forward transition.

    A normal continuation demands byte-identical compatible sources. A deliberate
    change is allowed instead when it is recorded with before/after hashes of every
    source it touches, names its kind, and is approved. Everything else is still
    verified exactly as before, and the parameter-identity guarantee is still
    enforced at load time by `restore_bridge_continuation`.
    """
    root=Path(root).resolve();state_path=Path(state_path).resolve()
    if not state_path.is_relative_to(root/'local'):
        raise ValueError('Bridge source state must be a local experiment record')
    transition_path=Path(transition_path).resolve()
    transition=json.loads(transition_path.read_text())
    kind=transition.get('kind');owner=TRANSITION_KINDS.get(kind)
    if owner is None or transition.get('approved_by_user') is not True:
        raise ValueError('Transition must be an approved, known transition kind')
    if not transition.get('reason') or not transition.get('change'):
        raise ValueError('Transition must state both its reason and its change')
    if transition.get('parameter_identity_unchanged') is not True \
       or transition.get('checkpoint_tensors_identical_on_load') is not True:
        raise ValueError('Transition must assert unchanged parameter identity')
    if transition.get('frozen_skills_sha256_before')!=transition.get('frozen_skills_sha256_after'):
        raise ValueError('Transition must not change the frozen expert digest')
    changed=transition.get('changed_sources') or {}
    if not changed or not set(changed).issubset(COMPATIBLE_SOURCES):
        raise ValueError('Transition must only change declared compatible sources')
    violation=transition_violation(kind,changed)
    if violation:
        raise ValueError(violation)
    state=json.loads(state_path.read_text())
    patched=deepcopy(state)
    for name,record in changed.items():
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record.get('sha256_after'):
            raise ValueError(f'Transition target changed after recording: {name}')
        if state.get('source_sha256',{}).get(str(path))!=record.get('sha256_before'):
            raise ValueError(f'Transition baseline is not the source run: {name}')
        patched['source_sha256'][str(path)]=record['sha256_after']
    evidence=_verified_state(patched,state_path,root)
    evidence.update(actor_transition=str(transition_path),
                    actor_transition_sha256=hashlib.sha256(transition_path.read_bytes()).hexdigest(),
                    transition_kind=kind,
                    transitioned_sources=sorted(changed))
    return evidence


def assert_tree_equal(actual,expected,path='state'):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.dtype!=expected.dtype or actual.shape!=expected.shape:
            raise ValueError(f'Tensor mismatch at {path}')
        if not torch.equal(actual.detach().cpu(),expected.detach().cpu()):
            raise ValueError(f'Tensor values changed at {path}')
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():
            raise ValueError(f'Keys changed at {path}')
        for key in expected:assert_tree_equal(actual[key],expected[key],f'{path}/{key}')
    elif isinstance(expected,(tuple,list)):
        if type(actual)!=type(expected) or len(actual)!=len(expected):
            raise ValueError(f'Sequence changed at {path}')
        for i,(a,b) in enumerate(zip(actual,expected)):assert_tree_equal(a,b,f'{path}/{i}')
    elif actual!=expected:
        raise ValueError(f'Value changed at {path}')


def restore_bridge_continuation(runner,evidence,device):
    path=Path(evidence['checkpoint'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed before restore')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    safe_runner_load(runner,path,strict=True,map_location=device)
    assert_tree_equal(runner.alg.actor.state_dict(),saved['actor_state_dict'],'actor')
    assert_tree_equal(runner.alg.critic.state_dict(),saved['critic_state_dict'],'critic')
    assert_tree_equal(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'],'optimizer')
    if retained_skills_digest(runner.alg.actor)!=evidence['frozen_skills_sha256']:
        raise ValueError('Retained expert state differs from the source run')
    rates=[group['lr'] for group in runner.alg.optimizer.param_groups]
    if runner.alg.learning_rate!=evidence['learning_rate'] or any(lr!=evidence['learning_rate'] for lr in rates):
        raise ValueError('Restored PPO/Adam learning rates disagree')
    if int(runner.current_learning_iteration)!=int(saved['iter']):
        raise ValueError('Runner iteration was not restored')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed during restore')
    return dict(source_iteration=int(saved['iter']),next_iteration=int(saved['iter'])+1,
                actor_exact=True,critic_exact=True,normalizers_and_std_exact=True,
                optimizer_exact=True,optimizer_states=len(saved['optimizer_state_dict']['state']),
                learning_rate=runner.alg.learning_rate,frozen_skills_sha256=evidence['frozen_skills_sha256'])


def bridge_stage_gate(reports,stage,checkpoint_sha256):
    """Two independent 256-first-episode checks, separate from the main gates."""
    from .tasks.microduck_onefoot_credit_bridge_env_cfg import SUSTAINED_STAGES
    duration='200' if stage in SUSTAINED_STAGES else '050'
    if len(reports)!=2 or len({r.get('seed') for r in reports})!=2:
        return False
    return all(r.get('stage')==stage and r.get('checkpoint_sha256')==checkpoint_sha256
        and r.get('episodes',0)>=256 and r.get('nan_episodes')==0
        and r.get('initial_assistance')==(stage!='full')
        and r.get(f'success_{duration}_rate',0)>=.80
        and r.get(f'credible_{duration}_rate',0)>=.80
        and r.get('discounted_shaping_max_abs',1)<2e-5 for r in reports)
