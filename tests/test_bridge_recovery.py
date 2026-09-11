import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from mjlab_microduck.bridge_recovery import (
    COMPATIBLE_SOURCES, assert_tree_equal, bridge_stage_gate,
    restore_bridge_continuation, verified_bridge_source, verified_bridge_transition,
)
from mjlab_microduck.retained_skill_bridge import retained_skills_digest


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std=torch.nn.Parameter(torch.ones(2)*-.7)
        self.register_buffer('normalizer_mean',torch.tensor([.2,.5]))
        self.linear=torch.nn.Linear(2,2)
        self.public=torch.nn.Linear(2,2).requires_grad_(False)
        self.onefoot=torch.nn.Linear(2,2).requires_grad_(False)

    def forward(self,x):return self.linear(x-self.normalizer_mean)+self.log_std*.01


class Runner:
    def __init__(self):
        self.alg=SimpleNamespace(actor=Actor(),critic=torch.nn.Linear(2,1),learning_rate=9e-4)
        self.alg.optimizer=torch.optim.Adam(list(self.alg.actor.parameters())+list(self.alg.critic.parameters()),lr=1e-4)
        self.current_learning_iteration=0

    def load(self,path,**kwargs):
        data=torch.load(path,weights_only=True,map_location=kwargs.get('map_location','cpu'))
        self.alg.actor.load_state_dict(data['actor_state_dict'])
        self.alg.critic.load_state_dict(data['critic_state_dict'])
        self.alg.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.current_learning_iteration=data['iter']


def update(runner):
    x=torch.tensor([[.1,.8],[.4,-.2]])
    runner.alg.optimizer.zero_grad()
    loss=runner.alg.actor(x).square().mean()+runner.alg.critic(x).square().mean()
    loss.backward();runner.alg.optimizer.step()


def test_exact_restore_preserves_moments_normalizer_std_and_next_update(tmp_path):
    torch.manual_seed(1);source=Runner()
    for _ in range(3):update(source)
    data={'actor_state_dict':source.alg.actor.state_dict(),'critic_state_dict':source.alg.critic.state_dict(),
          'optimizer_state_dict':source.alg.optimizer.state_dict(),'iter':299}
    path=tmp_path/'model_299.pt';torch.save(data,path)
    evidence={'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'frozen_skills_sha256':retained_skills_digest(source.alg.actor),'learning_rate':1e-4}
    target=Runner()
    with torch.inference_mode():target.alg.actor.normalizer_mean=torch.zeros(2)
    result=restore_bridge_continuation(target,evidence,'cpu')
    assert result['source_iteration']==299 and result['next_iteration']==300
    assert result['optimizer_states']>0 and result['optimizer_exact']
    assert target.alg.learning_rate==1e-4
    assert_tree_equal(target.alg.actor.state_dict(),source.alg.actor.state_dict())
    update(source);update(target)
    assert_tree_equal(target.alg.actor.state_dict(),source.alg.actor.state_dict())
    assert_tree_equal(target.alg.critic.state_dict(),source.alg.critic.state_dict())
    assert_tree_equal(target.alg.optimizer.state_dict(),source.alg.optimizer.state_dict())
    assert retained_skills_digest(target.alg.actor)==evidence['frozen_skills_sha256']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):restore_bridge_continuation(target,evidence,'cpu')


def source_record(root):
    main=root/'local/onefoot-curriculum-state.json';main.parent.mkdir(parents=True);main.write_text('{}')
    checkpoint=root/'logs/rsl_rl/onefoot/run/model_299.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint')
    sources={}
    for name in COMPATIBLE_SOURCES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    data={'status':'completed','stage':'full','learning_rate_schedule':'fixed','learning_rate':1e-4,
          'checkpoint':str(checkpoint),'reports':[{'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'nan_episodes':0}],
          'main_state_sha256':hashlib.sha256(main.read_bytes()).hexdigest(),
          'source_sha256':sources,'frozen_skills_sha256':'expert-hash'}
    path=root/'local/parent-state.json';path.write_text(json.dumps(data))
    return path,data


def test_source_requires_stopped_hash_verified_compatible_run(tmp_path):
    path,data=source_record(tmp_path)
    result=verified_bridge_source(path,tmp_path)
    assert result['checkpoint']==data['checkpoint']
    data['status']='running';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='stopped'):verified_bridge_source(path,tmp_path)
    data['status']='completed';data['reports'][0]['nan_episodes']=1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='finite'):verified_bridge_source(path,tmp_path)
    data['reports'][0]['nan_episodes']=0;path.write_text(json.dumps(data))
    (tmp_path/COMPATIBLE_SOURCES[0]).write_text('different architecture')
    with pytest.raises(ValueError,match='Incompatible'):verified_bridge_source(path,tmp_path)


def transition_record(root, data, out=None, changed=('src/mjlab_microduck/retained_skill_bridge.py',),
                      kind='actor_forward_transition', **overrides):
    edits={}
    for name in changed:
        path=root/name
        before=data['source_sha256'][str(path)]
        path.write_text('changed forward semantics: '+name)
        edits[name]={'sha256_before':before,'sha256_after':hashlib.sha256(path.read_bytes()).hexdigest()}
    record={'kind':kind,'approved_by_user':True,
            'parameter_identity_unchanged':True,'checkpoint_tensors_identical_on_load':True,
            'frozen_skills_sha256_before':data['frozen_skills_sha256'],
            'frozen_skills_sha256_after':data['frozen_skills_sha256'],
            'reason':'the hold gate left the residual with no authority',
            'change':'phase=1 keeps a reduced recorded share of the residual',
            'changed_sources':edits}
    record.update(overrides)
    path=(out or root/'local/transition.json');path.write_text(json.dumps(record))
    return path


def test_transition_continues_only_across_recorded_hashed_actor_changes(tmp_path):
    path,data=source_record(tmp_path)
    record=transition_record(tmp_path,data)
    result=verified_bridge_transition(path,tmp_path,record)
    assert result['transitioned_sources']==['src/mjlab_microduck/retained_skill_bridge.py']
    assert result['transition_kind']=='actor_forward_transition'
    assert result['checkpoint']==data['checkpoint']
    assert result['actor_transition_sha256']==hashlib.sha256(record.read_bytes()).hexdigest()
    # An undeclared source edit is still an incompatible continuation.
    (tmp_path/COMPATIBLE_SOURCES[1]).write_text('unrecorded edit')
    with pytest.raises(ValueError,match='Incompatible'):verified_bridge_transition(path,tmp_path,record)
    (tmp_path/COMPATIBLE_SOURCES[1]).write_text(COMPATIBLE_SOURCES[1])


def test_transition_record_must_be_approved_actor_only_and_baseline_exact(tmp_path):
    path,data=source_record(tmp_path)
    good=transition_record(tmp_path,data)
    assert verified_bridge_transition(path,tmp_path,good)
    with pytest.raises(ValueError,match='approved'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,approved_by_user=False))
    with pytest.raises(ValueError,match='known transition kind'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,kind='whatever_transition'))
    with pytest.raises(ValueError,match='reason and its change'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,reason=''))
    with pytest.raises(ValueError,match='parameter identity'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,parameter_identity_unchanged=False))
    with pytest.raises(ValueError,match='frozen expert'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,frozen_skills_sha256_after='other'))
    with pytest.raises(ValueError,match='compatible sources'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,changed_sources={}))
    with pytest.raises(ValueError,match='compatible sources'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,
            changed_sources={'local/train-credit-bridge.py':{'sha256_before':'a','sha256_after':'b'}}))
    other=json.loads(good.read_text())
    other['changed_sources']={'src/mjlab_microduck/checkpoint_safety.py':
        dict(list(other['changed_sources'].values())[0])}
    with pytest.raises(ValueError,match='must change'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,**other))
    record=json.loads(good.read_text())
    record['changed_sources']['src/mjlab_microduck/retained_skill_bridge.py']['sha256_before']='stale'
    with pytest.raises(ValueError,match='baseline'):
        verified_bridge_transition(path,tmp_path,_rewrite(good,**record))


def test_reward_forward_transition_is_a_separate_named_contract(tmp_path):
    path,data=source_record(tmp_path)
    record=transition_record(tmp_path,data,changed=('src/mjlab_microduck/tasks/mdp.py',),
                             kind='reward_forward_transition')
    result=verified_bridge_transition(path,tmp_path,record)
    assert result['transitioned_sources']==['src/mjlab_microduck/tasks/mdp.py']
    assert result['transition_kind']=='reward_forward_transition'
    # A record that is not an actor-forward transition may never move the actor.
    both={'src/mjlab_microduck/tasks/mdp.py':{'sha256_before':'a','sha256_after':'b'},
          'src/mjlab_microduck/retained_skill_bridge.py':{'sha256_before':'a','sha256_after':'b'}}
    with pytest.raises(ValueError,match='must not change the actor'):
        verified_bridge_transition(path,tmp_path,_rewrite(record,changed_sources=both))
    # An actor-forward transition may move the actor and nothing else.
    with pytest.raises(ValueError,match='must not also change'):
        verified_bridge_transition(path,tmp_path,_rewrite(record,kind='actor_forward_transition',
                                                          changed_sources=both))
    # The reward state machine is still pinned to the pre-change bytes by the record.
    mdp=tmp_path/'src/mjlab_microduck/tasks/mdp.py'
    mdp.write_text('an edit made after the record was written')
    with pytest.raises(ValueError,match='changed after recording'):
        verified_bridge_transition(path,tmp_path,record)


def test_task_contract_transition_owns_the_stage_definitions(tmp_path):
    path,data=source_record(tmp_path)
    record=transition_record(tmp_path,data,changed=('src/mjlab_microduck/tasks/mdp.py',
                             'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py'),
                             kind='task_contract_transition')
    result=verified_bridge_transition(path,tmp_path,record)
    assert result['transition_kind']=='task_contract_transition'
    assert result['transitioned_sources']==['src/mjlab_microduck/tasks/mdp.py',
        'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py']
    # Its own owner must move, and the actor still may not.
    with pytest.raises(ValueError,match='must change'):
        verified_bridge_transition(path,tmp_path,_rewrite(record,kind='task_contract_transition',
            changed_sources={'src/mjlab_microduck/tasks/mdp.py':{'sha256_before':'a','sha256_after':'b'}}))
    with pytest.raises(ValueError,match='must not change the actor'):
        verified_bridge_transition(path,tmp_path,_rewrite(record,changed_sources={
            'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py':{'sha256_before':'a','sha256_after':'b'},
            'src/mjlab_microduck/retained_skill_bridge.py':{'sha256_before':'a','sha256_after':'b'}}))


def _rewrite(path,**changes):
    record=json.loads(path.read_text());record.update(changes)
    out=path.with_name(path.stem+'-variant.json');out.write_text(json.dumps(record))
    return out


def test_curriculum_gate_requires_two_large_independent_credible_batches():
    row={'seed':1,'stage':'near','checkpoint_sha256':'abc','episodes':256,'nan_episodes':0,
         'initial_assistance':True,'success_050_rate':.9,'credible_050_rate':.85,'discounted_shaping_max_abs':1e-7}
    rows=[row,dict(row,seed=2)]
    assert bridge_stage_gate(rows,'near','abc')
    for changes in [{'episodes':64},{'seed':1},{'credible_050_rate':.79},{'nan_episodes':1},
                    {'checkpoint_sha256':'other'},{'initial_assistance':False}]:
        assert not bridge_stage_gate([row,dict(rows[1],**changes)],'near','abc')
    assert not bridge_stage_gate(rows,'full','abc')
    full=[dict(r,stage='full',initial_assistance=False,success_200_rate=.85,credible_200_rate=.82) for r in rows]
    assert bridge_stage_gate(full,'full','abc')
