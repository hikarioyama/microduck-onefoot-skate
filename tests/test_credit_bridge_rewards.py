import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import (
    REWARD_GAMMA, make_credit_bridge_env_cfg, make_credit_bridge_rl_cfg,
)


def step(state, phi=.5, valid=False, done=False, failure=False, cost=0., com=.3, skate=.2,
         balance=0.):
    n=len(state.previous_phi)
    f=lambda value:torch.full((n,),value,dtype=torch.float32)
    b=lambda value:torch.full((n,),value,dtype=torch.bool)
    return state.advance(phi=f(phi),valid=b(valid),com_speed=f(com),skate_speed=f(skate),
                         smooth_cost=f(cost),done=b(done),physical_failure=b(failure),
                         balance_cost=f(balance))


@pytest.mark.parametrize('gamma',[1.,.99,.97])
@pytest.mark.parametrize('length',[1,2,25,150,300])
def test_failed_potential_paths_cannot_create_discounted_profit(gamma,length):
    state=mdp.CreditBridgeRewardState(1,'cpu',gamma=gamma)
    generator=np.random.default_rng(4)
    shaped=0.;total=0.
    for t in range(length):
        r=step(state,phi=float(generator.uniform()),done=t==length-1,failure=t==length-1)
        shaped+=gamma**t*float(r['shaping'][0]);total+=gamma**t*float(r['total'][0])
    assert shaped==pytest.approx(0.,abs=2e-6)
    assert total==pytest.approx(-2*gamma**(length-1),abs=2e-6)


def test_early_failure_is_not_better_than_zero_credit_safe_continuation():
    returns=[]
    for length in (125,165):
        state=mdp.CreditBridgeRewardState(1,'cpu',gamma=.99)
        reward=0.
        for t in range(length):
            r=step(state,phi=(t%11)/10,done=t==length-1,failure=t==length-1,cost=1000.)
            reward+=.99**t*float(r['total'][0])
        returns.append(reward)
    assert returns[0] < returns[1] < 0


def test_no_task_credit_for_standing_torso_rocking_or_swing_recatches():
    state=mdp.CreditBridgeRewardState(1,'cpu')
    assert step(state,valid=True,skate=0.)['task'].item()==0
    assert step(state,valid=True,com=0.)['task'].item()==0
    state.reset()
    progress=[]
    for valid in (True,True,False,True,True,False,True,True,True):
        progress.append(step(state,valid=valid)['progress_steps'].item())
    assert progress==[1,1,0,0,0,0,0,0,1]
    assert state.best_steps.item()==3


def test_task_credit_is_rate_limited_nonnegative_and_safety_cost_cannot_be_escaped():
    low=mdp.CreditBridgeRewardState(1,'cpu');high=mdp.CreditBridgeRewardState(1,'cpu')
    for _ in range(10):
        a=step(low,valid=True,cost=0.)['task'].item()
        b=step(high,valid=True,cost=100.)['task'].item()
        assert 0<=b<=a<=12*.02+1e-7
    assert step(high,valid=False,cost=100.)['task'].item()==0


def test_transfer_overdrive_discounts_later_glide_and_cannot_be_erased_by_waiting():
    quiet=mdp.CreditBridgeRewardState(1,'cpu')
    rough=mdp.CreditBridgeRewardState(1,'cpu')
    step(quiet,valid=False,cost=0.)
    step(rough,valid=False,cost=100.)
    for _ in range(10):
        step(quiet,valid=False,cost=0.)
        step(rough,valid=False,cost=0.)
    assert step(rough,valid=True)['task'].item()<step(quiet,valid=True)['task'].item()
    assert rough.smooth_debt.item()==pytest.approx(2.)
    rough.reset();assert rough.smooth_debt.item()==0.


def test_goal_timeout_is_terminal_but_does_not_pay_a_completion_jackpot():
    state=mdp.CreditBridgeRewardState(1,'cpu',goal_s=.04)
    step(state,valid=True)
    r=step(state,valid=True,done=True)
    assert r['failed'].item()==0 and r['potential'].item()==0
    assert r['task'].item()<=.24+1e-7
    state.reset()
    r=step(state,done=True,failure=False)
    assert r['missed_goal'].item()==1 and r['failure'].item()==-2
    state.reset()
    step(state,valid=True);step(state,valid=True)
    r=step(state,valid=True,done=True,failure=True)
    assert r['failed'].item()==1 and r['task'].item()==0


def test_partial_reset_does_not_clear_other_worlds_or_carry_shaping_debt():
    state=mdp.CreditBridgeRewardState(2,'cpu')
    step(state,phi=.8,valid=True)
    state.reset(torch.tensor([0]))
    assert state.previous_phi.tolist()==pytest.approx([0.,.8])
    assert state.best_steps.tolist()==[0,1]
    state.reset();assert state.previous_phi.sum()==0 and state.best_steps.sum()==0


def test_reward_state_can_reset_outside_inference_after_ppo_rollout():
    state=mdp.CreditBridgeRewardState(2,'cpu')
    with torch.inference_mode():step(state,phi=.8,valid=True,cost=1.)
    state.reset(torch.tensor([0]))
    assert state.best_steps.tolist()==[0,1]
    assert state.previous_phi.tolist()==pytest.approx([0.,.8])
    assert state.smooth_debt.tolist()==pytest.approx([0.,.02])


def test_potential_rewards_no_stationary_or_fallen_pose_and_is_bounded():
    f=lambda x:torch.tensor([x],dtype=torch.float32)
    args=[f(.3),f(.2),f(0.),f(0.),f(3.),f(3.),f(0.),f(.02),f(1.),f(0.),torch.tensor([False])]
    assert mdp.credit_bridge_potential(*args).item()==pytest.approx(1.)
    for index,value in [(0,0.),(1,0.),(8,.4)]:
        altered=args.copy();altered[index]=f(value)
        assert mdp.credit_bridge_potential(*altered).item()==0
    # v16: a fore/aft capture error must scale the potential down, on its own axis.
    for value in (.04,.08):
        altered=args.copy();altered[3]=f(value)
        assert mdp.credit_bridge_potential(*altered).item()<.4
    altered=args.copy();altered[3]=f(float('nan'))
    assert mdp.credit_bridge_potential(*altered).item()==0
    altered=args.copy();altered[-1]=torch.tensor([True])
    assert mdp.credit_bridge_potential(*altered).item()==0


def test_balance_debt_is_free_inside_the_deadband_and_charged_outside_it():
    f=lambda x:torch.tensor([x],dtype=torch.float32)
    quiet=mdp.credit_bridge_balance_cost(f(mdp.BRIDGE_LATERAL_DEADBAND),
        f(mdp.BRIDGE_FOREAFT_DEADBAND),torch.tensor([True]))
    assert quiet.item()==0
    # Measured p75 balance still costs almost nothing over a 2 s glide.
    p75=mdp.credit_bridge_balance_cost(f(.0149),f(.0084),torch.tensor([True])).item()
    assert 0.<p75*2.<.30
    # Measured p90 drift is charged hard, and both axes are charged independently.
    p90=mdp.credit_bridge_balance_cost(f(.0308),f(.0136),torch.tensor([True])).item()
    assert p90*2>1.5
    assert mdp.credit_bridge_balance_cost(f(.0308),f(mdp.BRIDGE_FOREAFT_DEADBAND),
        torch.tensor([True])).item()>0
    assert mdp.credit_bridge_balance_cost(f(mdp.BRIDGE_LATERAL_DEADBAND),f(.0273),
        torch.tensor([True])).item()>0
    # Two-footed support cannot farm or dodge the debt.
    assert mdp.credit_bridge_balance_cost(f(.10),f(.10),torch.tensor([False])).item()==0


def test_balance_debt_discounts_later_glide_credit_and_is_sticky():
    quiet=mdp.CreditBridgeRewardState(1,'cpu')
    drifting=mdp.CreditBridgeRewardState(1,'cpu')
    for _ in range(10):
        step(quiet,valid=True,balance=0.)
        step(drifting,valid=True,balance=5.)
    assert drifting.balance_debt.item()==pytest.approx(1.)
    assert step(drifting,valid=True)['task'].item()<step(quiet,valid=True)['task'].item()
    drifting.reset()
    assert drifting.balance_debt.item()==0. and drifting.entry_speed.item()==0.
    assert drifting.handoff_paid.item()==0 and drifting.established.item()==0


def expected_handoff(com):
    span=mdp.BRIDGE_HANDOFF_SPAN
    return mdp.BRIDGE_HANDOFF_GAIN*min(1.,max(0.,(com-mdp.BRIDGE_HANDOFF_REF_SPEED)/span))


def test_handoff_bonus_is_paid_once_only_above_the_frozen_experts_own_ceiling():
    assert mdp.BRIDGE_HANDOFF_REF_SPEED==.38, 'Ceiling is the measured public-expert plateau'
    assert mdp.BRIDGE_HANDOFF_SPAN>.15, 'Must price speeds ABOVE the measured .55 m/s glide peak'
    hold=round(mdp.BRIDGE_HANDOFF_HOLD_S/.02)
    for com in (.336,.38,.455,.60,.70,.90):
        state=mdp.CreditBridgeRewardState(1,'cpu')
        got=0.
        for _ in range(hold+30):
            got+=float(step(state,valid=True,com=com)['handoff'][0])
        assert got==pytest.approx(expected_handoff(com)),com
        assert state.handoff_paid.item()==1
        assert state.entry_speed.item()==pytest.approx(com)
    assert expected_handoff(.38)==0. and expected_handoff(.336)==0.
    assert expected_handoff(.60)>0.
    # A throw that never reaches the hold duration pays nothing.
    state=mdp.CreditBridgeRewardState(1,'cpu')
    paid=0.
    for _ in range(hold-1):
        paid+=float(step(state,valid=True,com=.60)['handoff'][0])
    assert paid==0. and state.handoff_paid.item()==0
    # Lost support resets the hold clock, so the bonus is not a per-frame payment.
    state=mdp.CreditBridgeRewardState(1,'cpu')
    for _ in range(hold-1):
        step(state,valid=True,com=.60)
    step(state,valid=False,com=.60)
    assert step(state,valid=True,com=.60)['handoff'].item()==0


def test_handoff_bonus_cannot_make_a_reckless_lift_profitable():
    assert mdp.BRIDGE_HANDOFF_GAIN < 2., 'Must stay below failure_cost'
    # Sprint, hold just past the handoff clock, then fall: the one-shot bonus is
    # paid, but the failure event still dominates the whole episode.
    state=mdp.CreditBridgeRewardState(1,'cpu',handoff_hold_s=.02)
    r=step(state,valid=True,com=.70)
    assert r['handoff'].item()==pytest.approx(mdp.BRIDGE_HANDOFF_GAIN)
    assert state.handoff_paid.item()==1
    total=float(r['total'][0])
    r=step(state,valid=False,com=.70,failure=True,done=True)
    assert r['failure'].item()==-2.
    assert total+float(r['total'][0])<0
    # A frame that is itself a physical failure can never be credible, so it can
    # neither start nor extend the handoff clock.
    fail=mdp.CreditBridgeRewardState(1,'cpu',handoff_hold_s=.02)
    r=step(fail,valid=True,com=.70,failure=True,done=True)
    assert r['handoff'].item()==0. and fail.handoff_paid.item()==0
    assert fail.entry_speed.item()==pytest.approx(.70)


def test_new_config_preserves_physics_and_commands_but_disables_timeout_bootstrap():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('old_credit_comparison',root/'local/evaluate-public-roller-connection.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    baseline=old.environment_cfg(1,1,6.,2.,'home')
    cfg=make_credit_bridge_env_cfg()
    cfg.scene.num_envs=1;cfg.seed=1
    assert cfg.actions==baseline.actions and cfg.observations==baseline.observations
    # Terrain's default spec_fn is a freshly allocated lambda per factory call.
    # Compare what it builds before aligning callable identity for dataclass eq.
    assert cfg.scene.terrain.spec_fn().to_xml()==baseline.scene.terrain.spec_fn().to_xml()
    cfg.scene.terrain.spec_fn=baseline.scene.terrain.spec_fn
    assert cfg.scene==baseline.scene and cfg.sim==baseline.sim and cfg.events==baseline.events
    assert cfg.commands['twist'].goal_s==baseline.commands['twist'].goal_s
    assert cfg.commands['twist'].acceleration_s==2. and cfg.commands['twist'].lift_s==1.
    assert set(cfg.terminations)==set(baseline.terminations)
    for name,term in cfg.terminations.items():
        assert not term.time_out
        assert term.func is baseline.terminations[name].func
        assert term.params==baseline.terminations[name].params
    assert set(cfg.rewards)=={'task','shaping','failure','handoff'}
    assert all(term.weight==1 for term in cfg.rewards.values())
    rl=make_credit_bridge_rl_cfg()
    assert rl.algorithm.gamma==cfg.commands['twist'].reward_gamma==REWARD_GAMMA
    assert rl.algorithm.schedule=='fixed' and rl.algorithm.learning_rate==1e-4
    assert baseline.terminations['time_out'].time_out  # Legacy behavior was not mutated.
    near=make_credit_bridge_env_cfg(stage='near')
    assert near.events['curriculum_spawn'].params['phase_start']==.85
    assert near.events['curriculum_spawn'].params['speed_range']!=(0.,0.)
    assert near.commands['twist'].goal_s==.5
    for stage,phase in [('unload',.65),('transfer-near',.40),('transfer',.05)]:
        assisted=make_credit_bridge_env_cfg(stage=stage)
        assert assisted.events['curriculum_spawn'].params['phase_start']==phase
        assert assisted.events['curriculum_spawn'].params['speed_range']==(.25,.35)
        assert assisted.commands['twist'].goal_s==.5
        assert assisted.actions==cfg.actions and assisted.observations==cfg.observations
        assert set(assisted.rewards)==set(cfg.rewards)
    assert cfg.commands['twist'].goal_s==2.
    # The glide stage trains the sustained glide from its own measured entry state,
    # with no acceleration prefix and the full 2 s goal, so an episode never replays
    # the frozen (zero-authority) phase=0 window.
    from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import (
        ASSISTED_BRIDGE_STARTS, BRIDGE_CURRICULUM, SUSTAINED_STAGES)
    from mjlab_microduck.tasks.microduck_onefoot_curriculum_poses import POSES
    glide=make_credit_bridge_env_cfg(stage='glide')
    assert 'glide' in SUSTAINED_STAGES and 'full' in SUSTAINED_STAGES
    assert BRIDGE_CURRICULUM.index('glide')<BRIDGE_CURRICULUM.index('full')
    assert glide.commands['twist'].goal_s==2.
    assert glide.commands['twist'].acceleration_s==0.
    assert glide.events['curriculum_spawn'].params['phase_start']==1.
    lo,hi=glide.events['curriculum_spawn'].params['speed_range']
    # The band must reach the speed at which a 2 s credible glide first becomes
    # possible for this body. Measured by sweeping the injected speed over a frozen
    # policy: entry 0.62 m/s never reaches 2.0 s, entry 0.685 m/s does.
    assert lo>=.40 and hi>=.69, 'Must spawn into the band where 2.0 s is reachable'
    assert hi<.80, 'The valid predicate rejects root speed >= 0.80 m/s'
    # The recorded wheel speeds are only valid at the speed they were recorded at, so
    # the support wheels must track the injected speed. Measured on a frozen policy,
    # injecting 0.58 -> 0.78 m/s: entry moves 0.526 -> 0.580 m/s with the recorded
    # wheels against 0.558 -> 0.742 m/s when they follow.
    assert glide.events['curriculum_spawn'].params.get('wheel_speed_follow') is True
    assert glide.episode_length_s>2., 'The horizon must fit the sustained 2 s goal'
    assert ASSISTED_BRIDGE_STARTS['glide'][1]==1., 'phase_start=1.0 skips the frozen prefix'
    # The spawn restores the measured mid-motion state, not just a posture: the
    # glide entry carries real joint velocities and a real forward pitch, and the
    # stage switches to the faithful spawn helper to install them.
    from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import reset_credit_bridge_spawn
    spawn=glide.events['curriculum_spawn'].params
    assert glide.events['curriculum_spawn'].func is reset_credit_bridge_spawn
    assert 'joint_vel' in spawn and set(POSES['glide']['pose'])<set(spawn['joint_vel'])
    assert max(abs(v) for v in spawn['joint_vel'].values())>1.
    assert abs(spawn['root_pitch'])>0.05, 'Glide entry is measurably pitched forward'
    for stage in ('near','unload','transfer-near','transfer'):
        assisted_stage=make_credit_bridge_env_cfg(stage=stage)
        assert 'joint_vel' not in assisted_stage.events['curriculum_spawn'].params
        assert assisted_stage.events['curriculum_spawn'].func is mdp.reset_curriculum_onefoot
    # It is an assisted start: never from-rest evidence, and it says so.
    assert set(POSES['glide']['pose'])==set(POSES['transfer']['pose'])
    assert glide.commands['twist'].target_speed==cfg.commands['twist'].target_speed


def test_glide_pose_is_a_measured_single_support_configuration():
    import json
    root=Path(__file__).resolve().parents[1]
    record=root/'local/onefoot-bridge-curriculum/glide-pose-measurement.json'
    if not record.exists():pytest.skip('Glide pose measurement unavailable')
    measured=json.loads(record.read_text())
    from mjlab_microduck.tasks.microduck_onefoot_curriculum_poses import POSES
    pose=POSES['glide']
    assert measured['captured']>=32, 'Pose must come from many real single-support frames'
    assert measured['speed']['p50']>.40, 'The measured entry really is a fast glide'
    for name,value in pose['pose'].items():
        assert abs(value-measured['median_pose'][name])<.02, name
    assert pose['root_z']==pytest.approx(-measured['lowest_wheel_mesh_min_z'],abs=1e-6)
    lift=measured['wheel_lift']
    assert min(lift['tire_3'],lift['tire_4'])>.010, 'Swing skate clearly up'
    assert min(lift['tire'],lift['tire_2'])<.012, 'Support skate grounded'
    assert pose['root_roll']==pytest.approx(measured['roll']['p50'],abs=1e-9)
    assert pose['root_pitch']==pytest.approx(measured['pitch']['p50'],abs=1e-9)
    # Velocities cover ALL non-free joints, not just the servos: the unloaded swing
    # wheels must not be forced to the rolling spin v/r of the support wheels.
    assert set(pose['pose'])<set(pose['joint_vel'])
    for name,value in pose['joint_vel'].items():
        assert value==pytest.approx(measured['median_all_joint_vel'][name],abs=1e-9)
    wheels=[pose['joint_vel'][n] for n in ('passive_LF_wheel','passive_LR_wheel')]
    swing=[pose['joint_vel'][n] for n in ('passive_RF_wheel','passive_RR_wheel')]
    assert min(wheels)>30., 'Support wheels roll near v/r'
    assert max(abs(v) for v in swing)<15., 'Swing wheels are nearly still, NOT spun at v/r'


def test_old_mdp_and_registry_bytes_are_preserved():
    root=Path(__file__).resolve().parents[1]
    meta=root/'local/onefoot-reward-review/source-before.json'
    if not meta.exists():pytest.skip('Local baseline snapshot unavailable')
    record=json.loads(meta.read_text())
    for name in ('src/mjlab_microduck/tasks/mdp.py','src/mjlab_microduck/tasks/__init__.py'):
        old=record[name]
        assert hashlib.sha256((root/name).read_bytes()[:old['bytes']]).hexdigest()==old['sha256']
