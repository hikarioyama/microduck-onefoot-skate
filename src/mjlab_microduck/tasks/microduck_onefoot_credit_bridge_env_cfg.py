"""v15: finite-attempt glide credit plus terminal-correct potential guidance.

Both pretrained experts remain frozen. Legacy waist factories are not changed.
The full stage uses exactly the v14 HOME/rest start. Near/hold-check are explicitly
assisted diagnostics/curriculum entries and never unassisted completion evidence.
"""
from copy import deepcopy
import math
import re
import mujoco
import numpy as np
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg, make_waist_onefoot_rl_cfg
from .microduck_onefoot_curriculum_poses import POSES
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_walk_rollers_spec

TASK = 'Mjlab-OneFoot-Credit-Bridge-Flat-MicroDuck-Rollers'
REWARD_GAMMA = .99
STAGES = ('full', 'near', 'unload', 'transfer-near', 'transfer', 'glide', 'hold-check')
# `glide` is the assisted stage that trains the glide ALONE: it spawns at the
# measured single-support glide-entry configuration with the measured entry speed
# and phase_start=1.0, so no episode replays the 2.0 s acceleration prefix. That
# prefix has exactly zero learnable authority (the phase=0 correction gate is 0)
# and was 35% of every episode (2.0 s of a 5.73 s mean). Only `full`, which starts
# at HOME with exact zero velocity, can certify a from-rest result.
BRIDGE_CURRICULUM = ('near', 'unload', 'transfer-near', 'transfer', 'glide', 'full')
# Stages whose certification duration is the sustained 2.0 s glide.
SUSTAINED_STAGES = ('glide', 'full')
# Measured, existing spawn poses. Intermediate goals are 0.5 s; glide and full are 2 s.
ASSISTED_BRIDGE_STARTS = {
    'near': ('balance', .85, 1.7, None),
    'unload': ('unload', .65, 2.2, (.25, .35)),
    'transfer-near': ('transfer_near', .40, 2.8, (.25, .35)),
    'transfer': ('transfer', .05, 3.6, (.25, .35)),
    # Measured glide entry: speed p50 .521, p90 .559, max .568 (glide-pose-measurement.json).
    # The stage used to inject (.45,.58) -- inside a band where a credible glide is
    # capped at ~1.8 s by the speed budget alone. Sweeping the injected speed over a
    # frozen policy (probe-glide-entry-speed.py) puts the 2.0 s capability at an entry
    # speed of ~0.69 m/s and above: 0.62 m/s never reaches 2.0 s, 0.685 m/s does. The
    # range is therefore widened to straddle that boundary so the policy can train
    # where the target is actually reachable, while still covering the old band.
    'glide': ('glide', 1.0, 3.4, (.50, .76)),
}


def measured_home_spawn():
    model = get_walk_rollers_spec().compile()
    data = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError('Expected exactly one floating robot root')
    adr = model.jnt_qposadr[free[0]]
    data.qpos[adr:adr+7] = [0., 0., 0., 1., 0., 0., 0.]
    pose = {}
    for index in range(model.njnt):
        name = model.joint(index).name
        if index in free or name.startswith('passive_'):
            continue
        values = [value for pattern, value in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern, name)]
        if len(values) != 1:
            raise ValueError(f'Unresolved HOME coordinate: {name}')
        pose[name] = values[0]
        data.qpos[model.jnt_qposadr[index]] = values[0]
    mujoco.mj_forward(model, data)
    z = min(data.xpos[model.body(name).id, 2] for name in ('tire', 'tire_2', 'tire_3', 'tire_4'))
    return dict(pose=pose, root_z=mdp.CURRICULUM_WHEEL_RADIUS-z, root_roll=0.)


def reset_credit_bridge_spawn(env, env_ids, pose: dict, root_z: float, root_roll: float,
                              speed_range: tuple, phase_start: float,
                              roll_noise: float = .008, joint_noise: float = .004,
                              root_pitch: float = 0., joint_vel: dict = None,
                              wheel_speed_follow: bool = False):
    """Spawn-only assistance, extended so a mid-motion state is reproduced faithfully.

    `mdp.reset_curriculum_onefoot` starts every servo at rest and the root level. A
    spawn taken from the middle of a fast motion then presents a state that never
    occurs in a real trajectory: the posture of one instant with the joints of a
    standstill. At the glide entry the measured median |joint velocity| is 1.9 rad/s
    with the swing hip at 7.6 rad/s, and the root is pitched 0.11 rad forward, so
    spawning there at rest made the policy fail immediately (credible 0.5 s fell from
    0.97 to 0.30) even though it glides that state fine when it arrives under its own
    control.

    This lives here rather than in mdp.py on purpose. mdp.py's pre-v15 region is
    pinned byte-for-byte by tests/test_old_mdp_and_registry_bytes_are_preserved, and
    inserting optional parameters into the shared helper would shift that region and
    destroy an invariant that can never be restored. With root_pitch=0, joint_vel=None
    and wheel_speed_follow=False this is behaviourally the original spawn, and it is
    only installed for poses that actually carry the extra measured fields.
    Assistance is applied once, at reset; nothing is corrected mid-episode.

    `wheel_speed_follow` exists because the recorded wheel velocities are only valid
    at the speed they were recorded at. The glide pose was captured at 0.51 m/s, so
    its SUPPORT wheels read 36.7/36.5 rad/s. Injecting a different speed while
    leaving those wheels at a value that implies 0.55 m/s puts a contradiction in the
    actor's own observation, and the measured consequence is severe: sweeping the
    injected speed from 0.58 to 0.78 m/s raised the speed the policy actually settled
    at by only 0.526 -> 0.580 m/s with the recorded wheels, against 0.558 -> 0.742
    m/s when the support wheels track the injected speed instead. The two UNLOADED
    swing wheels are left at their recorded (near-still) values either way, because
    a wheel in mid-air does not roll with the ground.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, '_curriculum_phase_start'):
        env._curriculum_phase_start = torch.zeros(env.num_envs, device=env.device)
        env._curriculum_injected_speed = torch.zeros(env.num_envs, device=env.device)
    env._curriculum_phase_start[env_ids] = phase_start
    robot = env.scene['robot']; data = robot.data; n = len(env_ids)
    roots = data.data.qpos[env_ids[:, None], data.indexing.free_joint_q_adr].clone()
    roots[:, 2] = env.scene.env_origins[env_ids, 2] + root_z + .0005
    roll = root_roll + torch.empty(n, device=env.device).uniform_(-roll_noise, roll_noise)
    roots[:, 3:] = 0
    # Roll about x, then pitch about y. With pitch=0 this is the original quaternion.
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = math.cos(root_pitch / 2); sp = math.sin(root_pitch / 2)
    roots[:, 3] = cr * cp; roots[:, 4] = sr * cp
    roots[:, 5] = cr * sp; roots[:, 6] = -sr * sp
    velocity = torch.zeros(n, 6, device=env.device)
    velocity[:, 0] = torch.empty(n, device=env.device).uniform_(*speed_range)
    env._curriculum_injected_speed[env_ids] = velocity[:, 0]
    q = data.default_joint_pos[env_ids].clone(); dq = torch.zeros_like(q)
    for name, value in pose.items():
        idx = robot.find_joints(name)[0][0]
        q[:, idx] = value + torch.empty(n, device=env.device).uniform_(-joint_noise, joint_noise)
    if joint_vel:
        for name, value in joint_vel.items():
            dq[:, robot.find_joints(name)[0][0]] = value
    # At the glide entry the two SUPPORT wheels roll near v/r (measured 36.7/36.5
    # rad/s against v/r = 34) but the two UNLOADED swing wheels are nearly still
    # (measured -2.1/+11.1); forcing all four to v/r spins the swing skate's wheels
    # in mid-air and puts that spin straight into the actor observation.
    for name in ('passive_LF_wheel', 'passive_LR_wheel', 'passive_RF_wheel', 'passive_RR_wheel'):
        support = name in ('passive_LF_wheel', 'passive_LR_wheel')
        if not (wheel_speed_follow and support) and joint_vel and name in joint_vel:
            continue
        dq[:, robot.find_joints(name)[0][0]] = velocity[:, 0] / mdp.CURRICULUM_WHEEL_RADIUS
    robot.write_root_link_pose_to_sim(roots, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)


def make_credit_bridge_env_cfg(play=False, stage='full'):
    if stage not in STAGES:
        raise ValueError(stage)
    cfg = make_waist_onefoot_env_cfg(play=play, stage='self-launch' if stage == 'full' else 'balance-100')
    prior = cfg.commands['twist']
    # `full` keeps the official self-launch goal; `glide` must ask for exactly the
    # same sustained duration, otherwise the assisted stage would certify a
    # different task than the from-rest one.
    sustained_goal = prior.goal_s if stage == 'full' else 2.
    cfg.commands['twist'] = mdp.CreditBridgeCommandCfg(
        resampling_time_range=prior.resampling_time_range, target_speed=prior.target_speed,
        goal_s=sustained_goal if stage in SUSTAINED_STAGES else (prior.goal_s if stage == 'hold-check' else .5),
        acceleration_s=2. if stage == 'full' else 0., lift_s=1.,
        transfer_grace_s=prior.transfer_grace_s, reward_gamma=REWARD_GAMMA)
    if stage == 'full':
        cfg.episode_length_s = 6.
        cfg.events['curriculum_spawn'].params.update(measured_home_spawn())
    elif stage in ASSISTED_BRIDGE_STARTS:
        pose, phase, duration, speed_range = ASSISTED_BRIDGE_STARTS[stage]
        cfg.episode_length_s = duration
        cfg.events['curriculum_spawn'].params.update(deepcopy(POSES[pose]), phase_start=phase)
        if speed_range is not None:
            cfg.events['curriculum_spawn'].params['speed_range'] = speed_range
        # Only poses that carry the extra MEASURED fields (joint velocities, a pitched
        # root) switch to the faithful spawn; every other assisted stage keeps the
        # original helper byte for byte.
        recorded = POSES[pose]
        if 'joint_vel' in recorded or recorded.get('root_pitch'):
            cfg.events['curriculum_spawn'].func = reset_credit_bridge_spawn
            # The pose's recorded wheel speeds are only valid at the speed they were
            # recorded at, so the SUPPORT wheels track the injected speed. Measured
            # effect on a frozen policy, injecting 0.58 -> 0.78 m/s: the entry speed
            # the policy settles at moves 0.526 -> 0.580 m/s with the recorded wheels
            # and 0.558 -> 0.742 m/s with them following. See
            # local/probe-glide-entry-speed.py.
            cfg.events['curriculum_spawn'].params['wheel_speed_follow'] = True
    # This horizon is the TRUE end of the task attempt, not an artificial
    # truncation of a continuing process. RSL must not bootstrap across it.
    for term in cfg.terminations.values():
        term.time_out = False
    cfg.rewards = {name: RewardTermCfg(func=mdp.credit_bridge_reward, weight=1.,
                    params={'component':name}) for name in ('task', 'shaping', 'failure', 'handoff')}
    last = ('credible_best_s', 'entry_speed', 'balance_debt')
    for name in ('credible', 'credible_best_s', 'progress_steps', 'potential', 'quality',
                 'failed', 'missed_goal', 'handoff', 'entry_speed', 'balance_debt'):
        cfg.metrics['bridge/'+name] = MetricsTermCfg(func=mdp.credit_bridge_metric,
            params={'component':name}, reduce='last' if name in last else 'mean')
    cfg.metrics['onefoot/diagnostic_valid'] = MetricsTermCfg(
        func=mdp.observe_curriculum_onefoot_support_kinematics, reduce='last')
    cfg.sim.nan_guard.enabled = True
    return cfg


def make_credit_bridge_rl_cfg():
    cfg = make_waist_onefoot_rl_cfg()
    cfg.actor.class_name = 'mjlab_microduck.retained_skill_bridge:RetainedSkillBridgeModel'
    cfg.actor.distribution_cfg = dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',
                                     init_std=.08, std_type='log')
    cfg.algorithm.gamma = REWARD_GAMMA
    cfg.algorithm.learning_rate = 1e-4
    # Frozen-expert frames have zero bridge-mean KL. Adaptive scheduling raised
    # lr to .01 before the first bridge update; do not use that diluted signal.
    cfg.algorithm.schedule = 'fixed'
    cfg.algorithm.entropy_coef = .005
    cfg.run_name = 'credit-bridge-v15'
    cfg.experiment_name = 'onefoot_credit_bridge'
    cfg.logger = 'tensorboard'
    cfg.save_interval = 100
    return cfg
