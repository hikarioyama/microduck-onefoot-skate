"""Frozen public acceleration / real-state switch diagnostic, never stage proof.

No source checkpoint/state updates, no mid-episode state corrections. Physics,
noise, DR, delays and valid-glide checks are inherited from the waist task.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import safe_runner_load
from mjlab_microduck.onefoot_supervision import FAILURE_CHECK_NAMES, FAILURE_VALUE_NAMES
from mjlab_microduck.public_roller_policy import PublicRollerPolicy
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_walk_rollers_spec
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import (
    TASK, make_waist_onefoot_env_cfg, make_waist_onefoot_rl_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SHA256 = 'cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd'
PUBLIC_PATH = ROOT/'local/onefoot-public-acceleration/assets/088524a64e2557dc453256b6071dbb9d23888802/roller.onnx'
STATE = ROOT/'local/onefoot-curriculum-state.json'


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, p10=None, p90=None)
    return dict(n=int(len(values)), mean=float(values.mean()),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def grounded_home():
    """Measure the current HOME on the real roller geometry, not a target pose."""
    model = get_walk_rollers_spec().compile()
    data = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert len(free) == 1
    adr = model.jnt_qposadr[free[0]]
    data.qpos[adr:adr+7] = [0., 0., 0., 1., 0., 0., 0.]
    pose = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if name.startswith('passive_') or j in free:
            continue
        matches = [value for pattern, value in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern, name)]
        assert len(matches) == 1, name
        pose[name] = matches[0]
        data.qpos[model.jnt_qposadr[j]] = matches[0]
    mujoco.mj_forward(model, data)
    wheel_z = [data.xpos[model.body(name).id, 2] for name in ('tire', 'tire_2', 'tire_3', 'tire_4')]
    # The standard reset adds 0.5 mm clearance and retains small pose noise.
    return dict(pose=pose, root_z=mdp.CURRICULUM_WHEEL_RADIUS-min(wheel_z), root_roll=0.)


def environment_cfg(episodes, seed, duration_s, switch_s, spawn='home'):
    cfg = make_waist_onefoot_env_cfg(play=False, stage='self-launch')
    cfg.scene.num_envs = episodes
    cfg.seed = seed
    cfg.episode_length_s = duration_s
    cfg.sim.nan_guard.enabled = True
    cfg.commands['twist'].acceleration_s = switch_s if switch_s is not None else duration_s + 1.
    cfg.commands['twist'].lift_s = 1.
    if spawn == 'home':
        cfg.events['curriculum_spawn'].params.update(grounded_home())
    cfg.metrics['onefoot/diagnostic_valid'] = MetricsTermCfg(
        func=mdp.observe_curriculum_onefoot_support_kinematics, reduce='last')
    assert cfg.events['curriculum_spawn'].params['speed_range'] == (0., 0.)
    assert cfg.events['curriculum_spawn'].params['phase_start'] == 0.
    assert 'push_robot' not in cfg.events and not cfg.curriculum
    assert cfg.actions['joint_pos'].scale == 1.0
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--seed', type=int, default=70101)
    parser.add_argument('--duration-s', type=float, default=6.)
    parser.add_argument('--switch-s', type=float)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--action-scale', type=float, choices=(.8, 1.), required=True)
    parser.add_argument('--push-command', type=float, default=.6)
    parser.add_argument('--spawn', choices=('home', 'transfer'), default='home')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 256 or not 0 < args.duration_s <= 15:
        parser.error('Diagnostic is limited to 256 first episodes of at most 15 seconds')
    if (args.switch_s is None) != (args.checkpoint is None):
        parser.error('--switch-s and --checkpoint must be supplied together')
    if args.switch_s is not None and not 0 <= args.switch_s < args.duration_s:
        parser.error('Switch must occur within the episode')
    if args.device == 'cuda:0':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
        assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
        assert torch.cuda.device_count() == 1 and '5070 Ti' in torch.cuda.get_device_name(0)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    guard_files = [STATE, PUBLIC_PATH, ROOT/'src/mjlab_microduck/tasks/mdp.py',
                   ROOT/'src/mjlab_microduck/tasks/microduck_onefoot_waist_env_cfg.py']
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
        assert args.checkpoint.is_relative_to(ROOT/'logs/rsl_rl/onefoot')
        guard_files.append(args.checkpoint)
    guards = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in guard_files}
    configure_torch_backends()
    torch.set_num_threads(4 if args.device == 'cuda:0' else 1)
    policy = PublicRollerPolicy(PUBLIC_PATH, expected_sha256=PUBLIC_SHA256, action_scale=args.action_scale)
    cfg = environment_cfg(args.episodes, args.seed, args.duration_s, args.switch_s, args.spawn)
    rl = make_waist_onefoot_rl_cfg()
    started = time.monotonic()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=rl.clip_actions)
    try:
        raw = env.unwrapped
        obs = env.get_observations()
        robot = raw.scene['robot']; cmd = raw.command_manager.get_term('twist')
        ids, names = robot.find_joints(r'^(?!passive_).*')
        policy.validate_robot(names, robot.data.default_joint_pos[0, ids].cpu().numpy())
        assert obs['actor'].shape == (args.episodes, 61)
        assert obs['critic'].shape == (args.episodes, 78)
        assert env.num_actions == 14 and raw.step_dt == .02
        initial_qpos = raw.sim.data.qpos.clone()
        initial_qvel = raw.sim.data.qvel.clone()
        assert torch.count_nonzero(initial_qvel).item() == 0, 'Must start at EXACT rest'
        actor = None
        if args.checkpoint:
            devices = [0] if args.device == 'cuda:0' else []
            with torch.random.fork_rng(devices=devices):
                runner = load_runner_cls(TASK)(env, deepcopy(asdict(rl)), device=args.device)
                safe_runner_load(runner, args.checkpoint, load_cfg={'actor': True}, strict=True, map_location=args.device)
                actor = runner.get_inference_policy(device=args.device)
                assert not actor.is_recurrent, 'This first connection diagnostic uses the preserved MLP'
        alive = torch.ones(args.episodes, dtype=torch.bool, device=args.device)
        nan = torch.zeros_like(alive)
        best = torch.zeros(args.episodes, device=args.device)
        histories = {k: [] for k in ('values', 'checks', 'kinematics', 'alive', 'done', 'actions', 'qpos_before', 'qvel_before')}
        switch = None
        previous_action = torch.zeros(args.episodes, 14, device=args.device)
        with torch.no_grad():
            for step in range(raw.max_episode_length+1):
                use_onefoot = args.switch_s is not None and step*raw.step_dt >= args.switch_s-1e-9
                histories['qpos_before'].append(torch.where(alive[:, None], raw.sim.data.qpos, float('nan')).clone())
                histories['qvel_before'].append(torch.where(alive[:, None], raw.sim.data.qvel, float('nan')).clone())
                if use_onefoot:
                    actions = actor(obs)
                    if switch is None:
                        switch = dict(time_s=step*raw.step_dt, alive_episodes=int(alive.sum()),
                                      ended_before_switch=int((~alive).sum()),
                                      issued_action_jump_max_rad=statistics((actions-previous_action).abs().amax(1)[alive].cpu().numpy()),
                                      issued_action_jump_rms_rad=statistics((actions-previous_action).square().mean(1).sqrt()[alive].cpu().numpy()))
                else:
                    actions = torch.from_numpy(policy(obs['actor'].cpu().numpy(), push_command=args.push_command)).to(args.device)
                bad = ~torch.isfinite(actions).all(1)
                nan |= alive & bad
                if bool(bad.any()):
                    raise RuntimeError('Nonfinite actions; refusing to step simulator')
                histories['actions'].append(torch.where(alive[:, None], actions, float('nan')).clone())
                obs, _, done, _ = env.step(actions)
                snapshot = cmd.onefoot_diagnostics
                assert snapshot['step'] == raw.common_step_counter
                assert torch.equal(snapshot['checks'].all(1), cmd.curriculum_values['single_support'] > .5)
                histories['values'].append(torch.where(alive[:, None], snapshot['values'], float('nan')))
                histories['kinematics'].append(torch.where(alive[:, None], snapshot['support_kinematics'], float('nan')))
                histories['checks'].append(snapshot['checks'] & alive[:, None])
                histories['alive'].append(alive.clone())
                histories['done'].append(done.bool() & alive)
                nan |= alive & raw.termination_manager.get_term('nan_state')
                best = torch.maximum(best, torch.where(alive, cmd.curriculum_values['best_dwell'], 0))
                previous_action = actions.clone()
                alive &= ~done.bool()
                if actor is not None:
                    actor.reset(done)
                if not bool(alive.any()):
                    break
        arrays = {key: torch.stack(items).cpu().numpy() for key, items in histories.items()}
        arrays.update(initial_qpos=initial_qpos.cpu().numpy(), initial_qvel=initial_qvel.cpu().numpy(),
                      best_glide_s=best.cpu().numpy(), nan=nan.cpu().numpy(),
                      value_names=np.array(FAILURE_VALUE_NAMES), check_names=np.array(FAILURE_CHECK_NAMES),
                      kinematic_names=np.array(mdp.SUPPORT_KINEMATIC_NAMES),
                      time_s=np.arange(1, len(histories['values'])+1)*raw.step_dt)
        samples = []
        for t in (.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8.):
            index = round(t/raw.step_dt)-1
            present = arrays['alive'][index] if index < len(arrays['alive']) else np.zeros(args.episodes, bool)
            row = dict(time_s=t, present_episodes=int(present.sum()), ended_before_sample=int((~present).sum()))
            if present.any():
                row['values'] = {name: statistics(arrays['values'][index, present, j]) for j, name in enumerate(FAILURE_VALUE_NAMES)}
                row['kinematics'] = {name: statistics(arrays['kinematics'][index, present, j]) for j, name in enumerate(mdp.SUPPORT_KINEMATIC_NAMES)}
            samples.append(row)
        reached = np.zeros(args.episodes, bool)
        v = arrays['values']; k = arrays['kinematics']
        # Screening only: COM motion AND support-wheel translation, not torso rocking.
        moving = (arrays['alive'] & (v[:, :, 0] > .10) & (v[:, :, 1] > .10)
                  & (k[:, :, -1] > .05) & (v[:, :, 3] > .80) & arrays['checks'][:, :, -1])
        consecutive = np.zeros(args.episodes, int)
        for row in moving:
            consecutive = np.where(row, consecutive+1, 0)
            reached |= consecutive >= round(.5/raw.step_dt)
        for p, digest in guards.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
        np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
        report = dict(scope='Frozen public policy/connection diagnostic, NOT curriculum gate evidence',
                      seed=args.seed, episodes=args.episodes, device=args.device, spawn=args.spawn,
                      initial_velocity_max_abs=float(initial_qvel.abs().max()), physics_dt=cfg.sim.mujoco.timestep,
                      control_dt=raw.step_dt, duration_s=args.duration_s, switch=switch,
                      public_policy=str(PUBLIC_PATH), public_sha256=PUBLIC_SHA256,
                      deployment_action_scale=args.action_scale, onnx_metadata_action_scale=policy.metadata['action_scale'],
                      push_command=args.push_command, roller_heading_command=0.,
                      checkpoint=str(args.checkpoint) if args.checkpoint else None,
                      acceleration_screen_050_rate=float(reached.mean()),
                      mean_best_glide_s=float(best.mean()), success_050_rate=float((best >= .5-1e-5).float().mean()),
                      success_100_rate=float((best >= 1.-1e-5).float().mean()), success_200_rate=float((best >= 2.-1e-5).float().mean()),
                      nan_episodes=int(nan.sum()), fixed_time_samples=samples,
                      timeline_npz=str(args.output.with_suffix('.npz')), guarded_sha256=guards,
                      wall_time_s=time.monotonic()-started,
                      limitations=['Fixed-time statistics condition on surviving first episodes; terminal samples are included.',
                                   'Acceleration screening is not proof of straightness, safe switching, or task completion.',
                                   'HOME spawn retains the existing small pose noise, all initial velocities are exactly zero.',
                                   'Policy output is unfiltered; jump measures issued HOME-relative targets, before BAM delays.',
                                   'No external balancing force, velocity injection, or pose correction at the switch.'])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print('PUBLIC_DIAGNOSTIC_COMPLETE', json.dumps({key: report[key] for key in (
            'seed', 'episodes', 'deployment_action_scale', 'acceleration_screen_050_rate', 'switch',
            'mean_best_glide_s', 'success_100_rate', 'nan_episodes', 'wall_time_s')}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
