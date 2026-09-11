"""Compare the assisted glide SPAWN state against the real handoff state.

Read-only. Answers, joint by joint and in the actor's own 61-D observation, why a
policy that glides fine out of its own handoff fails when spawned directly at the
recorded glide entry. Records:

  * the state the `glide` stage actually installs at reset,
  * the state the `full` stage has at its first loaded-left-only-support frame,
  * every joint (including the passive linkage), the root height/orientation, and
    the actor observation at both instants.
"""
import argparse
import importlib.util
import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('credit_runner', ROOT/'local/train-credit-bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--envs', type=int, default=16)
a = p.parse_args()
assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
configure_torch_backends(allow_tf32=False)
torch.set_num_threads(4)


def resolve_joint_names(robot):
    from mjlab_microduck.tasks import mdp as _mdp
    model = _mdp._credit_bridge_joint_names(robot)
    return model


def joint_names():
    """Non-free joint names in the entity's own joint_pos column order."""
    import mujoco
    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec
    model = get_walk_rollers_spec().compile()
    return [model.joint(i).name for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def snapshot(raw, robot, names):
    return dict(
        joint_pos={names[i]: robot.data.joint_pos[0, i].item() for i in range(len(names))},
        joint_vel={names[i]: robot.data.joint_vel[0, i].item() for i in range(len(names))},
        root_z=float(raw.sim.data.qpos[0, 2]),
        root_quat=[float(x) for x in raw.sim.data.qpos[0, 3:7]],
        base_lin_vel=[float(x) for x in raw.sim.data.qvel[0, 0:3]],
    )


def make(stage, n, seed):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    cfg = module.make_credit_bridge_env_cfg(stage=stage)
    cfg.scene.num_envs = n
    cfg.seed = seed
    rl = module.make_credit_bridge_rl_cfg()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device='cuda:0'), clip_actions=rl.clip_actions)
    runner = load_runner_cls(module.TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    return env, rl, runner


@torch.no_grad()
def main():
    report = {}
    # 1) What the glide stage installs at reset (no step yet).
    env, rl, runner = make('glide', a.envs, 70601)
    raw = env.unwrapped; robot = raw.scene['robot']
    names = joint_names()
    report['joint_names'] = names
    report['glide_spawn_at_reset'] = snapshot(raw, robot, names)
    obs_glide = env.get_observations()
    report['glide_spawn_actor_obs'] = [float(x) for x in obs_glide['actor'][0]]
    # One step with the policy, to see the first commanded action.
    policy = evaluation_policy(runner.alg.actor, obs_glide, rl.actor, 14)
    action = policy(obs_glide)
    report['glide_spawn_first_action'] = [float(x) for x in action[0]]
    report['glide_prev_action_at_spawn'] = [float(x) for x in raw.action_manager.prev_action[0]]
    env.close()

    # 2) The real handoff frame in the full stage, and the action the policy takes there.
    env, rl, runner = make('full', a.envs, 70601)
    raw = env.unwrapped; robot = raw.scene['robot']
    cmd = raw.command_manager.get_term('twist')
    obs = env.get_observations()
    policy = evaluation_policy(runner.alg.actor, obs, rl.actor, 14)
    handoff = None
    for _ in range(raw.max_episode_length + 1):
        action = policy(obs)
        prev = raw.action_manager.prev_action.clone()
        obs, reward, done, extras = env.step(action)
        v = cmd.curriculum_values
        if bool((v['single_support'] > .5).any()) and handoff is None:
            idx = int((v['single_support'] > .5).nonzero()[0, 0])
            handoff = dict(env=idx, action_taken=[float(x) for x in action[idx]],
                           prev_action=[float(x) for x in prev[idx]],
                           actor_obs=[float(x) for x in obs['actor'][idx]])
            break
    assert handoff is not None
    idx = handoff['env']
    handoff['joint_pos'] = {names[i]: robot.data.joint_pos[idx, i].item() for i in range(len(names))}
    handoff['joint_vel'] = {names[i]: robot.data.joint_vel[idx, i].item() for i in range(len(names))}
    handoff['root_z'] = float(raw.sim.data.qpos[idx, 2])
    handoff['root_quat'] = [float(x) for x in raw.sim.data.qpos[idx, 3:7]]
    report['full_handoff_frame'] = handoff
    env.close()

    spawn = report['glide_spawn_at_reset']
    report['joint_pos_difference'] = {k: spawn['joint_pos'][k]-handoff['joint_pos'][k]
                                      for k in spawn['joint_pos']}
    report['joint_vel_difference'] = {k: spawn['joint_vel'][k]-handoff['joint_vel'][k]
                                      for k in spawn['joint_vel']}
    o1 = report['glide_spawn_actor_obs']; o2 = handoff['actor_obs']
    report['actor_obs_max_abs_diff'] = max(abs(x-y) for x, y in zip(o1, o2))
    report['actor_obs_index_of_max_diff'] = max(range(len(o1)), key=lambda i: abs(o1[i]-o2[i]))
    report['actor_obs_diff'] = [x-y for x, y in zip(o1, o2)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('SPAWN_VS_HANDOFF', json.dumps(dict(
        obs_max_abs_diff=report['actor_obs_max_abs_diff'],
        obs_worst_index=report['actor_obs_index_of_max_diff'],
        root_z_diff=spawn['root_z']-handoff['root_z']), ensure_ascii=False))
    print('joint_pos_difference:')
    for k, v in report['joint_pos_difference'].items():
        if abs(v) > 1e-3:
            print('   %-18s spawn-real %+.4f  (spawn %+.4f real %+.4f)'
                  % (k, v, spawn['joint_pos'][k], handoff['joint_pos'][k]))
    print('joint_vel_difference:')
    for k, v in report['joint_vel_difference'].items():
        if abs(v) > .05:
            print('   %-18s spawn-real %+.3f  (spawn %+.3f real %+.3f)'
                  % (k, v, spawn['joint_vel'][k], handoff['joint_vel'][k]))
    print('obs_diff (largest 12):')
    for i in sorted(range(len(o1)), key=lambda j: -abs(report['actor_obs_diff'][j]))[:12]:
        print('   idx %2d  spawn %+8.3f  real %+8.3f  diff %+8.3f' % (i, o1[i], o2[i], report['actor_obs_diff'][i]))


if __name__ == '__main__':
    main()
