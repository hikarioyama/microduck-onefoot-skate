"""Measure the real single-support glide-entry pose, for an assisted glide stage.

Read-only. Loads a saved credit-bridge checkpoint, rolls out the `full` stage from
rest, and records the servo-joint configuration at the exact frame where loaded
left-only support first holds (the same predicate the reward uses). The median of
those frames becomes a new measured spawn pose, in the same shape as the existing
`balance`/`unload`/`transfer_near`/`transfer` entries, so the glide can be trained
from its own distribution instead of replaying the frozen 2 s acceleration prefix.

Also reports the root orientation split into roll/pitch, because
`reset_curriculum_onefoot` can currently only inject a roll (it zeroes the rest of
the quaternion), so a large pitch would need that to be stated explicitly.
"""
import argparse
import importlib.util
import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
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
p.add_argument('--envs', type=int, default=64)
p.add_argument('--seed', type=int, default=70601)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
configure_torch_backends(allow_tf32=False)
torch.set_num_threads(4)


@torch.no_grad()
def measure():
    env, rl = module.environment(a.envs, a.seed, 'full')
    runner = load_runner_cls(module.TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    raw = env.unwrapped
    robot = raw.scene['robot']
    cmd = raw.command_manager.get_term('twist')
    servo_ids, servo_names = robot.find_joints(r'^(?!passive_).*')
    # Every non-free joint, including the four passive wheels: the glide entry has the
    # two support wheels rolling near v/r while the two unloaded swing wheels are nearly
    # still, so blindly setting all four to v/radius is a large state error.
    all_ids, all_names = robot.find_joints(r'^(?!trunk_base_freejoint$).*')
    obs = env.get_observations()
    policy = evaluation_policy(runner.alg.actor, obs, rl.actor, 14)
    n = a.envs
    captured = torch.zeros(n, dtype=torch.bool, device='cuda:0')
    poses = torch.zeros(n, len(servo_ids), device='cuda:0')
    vels = torch.zeros(n, len(servo_ids), device='cuda:0')
    all_vels = torch.zeros(n, len(all_ids), device='cuda:0')
    roots = torch.zeros(n, 3, device='cuda:0')       # roll, pitch, yaw
    speeds = torch.zeros(n, device='cuda:0')
    gravity = torch.zeros(n, 3, device='cuda:0')
    try:
        for _ in range(raw.max_episode_length + 1):
            action = policy(obs)
            obs, reward, done, extras = env.step(action)
            v = cmd.curriculum_values
            now = (v['single_support'] > .5) & ~captured
            if bool(now.any()):
                quat = robot.data.root_link_quat_w
                w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
                pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
                yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                poses[now] = robot.data.joint_pos[now][:, servo_ids]
                vels[now] = robot.data.joint_vel[now][:, servo_ids]
                all_vels[now] = robot.data.joint_vel[now][:, all_ids]
                roots[now, 0] = roll[now]; roots[now, 1] = pitch[now]; roots[now, 2] = yaw[now]
                speeds[now] = v['speed'][now]
                gravity[now] = robot.data.projected_gravity_b[now]
                captured |= now
            if bool(captured.all()):
                break
            if not bool((~done).any()):
                break
        got = int(captured.sum())
        if got == 0:
            raise SystemExit('No environment ever reached loaded left-only support')
        pose = poses[captured].cpu().numpy()
        roll = roots[captured, 0].cpu().numpy()
        pitch = roots[captured, 1].cpu().numpy()
        speed = speeds[captured].cpu().numpy()
        grav = gravity[captured].cpu().numpy()
        median = np.median(pose, axis=0)
        median_vel = np.median(vels.cpu().numpy(), axis=0)
        median_all_vel = np.median(all_vels.cpu().numpy(), axis=0)
        # Required root height for the median joint configuration. The curriculum
        # poses file is grounded on the TYRE MESH VERTICES of scene_rollers.xml
        # (that is what tests/test_onefoot_curriculum.py checks), not on body
        # origins, so compute it the same way: root at z=0, forward kinematics,
        # then lift by the lowest mesh vertex. Also report the left and right
        # pairs separately, because a glide spawn is single-support by definition
        # and the swing skate is expected to be off the ground.
        model = mujoco.MjModel.from_xml_path(
            str(ROOT/'src/mjlab_microduck/robot/microduck/scene_rollers.xml'))
        data = mujoco.MjData(model)
        key = model.key('STAND').id
        mujoco.mj_resetDataKeyframe(model, data, key)
        data.qpos[2] = 0.
        data.qpos[3:7] = [1., 0., 0., 0.]
        for name, value in zip(servo_names, median):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        wheel_min = {}
        for body in ('tire', 'tire_2', 'tire_3', 'tire_4'):
            bid = model.body(body).id
            gid = next(g for g in range(model.ngeom)
                       if model.geom_bodyid[g] == bid and model.geom_contype[g])
            mesh = model.geom_dataid[gid]
            off, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = (model.mesh_vert[off:off+num] @ data.geom_xmat[gid].reshape(3, 3).T
                     + data.geom_xpos[gid])
            wheel_min[body] = float(verts[:, 2].min())
        lowest = min(wheel_min.values())
        report = dict(
            checkpoint=str(a.checkpoint), envs=a.envs, seed=a.seed, captured=got,
            joint_names=list(servo_names),
            median_pose={name: float(value) for name, value in zip(servo_names, median)},
            median_joint_vel={name: float(value) for name, value in zip(servo_names, median_vel)},
            median_all_joint_vel={name: float(value) for name, value in zip(all_names, median_all_vel)},
            all_joint_vel_abs_p90={name: float(value) for name, value in
                                   zip(all_names, np.percentile(np.abs(all_vels.cpu().numpy()),
                                                                90, axis=0))},
            joint_vel_abs_p90={name: float(value) for name, value in
                               zip(servo_names, np.percentile(np.abs(vels.cpu().numpy()),
                                                              90, axis=0))},
            pose_spread_p95={name: float(value) for name, value in
                             zip(servo_names, np.percentile(np.abs(pose-median), 95, axis=0))},
            wheel_mesh_min_z_at_root_zero=wheel_min,
            wheel_lift={body: float(value-lowest) for body, value in wheel_min.items()},
            lowest_wheel_mesh_min_z=lowest,
            root_z_to_ground_the_lowest=-lowest,
            speed=dict(mean=float(speed.mean()), p50=float(np.median(speed)),
                       p90=float(np.percentile(speed, 90)), min=float(speed.min()),
                       max=float(speed.max())),
            roll=dict(mean=float(roll.mean()), p50=float(np.median(roll)),
                      p90=float(np.percentile(np.abs(roll), 90))),
            pitch=dict(mean=float(pitch.mean()), p50=float(np.median(pitch)),
                       p90=float(np.percentile(np.abs(pitch), 90))),
            yaw=dict(p90=float(np.percentile(np.abs(roots[captured, 2].cpu().numpy()), 90))),
            projected_gravity=dict(x=float(grav[:, 0].mean()), y=float(grav[:, 1].mean()),
                                   z=float(grav[:, 2].mean())),
            swing_lift=dict(
                left=max(wheel_min['tire'], wheel_min['tire_2'])-lowest,
                right=max(wheel_min['tire_3'], wheel_min['tire_4'])-lowest),
        )
    finally:
        env.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('GLIDE_POSE', json.dumps({k: report[k] for k in
          ('captured', 'wheel_lift', 'root_z_to_ground_the_lowest', 'swing_lift',
           'speed', 'roll', 'pitch', 'yaw', 'projected_gravity')}, ensure_ascii=False))
    print('GLIDE_POSE_JOINTS', json.dumps(report['median_pose'], ensure_ascii=False))


if __name__ == '__main__':
    measure()
