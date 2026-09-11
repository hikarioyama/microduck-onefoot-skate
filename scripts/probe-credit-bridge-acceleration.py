"""Probe: what speed can the FROZEN public acceleration expert actually reach?

Read-only. Builds the `full` credit-bridge env with an arbitrarily long
acceleration window, so the lift blend stays 0 and the bridge's output is EXACTLY
the frozen public roller mean for the whole episode (bridge_mean with blend=0 and
gate=0 is the public endpoint, independent of the learned residual).

Purpose: separate two hypotheses for the low handoff speed observed at t=3.0 s
(com ~0.34 m/s in full-trace-u1000.json).

  (a) the public expert plateaus at ~0.35 m/s -> more acceleration time is useless
  (b) it is still accelerating, the 2 s window is simply too short -> extend it

No source file of the running trainer is touched.
"""
import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import (
    TASK, make_credit_bridge_env_cfg, make_credit_bridge_rl_cfg)

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--envs', type=int, default=16)
p.add_argument('--seed', type=int, default=70601)
p.add_argument('--horizon-s', type=float, default=8.)
p.add_argument('--acceleration-s', type=float, default=1e6)
p.add_argument('--lift-s', type=float, default=None)
p.add_argument('--target-speed', type=float, default=None)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
configure_torch_backends(allow_tf32=False)
torch.set_num_threads(4)


@torch.no_grad()
def main():
    cfg = make_credit_bridge_env_cfg(stage='full')
    cfg.scene.num_envs = a.envs
    cfg.seed = a.seed
    cfg.episode_length_s = a.horizon_s
    cmd_cfg = cfg.commands['twist']
    cmd_cfg.acceleration_s = a.acceleration_s
    if a.lift_s is not None:
        cmd_cfg.lift_s = a.lift_s
    if a.target_speed is not None:
        cmd_cfg.target_speed = a.target_speed
    rl = make_credit_bridge_rl_cfg()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device='cuda:0'), clip_actions=rl.clip_actions)
    runner = load_runner_cls(TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    raw = env.unwrapped
    obs = env.get_observations()
    policy = evaluation_policy(runner.alg.actor, obs, rl.actor, 14)
    rows = []
    try:
        for step in range(raw.max_episode_length):
            action = policy(obs)
            obs, reward, done, extras = env.step(action)
            cmd = raw.command_manager.get_term('twist')
            v = cmd.curriculum_values
            rows.append(dict(t=round((step + 1) * raw.step_dt, 4),
                             alive=int((~done).sum()),
                             phase=float(cmd.command[0, 1]),
                             cmd_speed=float(cmd.command[0, 0]),
                             obs_cmd_speed=float(obs['actor'][0, 48]),
                             speed=float(v['speed'].mean()),
                             speed_max=float(v['speed'].max()),
                             com_speed=float(v['com_speed'].mean()),
                             right_force=float(v['right_force'].mean()),
                             left_force=float(v['left_force'].mean()),
                             upright=float(v['upright'].mean()),
                             clearance=float(v['clearance'].mean())))
    finally:
        env.close()
    out = dict(checkpoint=str(a.checkpoint), envs=a.envs, seed=a.seed,
               horizon_s=a.horizon_s, acceleration_s=a.acceleration_s,
               target_speed=float(cmd_cfg.target_speed), peak_com_speed=max(r['com_speed'] for r in rows),
               rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('ACCEL_PROBE', json.dumps({k: out[k] for k in ('acceleration_s', 'target_speed', 'peak_com_speed')}))
    for r in rows:
        if round(r['t'] * 5) == r['t'] * 5:
            print('%5.2f phase=%.3f cmd=%.3f obs48=%.3f speed=%.3f com=%.3f alive=%2d rightF=%.2f lift=%.4f' % (
                r['t'], r['phase'], r['cmd_speed'], r['obs_cmd_speed'], r['speed'],
                r['com_speed'], r['alive'], r['right_force'], r['clearance']))


if __name__ == '__main__':
    main()
