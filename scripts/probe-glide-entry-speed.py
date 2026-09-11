"""Does a faster glide entry buy a longer glide? Sweep the spawn speed, change nothing else.

Read-only probe. The `glide` stage spawns at a measured handoff state with a root
forward speed drawn from `speed_range=(.45,.58)`. That range was taken from the
handoff the frozen bridge actually produced, so the stage has never been asked to
glide FASTER than the policy already did.

The measured physics says it should work: above 0.4 m/s the glide deceleration is
essentially independent of speed (median -0.21 to -0.24 m/s^2), and the credible
time from entry v0 is (v0 - 0.10)/|a|. At v0 = 0.55 and |a| = 0.22 that is 2.05 s.

Two things are varied independently because they are separate hypotheses:

  --wheel-mode recorded   keep the pose's recorded wheel velocities (what the stage
                          does today; the support wheels spin as if v = 0.55 m/s
                          whatever speed is injected)
  --wheel-mode scaled     make the two SUPPORT wheels spin at v/r of the injected
                          speed, leaving the two unloaded swing wheels at their
                          recorded near-still values. This is what a real handoff at
                          that speed looks like.

The valid predicate rejects root speeds outside (0.10, 0.80) m/s, so the sweep stays
inside that window; entry speed is also the ceiling on what any glide can be.

Nothing here mutates a pinned source. The speed override is applied to the freshly
built cfg object inside this process only.
"""
import argparse
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('credit_runner', ROOT/'local/train-credit-bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from mjlab.tasks.registry import load_runner_cls
from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load
from mjlab_microduck.tasks import mdp as bridge_mdp
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import make_credit_bridge_env_cfg

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--episodes', type=int, default=64)
p.add_argument('--seed', type=int, default=70601)
p.add_argument('--speeds', type=float, nargs='+', default=[.45, .55, .65, .72, .78])
p.add_argument('--wheel-mode', choices=('recorded', 'scaled'), default='scaled')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

assert torch.cuda.is_available(), 'the probe needs the assigned GPU'


@torch.no_grad()
def run_one(speed):
    cfg = make_credit_bridge_env_cfg(stage='glide')
    cfg.scene.num_envs = a.episodes
    cfg.seed = a.seed
    params = cfg.events['curriculum_spawn'].params
    params['speed_range'] = (speed, speed)
    if a.wheel_mode == 'scaled':
        # The support wheels must roll with the injected speed; the recorded values
        # were measured at v = 0.51 m/s. The unloaded swing wheels stay as recorded.
        jv = dict(params['joint_vel'])
        for name in ('passive_LF_wheel', 'passive_LR_wheel'):
            jv[name] = speed / bridge_mdp.CURRICULUM_WHEEL_RADIUS
        params['joint_vel'] = jv
    rl = module.make_credit_bridge_rl_cfg()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device='cuda:0'), clip_actions=rl.clip_actions)
    runner = load_runner_cls(module.TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    try:
        raw = env.unwrapped
        obs = env.get_observations()
        policy = evaluation_policy(runner.alg.actor, obs, rl.actor, 14)
        n = a.episodes
        alive = torch.ones(n, dtype=torch.bool, device='cuda:0')
        nan = torch.zeros_like(alive)
        credible = torch.zeros(n, device='cuda:0')
        best = torch.zeros(n, device='cuda:0')
        single = torch.zeros(n, device='cuda:0')
        run_cred = torch.zeros(n, device='cuda:0')
        run_single = torch.zeros(n, device='cuda:0')
        lengths = torch.zeros(n, dtype=torch.long, device='cuda:0')
        entry = torch.zeros(n, device='cuda:0')
        terms = {name: torch.zeros_like(alive) for name in raw.termination_manager.active_terms}
        for _step in range(raw.max_episode_length + 1):
            action = policy(obs)
            obs, _reward, done, extras = env.step(action)
            assert not extras['time_outs'].any(), 'No finite-horizon bootstrap allowed'
            cmd = raw.command_manager.get_term('twist')
            r = cmd.bridge_reward_values
            v = cmd.curriculum_values
            fresh = (v['single_support'] > .5) & (entry == 0)
            entry = torch.where(fresh & alive, r['entry_speed'], entry)
            run_cred = torch.where((r['credible'] > .5) & alive, run_cred + 1, torch.zeros_like(run_cred))
            run_single = torch.where((v['single_support'] > .5) & alive, run_single + 1, torch.zeros_like(run_single))
            credible = torch.maximum(credible, run_cred)
            single = torch.maximum(single, run_single)
            best = torch.maximum(best, torch.where(alive, r['credible_best_s'], 0.))
            lengths += alive.long()
            nan |= alive & raw.termination_manager.get_term('nan_state')
            for name in terms:
                terms[name] |= alive & done.bool() & raw.termination_manager.get_term(name)
            alive &= ~done.bool()
            if not bool(alive.any()):
                break
        dt = raw.step_dt
        return dict(
            injected_speed=speed, wheel_mode=a.wheel_mode, episodes=a.episodes,
            mean_entry_speed=float(entry.mean()), mean_max_credible=float(best.mean()),
            max_credible=float(best.max()),
            credible_100=float(((credible * dt) >= 1. - 1e-5).float().mean()),
            credible_150=float(((credible * dt) >= 1.5 - 1e-5).float().mean()),
            credible_200=float(((credible * dt) >= 2. - 1e-5).float().mean()),
            credible_200_count=int(((credible * dt) >= 2. - 1e-5).sum()),
            max_single_support=float((single * dt).max()),
            single_200_count=int(((single * dt) >= 2. - 1e-5).sum()),
            mean_episode_s=float(lengths.float().mean() * dt),
            nan_episodes=int(nan.sum()),
            termination_counts={k: int(vv.sum()) for k, vv in terms.items()},
        )
    finally:
        env.close()


results = []
for s in a.speeds:
    row = run_one(s)
    results.append(row)
    print('GLIDE_SPEED_PROBE', json.dumps(row), flush=True)

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint), seed=a.seed, rows=results),
                               indent=2, allow_nan=False) + '\n')
print('PROBE_DONE', a.output)
