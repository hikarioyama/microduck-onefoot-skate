"""Is the glide drag in the FROZEN experts, or in the learned correction?

Read-only probe, and the question that decides the next change. During the hold the
composition is

    out = onefoot + 0.25 * correction

because blend = 1 at phase = 1 and gate = 4*phase*(1-phase) + .25*blend = 0.25. The
trainable correction therefore has only a QUARTER of the authority during the glide.
If the ~0.34 m/s^2 glide deceleration belongs to the frozen one-foot expert's fixed
posture, no amount of training at 25% authority can remove it, and the fix is to raise
HOLD_AUTHORITY rather than to keep tuning the reward.

`--policy frozen-onefoot` drives the frozen expert directly (actor.onefoot(obs), which
applies the expert's own input mask and hold command). `--policy composed` drives the
trained bridge. Same spawn, same seed, same everything else, so the difference in glide
duration is attributable to the correction.

Reported per policy: entry speed, credible glide duration, and the implied mean
deceleration |a| = (v0 - 0.10) / duration, which is the number that has to fall for a
2.0 s glide.
"""
import argparse
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('credit_runner', ROOT/'local/train-credit-bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from mjlab.tasks.registry import load_runner_cls
from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import make_credit_bridge_env_cfg

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--episodes', type=int, default=64)
p.add_argument('--seed', type=int, default=70601)
p.add_argument('--policy', choices=('composed', 'frozen-onefoot'), default='frozen-onefoot')
p.add_argument('--speed', type=float, default=None, help='pin the injected speed')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert torch.cuda.is_available(), 'the probe needs the assigned GPU'


@torch.no_grad()
def run():
    cfg = make_credit_bridge_env_cfg(stage='glide')
    cfg.scene.num_envs = a.episodes
    cfg.seed = a.seed
    if a.speed is not None:
        cfg.events['curriculum_spawn'].params['speed_range'] = (a.speed, a.speed)
    rl = module.make_credit_bridge_rl_cfg()
    assert cfg.commands['twist'].reward_gamma == rl.algorithm.gamma
    assert all(not term.time_out for term in cfg.terminations.values())
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device='cuda:0'), clip_actions=rl.clip_actions)
    assert env.get_observations()['actor'].shape == (a.episodes, 61)
    runner = load_runner_cls(module.TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    try:
        raw = env.unwrapped
        obs = env.get_observations()
        actor = runner.alg.actor
        if a.policy == 'composed':
            policy = evaluation_policy(actor, obs, rl.actor, 14)
        else:
            def policy(o):                     # the frozen one-foot expert alone
                # obs is a TensorDict, not a dict, so isinstance(o, dict) is False.
                return actor.onefoot(o['actor'])
        n = a.episodes
        alive = torch.ones(n, dtype=torch.bool, device='cuda:0')
        best = torch.zeros(n, device='cuda:0')
        entry = torch.zeros(n, device='cuda:0')
        lengths = torch.zeros(n, dtype=torch.long, device='cuda:0')
        terms = {name: torch.zeros_like(alive) for name in raw.termination_manager.active_terms}
        for _step in range(raw.max_episode_length + 1):
            action = policy(obs)
            obs, _reward, done, extras = env.step(action)
            assert not extras['time_outs'].any(), 'No finite-horizon bootstrap allowed'
            cmd = raw.command_manager.get_term('twist')
            v = cmd.curriculum_values
            r = cmd.bridge_reward_values
            fresh = (v['single_support'] > .5) & (entry == 0)
            entry = torch.where(fresh & alive, r['entry_speed'], entry)
            best = torch.maximum(best, torch.where(alive, r['credible_best_s'], 0.))
            lengths += alive.long()
            for name in terms:
                terms[name] |= alive & done.bool() & raw.termination_manager.get_term(name)
            alive &= ~done.bool()
            if not bool(alive.any()):
                break
        dt = raw.step_dt
        v0 = float(entry.mean()); dur = float(best.mean())
        return dict(policy=a.policy, episodes=a.episodes, injected_speed=a.speed,
                    mean_entry_speed=v0, max_entry_speed=float(entry.max()),
                    mean_max_credible=dur, max_credible=float(best.max()),
                    implied_mean_decel=((v0 - 0.10) / dur) if dur > 0 else None,
                    credible_100=float(((best * dt) >= 1. - 1e-5).float().mean()),
                    credible_200=float(((best * dt) >= 2. - 1e-5).float().mean()),
                    mean_episode_s=float(lengths.float().mean() * dt),
                    termination_counts={k: int(vv.sum()) for k, vv in terms.items()})
    finally:
        env.close()


row = run()
print('EXPERT_VS_CORRECTION', json.dumps(row), flush=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint), rows=[row]),
                               indent=2, allow_nan=False) + '\n')
print('PROBE_DONE', a.output)
