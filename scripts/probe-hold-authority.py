"""How much does the glide improve if the correction keeps MORE authority?

Read-only probe. It does NOT change any pinned source: it recomposes the bridge inside
this process with a chosen hold-authority coefficient and measures the glide.

Why this is the question. The composition at phase = 1 is

    out = onefoot + A * correction,     A = HOLD_AUTHORITY = 0.25

Measured on the frozen u10000 glide checkpoint at a pinned 0.72 m/s entry
(local/probe-expert-vs-correction.py):

    frozen one-foot expert alone   mean credible 0.022 s, fell 59/64
    the trained bridge             mean credible 1.50 s,  max 2.08 s, fell 2/64

The preserved expert cannot glide at all from a fast spawn, so the entire glide is
being produced by the correction - while the expert still supplies 100% of the base
action and the correction is scaled by a quarter. The correction is therefore fighting
a base that is, for this task, close to useless, and the ~0.34 m/s^2 residual drag that
blocks the 2.0 s glide has no identified mechanism. This probe asks whether lack of
authority is that mechanism.

`--authorities` sweeps A. A = 0.25 reproduces the shipped behaviour and is the control.
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
from mjlab_microduck.checkpoint_safety import safe_runner_load
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import make_credit_bridge_env_cfg

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--episodes', type=int, default=64)
p.add_argument('--seed', type=int, default=70601)
p.add_argument('--authorities', type=float, nargs='+', default=[.25, .50, .75, 1.0])
p.add_argument('--speed', type=float, default=0.72)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert torch.cuda.is_available(), 'the probe needs the assigned GPU'


def composed_policy(actor, authority):
    """Recompose exactly as RetainedSkillBridgeModel.bridge_mean does, with A overridden."""
    def policy(obs):
        with torch.no_grad():
            raw = torch.cat([obs[g] for g in actor.obs_groups], dim=-1)
            public = actor.public(raw)
            onefoot = actor.onefoot(raw)
            correction = actor.mlp(actor.obs_normalizer(raw))
        phase = raw[..., 49:50].clamp(0., 1.)
        blend = phase.square() * (3. - 2. * phase)
        gate = 4. * phase * (1. - phase) + authority * blend
        return (1. - blend) * public + blend * onefoot + gate * correction
    return policy


@torch.no_grad()
def run_one(authority):
    cfg = make_credit_bridge_env_cfg(stage='glide')
    cfg.scene.num_envs = a.episodes
    cfg.seed = a.seed
    cfg.events['curriculum_spawn'].params['speed_range'] = (a.speed, a.speed)
    rl = module.make_credit_bridge_rl_cfg()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device='cuda:0'), clip_actions=rl.clip_actions)
    runner = load_runner_cls(module.TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    try:
        raw = env.unwrapped
        obs = env.get_observations()
        policy = composed_policy(runner.alg.actor, authority)
        n = a.episodes
        alive = torch.ones(n, dtype=torch.bool, device='cuda:0')
        best = torch.zeros(n, device='cuda:0')
        entry = torch.zeros(n, device='cuda:0')
        lengths = torch.zeros(n, dtype=torch.long, device='cuda:0')
        terms = {name: torch.zeros_like(alive) for name in raw.termination_manager.active_terms}
        for _step in range(raw.max_episode_length + 1):
            obs, _reward, done, extras = env.step(policy(obs))
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
        return dict(authority=authority, injected_speed=a.speed, episodes=a.episodes,
                    mean_entry_speed=v0, max_entry_speed=float(entry.max()),
                    mean_max_credible=dur, max_credible=float(best.max()),
                    implied_mean_decel=((v0 - 0.10) / dur) if dur > 0 else None,
                    credible_100=float(((best * dt) >= 1. - 1e-5).float().mean()),
                    credible_200=float(((best * dt) >= 2. - 1e-5).float().mean()),
                    mean_episode_s=float(lengths.float().mean() * dt),
                    nan_episodes=int((~torch.isfinite(best)).sum()),
                    termination_counts={k: int(vv.sum()) for k, vv in terms.items()})
    finally:
        env.close()


rows = []
for auth in a.authorities:
    row = run_one(auth)
    rows.append(row)
    print('HOLD_AUTHORITY_PROBE', json.dumps(row), flush=True)

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint), rows=rows), indent=2,
                               allow_nan=False) + '\n')
print('PROBE_DONE', a.output)
