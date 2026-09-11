"""Verify retained onefoot skill and initial full-sequence bridge, no training."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import torch
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import safe_runner_load
from mjlab_microduck.retained_skill_bridge import ONEFOOT_PATH, ONEFOOT_SHA256, retained_skills_digest
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import TASK, make_waist_onefoot_env_cfg, make_waist_onefoot_rl_cfg

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retained_bridge_diagnostic_base', ROOT/'local/evaluate-public-roller-connection.py')
diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)


def bridge_rl_cfg():
    cfg = make_waist_onefoot_rl_cfg()
    cfg.actor.class_name = 'mjlab_microduck.retained_skill_bridge:RetainedSkillBridgeModel'
    cfg.actor.distribution_cfg = dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',
                                     init_std=.08, std_type='log')
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('hold-retention', 'full-sequence'), required=True)
    p.add_argument('--episodes', type=int, default=256)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.episodes <= 256:
        p.error('At most 256 first episodes per diagnostic batch')
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID' and os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
    assert torch.cuda.device_count() == 1 and '5070 Ti' in torch.cuda.get_device_name(0)
    configure_torch_backends(allow_tf32=False); torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    main_hash = hashlib.sha256(diag.STATE.read_bytes()).hexdigest()
    checkpoint_hash = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None
    if a.mode == 'hold-retention':
        cfg = make_waist_onefoot_env_cfg(stage='balance-100')
        cfg.scene.num_envs = a.episodes; cfg.seed = a.seed
        cfg.sim.nan_guard.enabled = True
        cfg.metrics['onefoot/diagnostic_valid'] = MetricsTermCfg(func=mdp.observe_curriculum_onefoot_support_kinematics, reduce='last')
    else:
        cfg = diag.environment_cfg(a.episodes, a.seed, 6., 2., 'home')
    rl = bridge_rl_cfg()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device='cuda:0'), clip_actions=rl.clip_actions)
    try:
        raw = env.unwrapped; obs = env.get_observations()
        initial_velocity_max = float(raw.sim.data.qvel.abs().max())
        if a.mode == 'full-sequence':
            assert initial_velocity_max == 0.
        with torch.random.fork_rng(devices=[0]):
            runner = load_runner_cls(TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
            if a.checkpoint:
                safe_runner_load(runner, a.checkpoint, load_cfg={'actor': True}, strict=True, map_location='cuda:0')
            actor = runner.get_inference_policy(device='cuda:0')
            original = MLPModel(obs=obs, obs_groups={'actor': ['actor']}, obs_set='actor',
                                output_dim=14, hidden_dims=[512, 256, 128], activation='elu',
                                obs_normalization=True).to('cuda:0')
            source = torch.load(ONEFOOT_PATH, map_location='cuda:0', weights_only=True)['actor_state_dict']
            original.load_state_dict({k:v for k,v in source.items() if not k.startswith('distribution.')}, strict=True)
            original.eval()
        digest = retained_skills_digest(actor)
        alive = torch.ones(a.episodes, dtype=torch.bool, device='cuda:0')
        nan = torch.zeros_like(alive); best = torch.zeros(a.episodes, device='cuda:0')
        prefix_difference = 0.; tail_difference = 0.
        prefix_samples = 0; tail_samples = 0
        with torch.no_grad():
            for step in range(raw.max_episode_length+1):
                actions = actor(obs)
                phase = obs['actor'][:, 49]
                prefix = (phase == 0) & alive; tail = (phase == 1) & alive
                if prefix.any():
                    delta = (actions-actor.public(obs['actor'])).abs().amax(1)
                    prefix_difference = max(prefix_difference, float(delta[prefix].max()))
                    prefix_samples += int(prefix.sum())
                if tail.any():
                    delta = (actions-original(obs)).abs().amax(1)
                    tail_difference = max(tail_difference, float(delta[tail].max()))
                    tail_samples += int(tail.sum())
                if not torch.isfinite(actions).all():
                    raise RuntimeError('Nonfinite verification action')
                obs, _, done, _ = env.step(actions)
                cmd = raw.command_manager.get_term('twist')
                best = torch.maximum(best, torch.where(alive, cmd.curriculum_values['best_dwell'], 0.))
                nan |= alive & raw.termination_manager.get_term('nan_state')
                alive &= ~done.bool()
                if not bool(alive.any()):
                    break
        assert retained_skills_digest(actor) == digest
        assert hashlib.sha256(diag.STATE.read_bytes()).hexdigest() == main_hash
        assert hashlib.sha256(ONEFOOT_PATH.read_bytes()).hexdigest() == ONEFOOT_SHA256
        if a.checkpoint:
            assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
        assert prefix_difference <= 1e-6 and tail_difference <= 1e-6
        report = dict(mode=a.mode, episodes=a.episodes, seed=a.seed,
                      onefoot_source=str(ONEFOOT_PATH), onefoot_sha256=ONEFOOT_SHA256,
                      bridge_checkpoint=str(a.checkpoint) if a.checkpoint else None,
                      bridge_checkpoint_sha256=checkpoint_hash,
                      initial_velocity_max_abs=initial_velocity_max,
                      initial_assistance=a.mode=='hold-retention',
                      mean_best_glide_s=float(best.mean()), max_glide_s=float(best.max()),
                      success_050_rate=float(((best >= .5-1e-5)&~nan).float().mean()),
                      success_100_rate=float(((best >= 1.-1e-5)&~nan).float().mean()),
                      success_200_rate=float(((best >= 2.-1e-5)&~nan).float().mean()),
                      nan_episodes=int(nan.sum()), prefix_output_difference_max_abs=prefix_difference,
                      tail_output_difference_max_abs=tail_difference, prefix_comparisons=prefix_samples,
                      tail_comparisons=tail_samples, frozen_skills_sha256=digest, main_state_unchanged=True,
                      gate_eligible=False,
                      scope='Retention verification or untrained bridge baseline, not stage promotion')
        np.savez_compressed(a.output.with_suffix('.npz'), best_glide_s=best.cpu().numpy(), nan=nan.cpu().numpy())
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print('RETAINED_SKILLS_VERIFIED', json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
