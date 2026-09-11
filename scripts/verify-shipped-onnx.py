"""Check that a shipped ONNX policy reproduces the training-time actor in the real env.

`tests/test_retained_skill_bridge.py::test_standard_exports_include_both_skills_and_bridge`
proves the EXPORTER is faithful, using a synthetic actor. It does not prove that the
`.onnx` files actually shipped in `checkpoints/` match the recorded `.pt`
checkpoints. This probe closes that gap: it builds the real environment, loads the
recorded checkpoint through the same safe loader the trainer uses, takes the
observations the environment produces, and compares

    onnx(obs)                      vs      actor(obs)      (deterministic mean)

on those real observations. Run it on CPU:

    CUDA_VISIBLE_DEVICES='' uv run --locked python scripts/verify-shipped-onnx.py
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

CASES = (
    # (label, onnx under the deliverable repo, checkpoint, stage)
    ('glide-02', 'checkpoints/glide/policy-glide-02.onnx', 'checkpoints/glide/model_10499.pt', 'glide'),
    ('glide-01', 'checkpoints/glide/policy.onnx', 'checkpoints/glide/model_9499.pt', 'glide'),
)


def load_trainer():
    spec = importlib.util.spec_from_file_location('credit_runner', ROOT/'local/train-credit-bridge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deliverable', type=Path, default=ROOT,
                        help='repository holding checkpoints/ (default: this checkout)')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--out', type=Path, default=None,
                        help='write a JSON report here')
    args = parser.parse_args()

    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load
    from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import (
        TASK, make_credit_bridge_env_cfg, make_credit_bridge_rl_cfg)

    trainer = load_trainer()
    report = {'cases': {}}
    for label, onnx_rel, ckpt_rel, stage in CASES:
        onnx_path = args.deliverable/onnx_rel
        ckpt_path = args.deliverable/ckpt_rel

        rl = make_credit_bridge_rl_cfg()
        cfg = make_credit_bridge_env_cfg(stage=stage)
        cfg.scene.num_envs = args.num_envs
        env = ManagerBasedRlEnv(cfg, device='cpu')
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = load_runner_cls(TASK)(wrapped, deepcopy(asdict(rl)),
                                       log_dir=str(ROOT/'local/onefoot-onnx-verify'),
                                       device='cpu')
        runner.logger.writer = None
        safe_runner_load(runner, str(ckpt_path), strict=True, map_location='cpu')

        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        policy = evaluation_policy(runner.alg.actor, obs, rl.actor, 14)
        raw = obs['actor'].detach().cpu()
        with torch.no_grad():
            # The actor takes the whole TensorDict (it selects its obs groups).
            expected = policy(obs).detach().cpu().numpy()

        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        rows = [session.run(['actions'], {'obs': raw[i:i+1].numpy().astype(np.float32)})[0][0]
                for i in range(raw.shape[0])]
        got = np.stack(rows)

        # The exporter is allowed to differ from the Torch path by float noise only.
        delta = float(np.abs(got - expected).max())
        close = bool(np.allclose(got, expected, atol=3e-6, rtol=2e-5))
        report['cases'][label] = {
            'onnx': onnx_rel,
            'onnx_sha256': hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            'checkpoint': ckpt_rel,
            'checkpoint_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
            'stage': stage,
            'num_envs': args.num_envs,
            'obs_shape': list(raw.shape),
            'max_abs_difference': delta,
            'allclose_atol_3e-6_rtol_2e-5': close,
            'onnx_action_abs_max': float(np.abs(got).max()),
            'actor_action_abs_max': float(np.abs(expected).max()),
            'phase_slot': float(raw[0, 49]),
            'command_slot': float(raw[0, 48]),
        }
        print(f'{label}: max|onnx - actor| = {delta:.3e}  allclose={close}  '
              f'|action|max = {float(np.abs(got).max()):.3f}  phase = {float(raw[0,49]):.2f}')

    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
        print(f'wrote {args.out}')
    failed = [k for k, v in report['cases'].items() if not v['allclose_atol_3e-6_rtol_2e-5']]
    if failed:
        raise SystemExit(f'Shipped ONNX does not reproduce the actor: {failed}')


if __name__ == '__main__':
    main()
