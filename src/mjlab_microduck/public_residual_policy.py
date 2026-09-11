"""Frozen public skating mean plus a phase-gated trainable connection residual.

raw61D -> HOME-relative14D, no new sensors or actuator filters. The public
normalizer and weights are frozen. At phase=0 the deterministic mean remains
the official acceleration policy. PPO exploration noise still applies during
training (including the prefix); deployment/evaluation uses the mean.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from rsl_rl.models import MLPModel
from rsl_rl.models.mlp_model import _OnnxMLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PATH = ROOT/'local/onefoot-public-acceleration/assets/088524a64e2557dc453256b6071dbb9d23888802/roller.onnx'
PUBLIC_SHA256 = 'cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd'


class FrozenPublicRoller(torch.nn.Module):
    """Exact pinned ONNX operations, batched in Torch for PPO simulation.

This imports the published graph, not an approximate missing normalizer. The
whole resulting actor is exported ONLY through the standard runner exporter.
"""
    def __init__(self):
        super().__init__()
        if hashlib.sha256(PUBLIC_PATH.read_bytes()).hexdigest() != PUBLIC_SHA256:
            raise ValueError('Pinned public roller policy SHA256 mismatch')
        graph = onnx.load(str(PUBLIC_PATH), load_external_data=False).graph
        if [n.op_type for n in graph.node] != ['Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']:
            raise ValueError('Unsupported public ONNX graph')
        arrays = {v.name: numpy_helper.to_array(v).copy() for v in graph.initializer}
        self.register_buffer('mean', torch.from_numpy(arrays['obs_normalizer._mean']))
        denominator = torch.from_numpy(arrays[graph.node[1].input[1]])
        if not torch.isfinite(denominator).all() or not (denominator > 0).all():
            raise ValueError('Invalid public normalization denominator')
        self.register_buffer('denominator', denominator)
        input_scale = torch.ones(1, 61)
        input_scale[:, 34:48] = 1./.8
        input_scale[:, 48:] = 0.
        command = torch.zeros(1, 61); command[:, 48] = .6
        self.register_buffer('input_scale', input_scale)
        self.register_buffer('command', command)
        self.mlp = MLP(61, 14, [512, 256, 128], 'elu')
        self.mlp.load_state_dict({name.removeprefix('mlp.'): torch.from_numpy(array)
                                 for name, array in arrays.items() if name.startswith('mlp.')}, strict=True)
        self.requires_grad_(False)

    def forward(self, raw):
        public_obs = raw*self.input_scale + self.command
        normalized = (public_obs-self.mean)/self.denominator
        return self.mlp(normalized)*.8


class PublicResidualModel(MLPModel):
    """Preserve acceleration; learn the transfer/tail without an unsafe hot-swap."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_dim != 61 or self.mlp[-1].out_features != 14:
            raise ValueError('Public residual actor requires raw61D and14D actions')
        self.public = FrozenPublicRoller()
        # Only the new correction starts at zero; pretrained skating is retained.
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        phase = raw[..., 49:50].clamp(0., 1.)
        correction = self.mlp(self.obs_normalizer(raw))
        output = self.public(raw) + phase*correction
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_onnx(self, verbose=False):
        return _OnnxPublicResidual(self, verbose)

    def as_jit(self):
        return _TorchPublicResidual(self)


class _OnnxPublicResidual(_OnnxMLPModel):
    def __init__(self, model, verbose=False):
        super().__init__(model, verbose)
        self.public = deepcopy(model.public)

    def forward(self, obs):
        phase = obs[..., 49:50].clamp(0., 1.)
        correction = self.mlp(self.obs_normalizer(obs))
        return self.deterministic_output(self.public(obs) + phase*correction)


class _TorchPublicResidual(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.obs_normalizer = deepcopy(model.obs_normalizer)
        self.mlp = deepcopy(model.mlp)
        self.public = deepcopy(model.public)

    def forward(self, obs):
        phase = obs[..., 49:50].clamp(0., 1.)
        return self.public(obs) + phase*self.mlp(self.obs_normalizer(obs))

    @torch.jit.export
    def reset(self):
        pass


def frozen_public_digest(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.public.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
