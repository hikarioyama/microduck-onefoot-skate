"""Retain BOTH pretrained skills; learn only a bounded residual around them.

phase=0: official public acceleration, exactly, for any learned residual.
phase=1: preserved onefoot MLP with its original normalizer, plus a reduced,
         explicitly recorded residual share (HOLD_AUTHORITY), so the hold can
         be extended past the frozen expert's own ceiling.
0<phase<1: explicit smooth expert transition plus the learned bridge correction.
This is part of the exported policy, not a deployment-only output filter.
PPO's stochastic exploration still affects executed actions during training.

The residual's last layer is zero-initialized, so at the START of training both
endpoints equal the retained experts bit for bit. Relocating the hold share from
zero to HOLD_AUTHORITY is a deliberate, hashed and approved actor-forward
transition; record:
local/onefoot-bridge-curriculum/actor-transition-hold-authority.json
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from rsl_rl.models import MLPModel
from rsl_rl.models.mlp_model import _OnnxMLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.utils import unpad_trajectories
from mjlab_microduck.public_residual_policy import FrozenPublicRoller

ROOT = Path(__file__).resolve().parents[2]
ONEFOOT_PATH = ROOT/'logs/rsl_rl/onefoot/2026-09-10_19-55-19_history-pilot-control/balance-100/model_16699.pt'
ONEFOOT_SHA256 = 'f5127a67d39e1e1eb16599dfc392f9a8dbc702073fc69045432d1a2d9686748b'


class FrozenOnefoot(torch.nn.Module):
    """Load the actual previously evaluated onefoot skill, not a fresh head."""
    def __init__(self):
        super().__init__()
        if hashlib.sha256(ONEFOOT_PATH.read_bytes()).hexdigest() != ONEFOOT_SHA256:
            raise ValueError('Preserved onefoot checkpoint SHA256 mismatch')
        state = torch.load(ONEFOOT_PATH, map_location='cpu', weights_only=True)['actor_state_dict']
        self.obs_normalizer = EmpiricalNormalization(61)
        self.obs_normalizer.load_state_dict({name.removeprefix('obs_normalizer.'): value
            for name, value in state.items() if name.startswith('obs_normalizer.')}, strict=True)
        self.mlp = MLP(61, 14, [512, 256, 128], 'elu')
        self.mlp.load_state_dict({name.removeprefix('mlp.'): value
            for name, value in state.items() if name.startswith('mlp.')}, strict=True)
        # The frozen receiving skill is the HOLD expert. Do not feed it an
        # untrained phase=0 and call the resulting extrapolation its old skill.
        mask = torch.ones(1, 61); mask[:, 49] = 0.
        command = torch.zeros(1, 61); command[:, 49] = 1.
        self.register_buffer('input_mask', mask)
        self.register_buffer('hold_command', command)
        self.requires_grad_(False)

    def forward(self, raw):
        hold_obs = raw*self.input_mask+self.hold_command
        return self.mlp(self.obs_normalizer(hold_obs))


# The strict valid-glide predicate requires phase >= .99, so a correction gate of
# exactly 4*phase*(1-phase) left the learnable part with NO authority during the
# sustained hold. The frozen onefoot expert then capped the whole task at its own
# ceiling: 256 first episodes x 2 seeds from the balance pose gave credible 0.5 s
# 86.3/87.1% but credible 1.0 s and 2.0 s 0.0%, with a maximum credible glide of
# 0.88 s. The final 2 s goal was therefore unreachable by construction, not by
# undertraining. The hold keeps a reduced, recorded share of the residual instead.
HOLD_AUTHORITY = .25


def bridge_mean(raw: torch.Tensor, public: torch.Tensor, onefoot: torch.Tensor,
                correction: torch.Tensor) -> torch.Tensor:
    phase = raw[..., 49:50].clamp(0., 1.)
    blend = phase.square()*(3.-2.*phase)
    # TorchScript cannot close over a module-level float, so the hold authority is
    # inlined as the literal .25 and must stay equal to HOLD_AUTHORITY above;
    # tests/test_retained_skill_bridge.py ties the constant to the actual gate
    # behaviour, and the transition record documents the recorded value.
    # phase=0 stays exactly the public roller policy (the gate is zero there).
    gate = 4.*phase*(1.-phase)+.25*blend
    return (1.-blend)*public + blend*onefoot + gate*correction


class RetainedSkillBridgeModel(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_dim != 61 or self.mlp[-1].out_features != 14:
            raise ValueError('Retained-skill bridge requires raw61D and14D actions')
        self.public = FrozenPublicRoller()
        self.onefoot = FrozenOnefoot()
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        correction = self.mlp(self.obs_normalizer(raw))
        output = bridge_mean(raw, self.public(raw), self.onefoot(raw), correction)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_onnx(self, verbose=False):
        return _OnnxRetainedBridge(self, verbose)

    def as_jit(self):
        return _TorchRetainedBridge(self)


class _OnnxRetainedBridge(_OnnxMLPModel):
    def __init__(self, model, verbose=False):
        super().__init__(model, verbose)
        self.public = deepcopy(model.public)
        self.onefoot = deepcopy(model.onefoot)

    def forward(self, obs):
        correction = self.mlp(self.obs_normalizer(obs))
        return self.deterministic_output(bridge_mean(obs, self.public(obs), self.onefoot(obs), correction))


class _TorchRetainedBridge(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.obs_normalizer = deepcopy(model.obs_normalizer)
        self.mlp = deepcopy(model.mlp)
        self.public = deepcopy(model.public)
        self.onefoot = deepcopy(model.onefoot)

    def forward(self, obs):
        correction = self.mlp(self.obs_normalizer(obs))
        return bridge_mean(obs, self.public(obs), self.onefoot(obs), correction)

    @torch.jit.export
    def reset(self):
        pass


def retained_skills_digest(actor):
    digest = hashlib.sha256()
    for expert in ('public', 'onefoot'):
        for name, value in sorted(getattr(actor, expert).state_dict().items()):
            digest.update((expert+'.'+name).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
