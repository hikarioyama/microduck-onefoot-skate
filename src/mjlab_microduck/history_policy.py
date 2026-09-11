"""RSL-RL GRU with a direct current-observation path for controlled warm starts.

Raw observations stay 61D; internal latent is [normalized_current, GRU_history].
An old MLP embeds without tanh rescaling: copy its current-input columns and
zero only the new history columns. History can then be learned without forcing
an initial policy change. Canonical mjlab export still bakes normalization;
ONNX additionally requires h_in/h_out, carried/reset by the inference runtime.
"""
import torch
from rsl_rl.models import MLPModel,RNNModel
from rsl_rl.models.rnn_model import _OnnxRNNModel,_TorchGRUModel
from rsl_rl.utils import unpad_trajectories


class HistoryGRUModel(RNNModel):
    """Current proprioception plus trainable recurrent features, not new sensors."""

    def _get_latent_dim(self):
        return self.obs_dim+self.latent_dim

    def get_latent(self,obs,masks=None,hidden_state=None):
        current=MLPModel.get_latent(self,obs)
        memory=self.rnn(current,masks,hidden_state).squeeze(0)
        if masks is not None:
            # RSL's RNN unpads its output. Unpad the direct path in exactly the
            # same order; concatenating padded current inputs would corrupt PPO.
            current=unpad_trajectories(current,masks).squeeze(0)
        return torch.cat((current,memory),dim=-1)

    def as_onnx(self,verbose=False):
        if not isinstance(self.rnn.rnn,torch.nn.GRU):
            raise ValueError('HistoryGRUModel currently exports GRU only')
        return _OnnxHistoryGRU(self,verbose)

    def as_jit(self):
        if not isinstance(self.rnn.rnn,torch.nn.GRU):
            raise ValueError('HistoryGRUModel currently exports GRU only')
        return _TorchHistoryGRU(self)


class _OnnxHistoryGRU(_OnnxRNNModel):
    # Reuse the pinned RSL-RL exporter protocol, normalizer copy, names and
    # hidden-state dimensions. Export is invoked via runner.export_policy_to_onnx.
    def forward(self,obs,h_in):
        current=self.obs_normalizer(obs)
        memory,h_out=self.rnn(current.unsqueeze(0),h_in)
        out=self.mlp(torch.cat((current,memory.squeeze(0)),dim=-1))
        return self.deterministic_output(out),h_out,None


class _TorchHistoryGRU(_TorchGRUModel):
    def forward(self,obs):
        current=self.obs_normalizer(obs)
        memory,h_out=self.rnn(current.unsqueeze(0),self.hidden_state)
        self.hidden_state[:]=h_out
        out=self.mlp(torch.cat((current,memory.squeeze(0)),dim=-1))
        return self.deterministic_output(out)


def warm_history_from_mlp(previous,current):
    """Copy actor/normalizer/std; initialize only history influence to zero.

    Does not mutate either input or copy incompatible optimizer moments. The
    matched pilot uses a fresh optimizer for BOTH this actor and its MLP control.
    Exact functional embedding is subject only to floating-point GEMM rounding.
    """
    key='mlp.0.weight'
    if any(name.startswith('rnn.') for name in previous):
        raise ValueError('The source must be a stateless MLP')
    if key not in previous or key not in current:
        raise ValueError('Missing first MLP layer')
    n=previous[key].shape[1]
    if n!=61 or current[key].shape[0]!=previous[key].shape[0] or current[key].shape[1]<=n:
        raise ValueError('Expected a 61D MLP and a larger current-plus-history latent')
    extra=set(current)-set(previous)
    if not extra or any(not name.startswith('rnn.rnn.') for name in extra):
        raise ValueError('Unexpected target parameters beyond the recurrent module')
    result={name:value.clone() for name,value in current.items()}
    for name,value in previous.items():
        if name not in result:raise ValueError(f'Missing target parameter: {name}')
        if name==key:
            result[name].zero_();result[name][:,:n]=value
        else:
            if result[name].shape!=value.shape:raise ValueError(f'Shape mismatch: {name}')
            result[name]=value.clone()
    return result


class CurrentHistoryLinear(torch.nn.Module):
    """Keep old current-path parameters separate from newly learned memory.

    Sharing one expanded Adam parameter would give new zero columns the OLD
    scalar optimizer step. Separate parameters preserve old moments and let
    only the new memory weights start with a genuinely fresh optimizer state.
    """
    def __init__(self,initial,current_dim):
        super().__init__()
        self.current_dim=current_dim
        self.in_features=initial.in_features;self.out_features=initial.out_features
        self.weight=torch.nn.Parameter(initial.weight[:,:current_dim].detach().clone())
        self.bias=torch.nn.Parameter(initial.bias.detach().clone()) if initial.bias is not None else None
        self.history_weight=torch.nn.Parameter(torch.zeros_like(initial.weight[:,current_dim:]))

    def forward(self,latent):
        current=torch.nn.functional.linear(latent[...,:self.current_dim].contiguous(),self.weight,self.bias)
        memory=torch.nn.functional.linear(latent[...,self.current_dim:].contiguous(),self.history_weight)
        return current+memory


class SplitHistoryGRUModel(HistoryGRUModel):
    """Same current-plus-GRU function, with faithful old-optimizer transfer."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mlp[0]=CurrentHistoryLinear(self.mlp[0],self.obs_dim)


def warm_split_history_from_mlp(previous,current):
    """Preserve every existing parameter shape/value, add zero memory influence."""
    memory='mlp.0.history_weight'
    if any(name.startswith('rnn.') for name in previous) or memory in previous:
        raise ValueError('The source must be a stateless MLP')
    if previous['mlp.0.weight'].shape[1]!=61 or memory not in current:
        raise ValueError('Expected a 61D MLP and split history projection')
    extra=set(current)-set(previous)
    if any(name!=memory and not name.startswith('rnn.rnn.') for name in extra):
        raise ValueError('Unexpected new parameter in history target')
    result={name:value.clone() for name,value in current.items()}
    for name,value in previous.items():
        if name not in result or value.shape!=result[name].shape:raise ValueError(f'Incompatible parameter: {name}')
        result[name]=value.clone()
    result[memory].zero_()
    return result
