"""Approximately embed a learned MLP in the official RSL-RL GRU model.

The small near-linear tanh embedding preserves useful behavior approximately,
NOT bit-exactly. Always check numerical and rollout parity before training.
"""
import torch


def warm_gru_from_mlp(previous, current, scale=.02):
    assert 0<scale<=.1
    key='mlp.0.weight';rnn_key='rnn.rnn.weight_ih_l0'
    assert rnn_key in current and rnn_key not in previous
    n=previous[key].shape[1];h=current[key].shape[1]
    assert n==61 and h>=n
    assert current[rnn_key].shape==(3*h,n)
    assert not any('_l1' in k for k in current if k.startswith('rnn.'))
    result={k:v.clone() for k,v in current.items()}
    for name,value in previous.items():
        if name.startswith('distribution.'):
            continue  # fresh, explicitly configured exploration
        if name==key:
            result[name].zero_();result[name][:,:n]=value/scale
        else:
            assert name in result and result[name].shape==value.shape,name
            result[name]=value.clone()
    for name in result:
        if name.startswith('rnn.rnn.'):
            result[name].zero_()
    # PyTorch GRU gate order is reset, update, new.
    result[rnn_key][2*h:2*h+n,:]=torch.eye(n,device=result[rnn_key].device)*scale
    result['rnn.rnn.bias_ih_l0'][h:2*h]=-8.
    if h>n:
        # Extra channels already contain short/long history, but their MLP
        # columns are zero initially, so they do not change the initial action.
        result[rnn_key][2*h+n:3*h,:]=current[rnn_key][2*h+n:3*h,:]
        result['rnn.rnn.bias_ih_l0'][h+n:2*h]=torch.linspace(
            0.,4.6,h-n,device=result[rnn_key].device)
    return result
