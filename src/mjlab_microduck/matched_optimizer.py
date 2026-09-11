"""Name- and shape-checked Adam transfer for an additive policy extension."""
from copy import deepcopy
import math
import torch


def named_policy_parameters(algorithm):
    return [(f'actor.{name}',value) for name,value in algorithm.actor.named_parameters()]+[
            (f'critic.{name}',value) for name,value in algorithm.critic.named_parameters()]


def sync_optimizer_learning_rate(algorithm):
    """PPO's adaptive scheduler has a scalar LR separate from Adam's groups."""
    rates={float(group['lr']) for group in algorithm.optimizer.param_groups}
    if len(rates)!=1 or not all(math.isfinite(rate) and rate>0 for rate in rates):
        raise ValueError('Cannot synchronize PPO from unequal or invalid optimizer learning rates')
    algorithm.learning_rate=next(iter(rates))
    return algorithm.learning_rate


def restore_matched_optimizer(target,saved,source_parameters):
    """Restore ALL old moments/steps; genuinely new parameters have NO state.

    Source order is obtained from the original MLP actor/critic, never guessed
    from interleaved joint indices or a tensor-size match. Only one Adam group
    is currently supported, matching the pinned RSL-RL PPO implementation.
    """
    source_parameters=list(source_parameters);target_parameters=named_policy_parameters(target)
    template=target.optimizer.state_dict()
    if len(saved['param_groups'])!=1 or len(template['param_groups'])!=1:
        raise ValueError('Matched transfer supports one optimizer group')
    source_ids=saved['param_groups'][0]['params'];target_ids=template['param_groups'][0]['params']
    if len(source_ids)!=len(source_parameters) or len(target_ids)!=len(target_parameters):
        raise ValueError('Optimizer order does not match model parameters')
    source_by_name=dict(source_parameters);target_by_name=dict(target_parameters)
    if len(source_by_name)!=len(source_parameters) or len(target_by_name)!=len(target_parameters):
        raise ValueError('Duplicate model parameter names')
    if not set(source_by_name)<=set(target_by_name):raise ValueError('Old parameter missing from target')
    added=set(target_by_name)-set(source_by_name)
    if any(name!='actor.mlp.0.history_weight' and not name.startswith('actor.rnn.rnn.') for name in added):
        raise ValueError('Unexpected new parameter outside the memory path')
    target_id_by_name={name:pid for (name,_),pid in zip(target_parameters,target_ids)}
    restored={}
    for (name,parameter),pid in zip(source_parameters,source_ids):
        if parameter.shape!=target_by_name[name].shape:raise ValueError(f'Old parameter shape changed: {name}')
        item=saved['state'].get(pid)
        if item is None:continue
        for key,value in item.items():
            if torch.is_tensor(value) and key!='step' and value.shape!=parameter.shape:
                raise ValueError(f'Optimizer shape mismatch for {name}/{key}')
        restored[target_id_by_name[name]]=deepcopy(item)
    if not set(saved['state'])<=set(source_ids):raise ValueError('Unknown saved optimizer parameter id')
    group=deepcopy(saved['param_groups'][0]);group['params']=target_ids
    target.optimizer.load_state_dict(dict(state=restored,param_groups=[group]))
    lr=sync_optimizer_learning_rate(target)
    return dict(restored_parameter_states=len(restored),new_parameters=sorted(added),learning_rate=lr,
                new_parameters_have_fresh_state=True)
