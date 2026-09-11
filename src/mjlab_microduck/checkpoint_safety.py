"""Safe checkpoint/evaluation transitions across PPO inference and gradient phases."""
from dataclasses import asdict
import torch


def safe_runner_load(runner, checkpoint, **kwargs):
    """Materialize inference-created buffers before ordinary checkpoint loading.

    Keep torch.load/optimizer restoration OUTSIDE InferenceMode so restored Adam
    moments remain normal tensors that can be updated during the next PPO step.
    """
    with torch.inference_mode(False),torch.no_grad():
        for model in (runner.alg.actor,runner.alg.critic):
            for name,buffer in list(model.named_buffers()):
                if buffer.is_inference():
                    parent,_,leaf=name.rpartition('.')
                    owner=model.get_submodule(parent) if parent else model
                    owner._buffers[leaf]=buffer.clone()
        result=runner.load(str(checkpoint),**kwargs)
        load_cfg=kwargs.get('load_cfg')
        if ((load_cfg is None or load_cfg.get('optimizer',False)) and
                hasattr(runner.alg,'optimizer') and hasattr(runner.alg,'learning_rate')):
            # RSL restores Adam's lr but not PPO's separate adaptive-lr scalar.
            # Without this synchronization the next update overwrites the saved
            # rate with the constructor default. Actor-only evaluation is exempt.
            from mjlab_microduck.matched_optimizer import sync_optimizer_learning_rate
            sync_optimizer_learning_rate(runner.alg)
        return result


def evaluation_policy(actor, observations, model_cfg, output_dim):
    """Construct from state_dict, not deepcopy of cached non-leaf distributions.

    The original normalization, recurrent state and train/eval mode are untouched.
    Restore RNG after construction so evaluation noise is not changed by random
    initialization of the temporary network.
    """
    kwargs=asdict(model_cfg)
    kwargs.pop('class_name',None)
    for name in ('cnn_cfg','distribution_cfg'):
        if kwargs.get(name) is None:kwargs.pop(name,None)
    if kwargs.get('rnn_type') is None:
        for name in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):kwargs.pop(name,None)
    device=next(actor.parameters()).device
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        copy=type(actor)(obs=observations,obs_groups={'actor':list(actor.obs_groups)},
                        obs_set='actor',output_dim=output_dim,**kwargs).to(device)
    copy.load_state_dict(actor.state_dict(),strict=True)
    copy.reset();copy.eval()
    if getattr(copy,'is_recurrent',False):copy.rnn.rnn.flatten_parameters()
    return copy


def verified_recovery(state, stages, gate_passes, *, prefer_latest=False):
    """Resume only after verifiable, consecutive earlier gates; never skip one."""
    if state['status'] not in ('error','interrupted'):
        raise ValueError('Recovery requires a stopped or failed controller')
    passed=state.get('passed_stages',[])
    for index,item in enumerate(passed):
        stage=stages[index]
        reports=item['reports']
        if (item['stage']!=stage.name or not gate_passes(reports,stage.goal_s) or
            len({r['seed'] for r in reports[-2:]})!=2 or
            any(r['stage']!=stage.name or r['checkpoint']!=item['checkpoint'] for r in reports[-2:])):
            raise ValueError('Prior stage gate is incomplete or inconsistent')
    current=next(i for i,s in enumerate(stages) if s.name==state['stage'])
    if current>len(passed):raise ValueError('Cannot skip an unpassed stage')
    index=max(current,len(passed))
    if index>=len(stages):raise ValueError('All gates passed; perform final validation instead')
    if prefer_latest:
        if state['status']!='interrupted' or not state.get('checkpoint'):
            raise ValueError('Latest-weight continuation requires an intentional saved pause')
        checkpoint=state['checkpoint']
    else:
        checkpoint=state.get('best_checkpoint') or state['checkpoint']
    return stages[index].name,checkpoint,int(state['iteration'])+1
