"""Experimental official GRU actor: same per-step sensors, physics and skill gates.

61D observations and 14D actions remain unchanged. Export uses standard RSL-RL
obs/h_in -> actions/h_out; a hardware runtime would have to carry/reset h_in.
No hardware compatibility or skill success is implied by this configuration.
"""
from .microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg,make_continuous_onefoot_rl_cfg

TASK='Mjlab-OneFoot-Memory-Curriculum-Flat-MicroDuck-Rollers'


def make_memory_onefoot_env_cfg(play=False,stage='balance-050'):
    return make_continuous_onefoot_env_cfg(play=play,stage=stage)


def make_memory_onefoot_rl_cfg():
    cfg=make_continuous_onefoot_rl_cfg()
    cfg.run_name='memory-curriculum-v8'
    cfg.actor.class_name='RNNModel';cfg.actor.rnn_type='gru'
    cfg.actor.rnn_hidden_dim=128;cfg.actor.rnn_num_layers=1
    cfg.algorithm.learning_rate=1e-4
    return cfg
