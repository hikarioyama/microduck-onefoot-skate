"""Experimental waist-v10 task with a warm-start-preserving GRU actor.

Only the actor architecture changes. The original MLP current-input path is
retained; new history inputs start with zero influence. The critic, algorithm,
61D observations, 14D actions, rewards, physics, DR and gates are unchanged.
A recurrent inference runtime must carry/reset the additional hidden state.
"""
from .microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg

TASK='Mjlab-OneFoot-Waist-History-Curriculum-Flat-MicroDuck-Rollers'


def make_history_onefoot_env_cfg(play=False,stage='balance-050'):
    return make_waist_onefoot_env_cfg(play=play,stage=stage)


def make_history_onefoot_rl_cfg():
    cfg=make_waist_onefoot_rl_cfg();cfg.run_name='waist-history-v12-pilot'
    cfg.actor.class_name='mjlab_microduck.history_policy:HistoryGRUModel'
    cfg.actor.rnn_type='gru';cfg.actor.rnn_hidden_dim=128;cfg.actor.rnn_num_layers=1
    return cfg


MATCHED_TASK='Mjlab-OneFoot-Waist-History-Matched-Curriculum-Flat-MicroDuck-Rollers'


def make_matched_history_onefoot_env_cfg(play=False,stage='balance-050'):
    return make_waist_onefoot_env_cfg(play=play,stage=stage)


def make_matched_history_onefoot_rl_cfg():
    cfg=make_history_onefoot_rl_cfg();cfg.run_name='waist-history-matched-v12b-pilot'
    cfg.actor.class_name='mjlab_microduck.history_policy:SplitHistoryGRUModel'
    return cfg
