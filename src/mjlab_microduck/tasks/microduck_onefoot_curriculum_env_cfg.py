"""v5 reverse curriculum with externally evaluated, non-skippable stage gates.

Same 61D actor, HOME-relative 14D actions, real BAM/noise/delay/DR. The custom
command contract is [speed, lift phase, heading correction], NOT hardware velocity.
Spawn assistance is explicitly recorded and exactly zero in the final stage.
"""
from copy import deepcopy
from dataclasses import dataclass
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_dynamic_env_cfg import make_dynamic_onefoot_env_cfg, make_dynamic_onefoot_rl_cfg
from .microduck_onefoot_curriculum_poses import POSES

TASK = 'Mjlab-OneFoot-Curriculum-Flat-MicroDuck-Rollers'


@dataclass(frozen=True)
class Stage:
    name: str
    group: int
    spawn: str
    goal_s: float
    speed_range: tuple[float,float]
    acceleration_s: float
    lift_s: float
    phase_start: float
    episode_s: float


STAGES = (
    Stage('balance-050',1,'balance',.5,(.28,.32),0.,1.,1.,1.5),
    Stage('balance-100',1,'balance',1.,(.28,.32),0.,1.,1.,2.0),
    Stage('balance-200',1,'balance',2.,(.25,.35),0.,1.,1.,3.0),
    Stage('unload',2,'unload',2.,(.25,.35),.1,.8,.62,4.0),
    Stage('transfer-near',3,'transfer_near',2.,(.25,.35),.2,1.,.2,4.5),
    Stage('transfer-mid',3,'transfer_mid',2.,(.25,.35),.2,1.,.2,4.5),
    Stage('transfer',3,'transfer',2.,(.25,.35),.3,1.,0.,5.0),
    Stage('launch-020',4,'transfer',2.,(.18,.22),2.0,1.,0.,6.5),
    Stage('launch-010',4,'transfer',2.,(.08,.12),2.5,1.,0.,7.0),
    Stage('self-launch',4,'transfer',2.,(0.,0.),2.5,1.,0.,8.0),
)
STAGE_BY_NAME = {stage.name:stage for stage in STAGES}
GATE_EPISODES = 256
GATE_SUCCESS_RATE = .80
GATE_BATCHES = 2


def make_curriculum_onefoot_env_cfg(play=False, stage='balance-050'):
    stage = STAGE_BY_NAME[stage]
    cfg = make_dynamic_onefoot_env_cfg(play=play,assisted=True)
    cfg.commands['twist'] = mdp.CurriculumOneFootCommandCfg(
        resampling_time_range=(1000,1000),goal_s=stage.goal_s,
        acceleration_s=stage.acceleration_s,lift_s=stage.lift_s)
    cfg.episode_length_s = stage.episode_s
    cfg.events.pop('onefoot_seed')
    cfg.events['curriculum_spawn'] = EventTermCfg(func=mdp.reset_curriculum_onefoot,
        mode='reset',params=dict(**deepcopy(POSES[stage.spawn]),speed_range=stage.speed_range,
                                 phase_start=stage.phase_start))
    # This preceding base reset is overwritten by curriculum_spawn. Still make
    # the final-stage no-injection guarantee explicit in both reset terms.
    cfg.events['reset_base'].params['velocity_range'] = {
        key:(0.,0.) for key in ('x','y','z','roll','pitch','yaw')}
    for sensor in cfg.scene.sensors:
        if sensor.name in ('onefoot_left','onefoot_right'):
            sensor.fields = ('found','force')
    keep = {'self_collisions','joint_torques_l2','action_over_limit'}
    cfg.rewards = {k:v for k,v in cfg.rewards.items() if k in keep}
    for component,weight in [('hold',12.),('continuous',6.),('transfer',5.),
                             ('accelerate',3.),('alignment',-.3),('steering',.5)]:
        cfg.rewards['curriculum_'+component] = RewardTermCfg(
            func=mdp.curriculum_onefoot_signal,weight=weight,params={'component':component})
    cfg.rewards['failure'] = RewardTermCfg(func=mdp.curriculum_onefoot_failure,weight=-100.)
    # Start with very small smoothness tax; no head HOME or angular-speed blocker.
    cfg.rewards['dynamic_action_rate'] = RewardTermCfg(func=mdp.dynamic_onefoot_motion_cost,
        weight=-.003,params={'kind':'action_rate'})
    cfg.terminations['stalled_transfer'] = TerminationTermCfg(
        func=mdp.curriculum_onefoot_stalled,time_out=False)
    last = {'best_dwell','success','success_050','success_100','success_200','distance','injected_speed'}
    names = ('speed','com_speed','lateral','clearance','upright','cross_track','heading_error',
             'support_yaw','capture_error','com_offset','single_support','left_contact','right_contact',
             'left_load','left_force','right_force','dwell','best_dwell','success','success_050',
             'success_100','success_200','distance','injected_speed','head_speed','phase')
    cfg.metrics = {'onefoot/'+name:MetricsTermCfg(func=mdp.curriculum_onefoot_signal,
        params={'component':name},reduce='last' if name in last else 'mean') for name in names}
    return cfg


def make_curriculum_onefoot_rl_cfg():
    cfg = make_dynamic_onefoot_rl_cfg(assisted=True)
    cfg.run_name = 'curriculum-v5'
    cfg.actor.distribution_cfg['init_std'] = .12
    cfg.algorithm.entropy_coef = .005
    cfg.algorithm.learning_rate = 3e-4
    cfg.save_interval = 100
    return cfg


def gate_passes(reports, goal_s):
    """Independent evaluations, never training reward or assisted/full averages."""
    return (len(reports) >= GATE_BATCHES and all(
        row['episodes'] >= GATE_EPISODES and row['goal_s'] == goal_s and
        row['success_rate'] >= GATE_SUCCESS_RATE and row['nan_episodes'] == 0
        for row in reports[-GATE_BATCHES:]))
