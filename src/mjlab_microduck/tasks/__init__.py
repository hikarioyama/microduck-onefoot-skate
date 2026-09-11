from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

# Custom one-foot roller glide: keep all upstream tasks unchanged.
from .microduck_onefoot_env_cfg import (
    TASK as _ONEFOOT_TASK, ASSISTED_TASK as _ONEFOOT_ASSISTED_TASK,
    make_onefoot_env_cfg, make_onefoot_rl_cfg,
)
for _task, _assisted in ((_ONEFOOT_TASK, False), (_ONEFOOT_ASSISTED_TASK, True)):
    register_mjlab_task(
        task_id=_task,
        env_cfg=make_onefoot_env_cfg(assisted=_assisted),
        play_env_cfg=make_onefoot_env_cfg(play=True, assisted=_assisted),
        rl_cfg=make_onefoot_rl_cfg(assisted=_assisted),
        runner_cls=MicroduckOnPolicyRunner,
    )

from .microduck_onefoot_dynamic_env_cfg import (
    DYNAMIC_TASK as _DYNAMIC_ONEFOOT_TASK,
    DYNAMIC_ASSISTED_TASK as _DYNAMIC_ONEFOOT_ASSISTED_TASK,
    make_dynamic_onefoot_env_cfg, make_dynamic_onefoot_rl_cfg,
)
for _task,_assisted in ((_DYNAMIC_ONEFOOT_TASK,False),(_DYNAMIC_ONEFOOT_ASSISTED_TASK,True)):
    register_mjlab_task(task_id=_task,
        env_cfg=make_dynamic_onefoot_env_cfg(assisted=_assisted),
        play_env_cfg=make_dynamic_onefoot_env_cfg(play=True,assisted=_assisted),
        rl_cfg=make_dynamic_onefoot_rl_cfg(assisted=_assisted),
        runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_aligned_env_cfg import (
    ALIGNED_TASK as _ALIGNED_ONEFOOT_TASK,
    ALIGNED_ASSISTED_TASK as _ALIGNED_ONEFOOT_ASSISTED_TASK,
    make_aligned_onefoot_env_cfg,make_aligned_onefoot_rl_cfg,
)
for _task,_assisted in ((_ALIGNED_ONEFOOT_TASK,False),(_ALIGNED_ONEFOOT_ASSISTED_TASK,True)):
    register_mjlab_task(task_id=_task,
        env_cfg=make_aligned_onefoot_env_cfg(assisted=_assisted),
        play_env_cfg=make_aligned_onefoot_env_cfg(play=True,assisted=_assisted),
        rl_cfg=make_aligned_onefoot_rl_cfg(assisted=_assisted),
        runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_curriculum_env_cfg import (
    TASK as _CURRICULUM_ONEFOOT_TASK,
    make_curriculum_onefoot_env_cfg, make_curriculum_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_CURRICULUM_ONEFOOT_TASK,
    env_cfg=make_curriculum_onefoot_env_cfg(),
    play_env_cfg=make_curriculum_onefoot_env_cfg(play=True),
    rl_cfg=make_curriculum_onefoot_rl_cfg(), runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_continuous_env_cfg import (
    TASK as _CONTINUOUS_ONEFOOT_TASK,
    make_continuous_onefoot_env_cfg, make_continuous_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_CONTINUOUS_ONEFOOT_TASK,
    env_cfg=make_continuous_onefoot_env_cfg(),
    play_env_cfg=make_continuous_onefoot_env_cfg(play=True),
    rl_cfg=make_continuous_onefoot_rl_cfg(), runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_centroid_env_cfg import (
    TASK as _CENTROID_ONEFOOT_TASK,make_centroid_onefoot_env_cfg,make_centroid_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_CENTROID_ONEFOOT_TASK,
    env_cfg=make_centroid_onefoot_env_cfg(),
    play_env_cfg=make_centroid_onefoot_env_cfg(play=True),
    rl_cfg=make_centroid_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_continuous_env_cfg import STAGE_TASKS as _CONTINUOUS_STAGE_TASKS
for _stage,_task_id in _CONTINUOUS_STAGE_TASKS.items():
    register_mjlab_task(task_id=_task_id,
        env_cfg=make_continuous_onefoot_env_cfg(stage=_stage),
        play_env_cfg=make_continuous_onefoot_env_cfg(play=True,stage=_stage),
        rl_cfg=make_continuous_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_memory_env_cfg import (
    TASK as _MEMORY_ONEFOOT_TASK,make_memory_onefoot_env_cfg,make_memory_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_MEMORY_ONEFOOT_TASK,
    env_cfg=make_memory_onefoot_env_cfg(),play_env_cfg=make_memory_onefoot_env_cfg(play=True),
    rl_cfg=make_memory_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_lean_env_cfg import (
    TASK as _LEAN_ONEFOOT_TASK,make_lean_onefoot_env_cfg,make_lean_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_LEAN_ONEFOOT_TASK,
    env_cfg=make_lean_onefoot_env_cfg(),play_env_cfg=make_lean_onefoot_env_cfg(play=True),
    rl_cfg=make_lean_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_waist_env_cfg import (
    TASK as _WAIST_ONEFOOT_TASK,make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_WAIST_ONEFOOT_TASK,
    env_cfg=make_waist_onefoot_env_cfg(),play_env_cfg=make_waist_onefoot_env_cfg(play=True),
    rl_cfg=make_waist_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_knee_env_cfg import (
    TASK as _KNEE_ONEFOOT_TASK,make_knee_onefoot_env_cfg,make_knee_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_KNEE_ONEFOOT_TASK,
    env_cfg=make_knee_onefoot_env_cfg(),play_env_cfg=make_knee_onefoot_env_cfg(play=True),
    rl_cfg=make_knee_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_history_env_cfg import (
    TASK as _HISTORY_ONEFOOT_TASK,make_history_onefoot_env_cfg,make_history_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_HISTORY_ONEFOOT_TASK,
    env_cfg=make_history_onefoot_env_cfg(),play_env_cfg=make_history_onefoot_env_cfg(play=True),
    rl_cfg=make_history_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_history_env_cfg import (
    MATCHED_TASK as _MATCHED_HISTORY_ONEFOOT_TASK,
    make_matched_history_onefoot_env_cfg,make_matched_history_onefoot_rl_cfg,
)
register_mjlab_task(task_id=_MATCHED_HISTORY_ONEFOOT_TASK,
    env_cfg=make_matched_history_onefoot_env_cfg(),play_env_cfg=make_matched_history_onefoot_env_cfg(play=True),
    rl_cfg=make_matched_history_onefoot_rl_cfg(),runner_cls=MicroduckOnPolicyRunner)

from .microduck_onefoot_credit_bridge_env_cfg import (
    TASK as _CREDIT_BRIDGE_TASK, make_credit_bridge_env_cfg, make_credit_bridge_rl_cfg,
)
register_mjlab_task(task_id=_CREDIT_BRIDGE_TASK,
    env_cfg=make_credit_bridge_env_cfg(), play_env_cfg=make_credit_bridge_env_cfg(play=True),
    rl_cfg=make_credit_bridge_rl_cfg(), runner_cls=MicroduckOnPolicyRunner)
