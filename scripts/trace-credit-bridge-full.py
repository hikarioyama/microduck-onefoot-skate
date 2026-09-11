"""Why does the `full` glide stop? Record the physics of every failed episode.

Read-only diagnostic. Loads one saved credit-bridge checkpoint, rolls out a small
batched `full` stage (HOME spawn, exact zero velocity) and, for each environment,
writes the per-step glide quantities plus the termination reason. The point is to
separate two very different failure families the aggregate counters conflate:

  * never reached left-only single support  (curriculum_onefoot_stalled)
  * reached it, then lost it                (fell / replanted / slowed down)

For the credible-glide window it also fits the forward coast-down, so we can tell
whether a 2 s glide is even reachable for a free-rolling skate that may not push.
"""
import argparse
import importlib.util
import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('credit_runner', ROOT/'local/train-credit-bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--episodes', type=int, default=64)
p.add_argument('--seed', type=int, default=70601)
p.add_argument('--stage', default='full')
p.add_argument('--output', type=Path, required=True)

a = p.parse_args()
assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
configure_torch_backends(allow_tf32=False)
torch.set_num_threads(4)

TRACK = ('phase', 'speed', 'com_speed', 'lateral', 'upright', 'support_yaw',
         'cross_track', 'clearance', 'com_offset', 'capture_error',
         'left_force', 'right_force', 'left_contact', 'right_contact',
         'single_support', 'dwell', 'best_dwell')


@torch.no_grad()
def trace():
    env, rl = module.environment(a.episodes, a.seed, a.stage)
    runner = load_runner_cls(module.TASK)(env, deepcopy(asdict(rl)), device='cuda:0')
    safe_runner_load(runner, a.checkpoint, strict=True, map_location='cuda:0')
    raw = env.unwrapped
    obs = env.get_observations()
    policy = evaluation_policy(runner.alg.actor, obs, rl.actor, 14)
    # Wheel JOINTS by name. cmd.wheel_ids are body ids, so they must never index
    # joint_vel; the support pair comes first, matching wheel_ids.
    robot0 = raw.scene['robot']
    wheel_joint_ids = torch.tensor(
        [robot0.find_joints(n)[0][0]
         for n in ('passive_LF_wheel', 'passive_LR_wheel',
                   'passive_RF_wheel', 'passive_RR_wheel')],
        dtype=torch.long, device='cuda:0')
    n = a.episodes
    alive = torch.ones(n, dtype=torch.bool, device='cuda:0')
    frames = []
    steps = []
    # Fore/aft capture is not exported by the env, so it is reconstructed here with
    # the env's own one-step finite difference, exactly as it does for the lateral axis.
    dt = env.unwrapped.step_dt
    prev = {'com': None, 'support': None}
    try:
        for step in range(raw.max_episode_length + 1):
            action = policy(obs)
            obs, reward, done, extras = env.step(action)
            cmd = raw.command_manager.get_term('twist')
            v = cmd.curriculum_values
            r = cmd.bridge_reward_values
            robot = raw.scene['robot']
            data = robot.data
            com = data.data.subtree_com[:, data.indexing.root_body_id]
            wheels = data.body_link_pos_w[:, cmd.wheel_ids_tensor]
            support = wheels[:, :2].mean(1)
            if prev['com'] is None:
                com_v = torch.zeros_like(com)
                support_v = torch.zeros_like(support)
            else:
                com_v = (com - prev['com']) / dt
                support_v = (support - prev['support']) / dt
            prev['com'] = com.clone()
            prev['support'] = support.clone()
            height = (com[:, 2] - wheels[:, :2, 2].mean(1)).clamp(.05, .30)
            tau = torch.sqrt(height / 9.81)
            lateral_capture = com[:, 1] - support[:, 1] + (com_v[:, 1] - support_v[:, 1]) * tau
            fore_aft_capture = com[:, 0] - support[:, 0] + (com_v[:, 0] - support_v[:, 0]) * tau
            skate = robot.data.body_link_lin_vel_w[:, cmd.wheel_ids[:2], 0].mean(1)
            left = raw.scene['onefoot_left'].data.force[..., 2].reshape(n, -1).abs()
            right = raw.scene['onefoot_right'].data.force[..., 2].reshape(n, -1).abs().sum(1)
            terms = {name: raw.termination_manager.get_term(name).cpu().numpy().astype(bool)
                     for name in raw.termination_manager.active_terms}
            row = {k: v[k].cpu().numpy().copy() for k in TRACK}
            row['skate_speed'] = skate.cpu().numpy()
            row['left_min_force'] = left.min(1).values.cpu().numpy()
            row['right_raw'] = right.cpu().numpy()
            row['credible'] = r['credible'].cpu().numpy()
            row['lateral_capture'] = lateral_capture.cpu().numpy()
            row['fore_aft_capture'] = fore_aft_capture.cpu().numpy()
            row['com_x'] = com[:, 0].cpu().numpy()
            row['support_x'] = support[:, 0].cpu().numpy()
            row['com_y'] = com[:, 1].cpu().numpy()
            row['support_y'] = support[:, 1].cpu().numpy()
            # Extra candidates for the unexplained glide drag: body pitch/roll,
            # yaw rate and the support-skate wheels' own spin.
            g = robot.data.projected_gravity_b
            av = data.body_link_ang_vel_w[:, data.indexing.root_body_id]
            # `cmd.wheel_ids` holds BODY ids (find_bodies) for body_link_pos_w above.
            # Indexing joint_vel with them silently reads four unrelated joints and
            # produces a wheel-spin series that looks like a 50% skid. Resolve the
            # wheel JOINTS by name instead.
            spin = data.joint_vel[:, wheel_joint_ids].mean(1)
            row['proj_g_x'] = g[:, 0].cpu().numpy()
            row['proj_g_y'] = g[:, 1].cpu().numpy()
            row['ang_vel_x'] = av[:, 0].cpu().numpy()
            row['ang_vel_y'] = av[:, 1].cpu().numpy()
            row['ang_vel_z'] = av[:, 2].cpu().numpy()
            row['wheel_spin'] = spin.cpu().numpy()
            row['support_yaw_raw'] = torch.atan2(wheels[:, 0, 1]-wheels[:, 1, 1],
                                                 wheels[:, 0, 0]-wheels[:, 1, 0]).cpu().numpy()
            row['alive'] = alive.cpu().numpy()
            frames.append(row)
            steps.append(dict(t=round((step + 1) * raw.step_dt, 4),
                              done=done.cpu().numpy().copy(),
                              alive=alive.cpu().numpy().copy(),
                              terms={k: vv.copy() for k, vv in terms.items()}))
            alive &= ~done.bool()
            if not bool(alive.any()):
                break
        report = summarise(frames, steps)
    finally:
        env.close()
    report.update(checkpoint=str(a.checkpoint), episodes=a.episodes, seed=a.seed, stage=a.stage,
                  steps_executed=len(steps))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('TRACE_DONE', json.dumps({k: report[k] for k in
          ('episodes', 'reached_single_support', 'fell', 'replanted', 'never_transferred')}))
    return report


def summarise(frames, steps):
    n = frames[0]['alive'].shape[0]
    credible_streak = np.zeros(n, dtype=int)
    best_credible = np.zeros(n, dtype=float)
    entry_speed = np.full(n, np.nan)
    reached = np.zeros(n, dtype=bool)
    never = np.zeros(n, dtype=bool)
    fell = np.zeros(n, dtype=bool)
    replanted = np.zeros(n, dtype=bool)
    exit_reason = [''] * n
    coast = []
    per_env_credible_speed = [[] for _ in range(n)]
    per_env_credible_t = [[] for _ in range(n)]
    credible_detail = [[] for _ in range(n)]
    lateral_abs = np.zeros(n)
    series_keys = ('speed', 'com_speed', 'phase', 'clearance', 'upright', 'lateral',
                   'com_offset', 'capture_error', 'left_force', 'right_force')
    timeseries = []
    # Coarse per-environment record (every .1 s) so episodes that SURVIVE the whole
    # 6 s horizon without ever meeting the credible predicate can still be read:
    # the question there is no longer balance but what the speed did.
    coarse = [[] for _ in range(n)]
    for step_index, (f, s) in enumerate(zip(frames, steps)):
        alive = f['alive']
        timeseries.append(dict(t=float(s['t']), alive=int(alive.sum()), **{
            k: (float(f[k][alive].mean()) if bool(alive.any()) else None) for k in series_keys}))
        if step_index % 5 == 4:
            for i in range(n):
                if alive[i]:
                    coarse[i].append(dict(t=float(s['t']), speed=float(f['speed'][i]),
                                          com_speed=float(f['com_speed'][i]),
                                          upright=float(f['upright'][i]),
                                          clearance=float(f['clearance'][i]),
                                          left_force=float(f['left_force'][i]),
                                          right_force=float(f['right_force'][i]),
                                          single_support=float(f['single_support'][i]),
                                          credible=float(f['credible'][i])))
        cred = f['credible'] > .5
        ss = f['single_support'] > .5
        credible_streak = np.where(cred, credible_streak + 1, 0)
        best_credible = np.maximum(best_credible, credible_streak * .02)
        newly = ss & ~reached
        entry_speed[newly] = f['com_speed'][newly]
        reached |= ss
        lateral_abs = np.maximum(lateral_abs, np.where(alive, np.abs(f['lateral']), 0.))
        for i in np.where(cred & alive)[0]:
            per_env_credible_speed[i].append(float(f['com_speed'][i]))
            per_env_credible_t[i].append(float(s['t']))
            credible_detail[i].append(dict(t=float(s['t']), com_speed=float(f['com_speed'][i]),
                                          speed=float(f['speed'][i]), upright=float(f['upright'][i]),
                                          com_offset=float(f['com_offset'][i]),
                                          support_yaw=float(f['support_yaw'][i]),
                                          lateral=float(f['lateral'][i]),
                                          lateral_capture=float(f['lateral_capture'][i]),
                                          fore_aft_capture=float(f['fore_aft_capture'][i]),
                                          proj_g_x=float(f['proj_g_x'][i]),
                                          proj_g_y=float(f['proj_g_y'][i]),
                                          ang_vel_x=float(f['ang_vel_x'][i]),
                                          ang_vel_y=float(f['ang_vel_y'][i]),
                                          ang_vel_z=float(f['ang_vel_z'][i]),
                                          wheel_spin=float(f['wheel_spin'][i]),
                                          support_yaw_raw=float(f['support_yaw_raw'][i]),
                                          clearance=float(f['clearance'][i])))
        done = s['done'] & alive
        for i in np.where(done)[0]:
            names = [k for k, v in s['terms'].items() if v[i]]
            exit_reason[i] = ','.join(names) or 'unflagged'
            never[i] = 'stalled_transfer' in names
            replanted[i] = 'replanted_swing' in names
            fell[i] = bool({'fell_over', 'body_ground'} & set(names))
    for i in range(n):
        sp = per_env_credible_speed[i]
        tt = per_env_credible_t[i]
        if len(sp) >= 5:
            slope = float(np.polyfit(tt, sp, 1)[0])
            coast.append(dict(env=i, entry_speed=float(sp[0]), exit_speed=float(sp[-1]),
                              duration_s=float(tt[-1] - tt[0] + .02), accel=slope,
                              best_credible_s=float(best_credible[i]),
                              exit_reason=exit_reason[i], lateral_abs=float(lateral_abs[i])))
    entries = entry_speed[~np.isnan(entry_speed)]
    report = dict(
        reached_single_support=int(reached.sum()), never_transferred=int(never.sum()),
        fell=int(fell.sum()), replanted=int(replanted.sum()),
        exit_reason_counts={r: exit_reason.count(r) for r in sorted(set(exit_reason))},
        mean_best_credible_s=float(best_credible.mean()), max_best_credible_s=float(best_credible.max()),
        credible_050=int((best_credible >= .5 - 1e-5).sum()),
        credible_100=int((best_credible >= 1. - 1e-5).sum()),
        credible_200=int((best_credible >= 2. - 1e-5).sum()),
        entry_speed_com=dict(count=int(entries.size), mean=float(entries.mean()) if entries.size else None,
                             min=float(entries.min()) if entries.size else None,
                             max=float(entries.max()) if entries.size else None,
                             p50=float(np.percentile(entries, 50)) if entries.size else None,
                             p90=float(np.percentile(entries, 90)) if entries.size else None),
        coast_down=coast,
        credible_series=[dict(env=i, series=credible_detail[i]) for i in range(n) if credible_detail[i]],
        per_env_coarse=[dict(env=i, series=coarse[i], exit_reason=exit_reason[i],
                             best_credible_s=float(best_credible[i])) for i in range(n) if coarse[i]],
        timeseries=timeseries,
        coast_down_mean_accel=float(np.mean([c['accel'] for c in coast])) if coast else None,
        coast_down_worst_accel=float(np.min([c['accel'] for c in coast])) if coast else None,
        implied_max_glide_s_from_coast=coast_ceiling(coast),
    )
    return report


def coast_ceiling(coast):
    """Longest glide implied by coasting from the entry speed down to the 0.10 m/s floor."""
    best = []
    for c in coast:
        if c['accel'] < 0 and c['entry_speed'] > .10:
            best.append((c['entry_speed'] - .10) / -c['accel'])
    if not best:
        return None
    return dict(count=len(best), mean=float(np.mean(best)), max=float(np.max(best)))


if __name__ == '__main__':
    trace()
