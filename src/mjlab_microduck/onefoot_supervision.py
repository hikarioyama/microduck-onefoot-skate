"""Evidence-based curriculum supervision; no hardware/process control on import.

Exploration is allowed a measured recovery window. This module never changes
skill criteria, rewards, physics, observations, actions, or checkpoint contents.
"""
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import statistics


@dataclass(frozen=True)
class ExplorationBudget:
    block_evaluations: int = 6
    maximum_evaluations: int = 18
    minimum_episodes: int = 256
    success_improvement: float = .04
    duration_improvement_s: float = .03
    success_regression_tolerance: float = .03
    duration_regression_tolerance_s: float = .01

    def __post_init__(self):
        if self.block_evaluations < 6 or self.block_evaluations % 2:
            raise ValueError('Use an even evidence block of at least six evaluations')
        if self.maximum_evaluations < self.block_evaluations or self.maximum_evaluations % self.block_evaluations:
            raise ValueError('Maximum budget must contain complete evidence blocks')
        if self.minimum_episodes < 256:
            raise ValueError('Supervision cannot weaken independent evaluation size')


def checkpoint_iteration(path):
    match = re.fullmatch(r'model_(\d+)\.pt', Path(path).name)
    if match is None:
        raise ValueError(f'Invalid checkpoint name: {path}')
    return int(match[1])


def _primary_reports(reports, stage, minimum_episodes):
    by_checkpoint = {}
    for row in reports:
        if row['stage'] != stage:
            continue
        p = float(row['success_rate']); duration = float(row['mean_best_glide_s'])
        if (not math.isfinite(p) or not 0 <= p <= 1 or
                not math.isfinite(duration) or duration < 0 or
                row['episodes'] < minimum_episodes):
            raise ValueError('Exploration decisions require valid independent evaluations')
        checkpoint_iteration(row['checkpoint'])
        # A confirmation of the SAME checkpoint is not another learning chunk.
        by_checkpoint.setdefault(row['checkpoint'], row)
    return sorted(by_checkpoint.values(), key=lambda r: checkpoint_iteration(r['checkpoint']))


def _block_trend(rows, budget):
    mid = len(rows) // 2
    early, late = rows[:mid], rows[mid:]
    def mean(part, key):
        return statistics.mean(float(row[key]) for row in part)
    p0, p1 = mean(early, 'success_rate'), mean(late, 'success_rate')
    d0, d1 = mean(early, 'mean_best_glide_s'), mean(late, 'mean_best_glide_s')
    dp, dd = p1 - p0, d1 - d0
    improving = ((dp >= budget.success_improvement - 1e-12 and
                  dd >= -budget.duration_regression_tolerance_s - 1e-12) or
                 (dd >= budget.duration_improvement_s - 1e-12 and
                  dp >= -budget.success_regression_tolerance - 1e-12))
    return dict(early_success_rate=p0, late_success_rate=p1,
                early_mean_glide_s=d0, late_mean_glide_s=d1,
                success_delta=dp, glide_delta_s=dd, improving=improving)


def exploration_plan(reports, stage, adaptations=(), budget=ExplorationBudget()):
    """Grant 6 -> 12 -> 18 evaluated chunks only for measured recovery.

    The first three versus last three DISTINCT checkpoint evaluations form a
    recovery block. Extension decisions hold for a whole block, so a single
    weak batch cannot immediately revoke an extension. New global best weights
    or a recorded adaptation reset the window. No policy is adopted here.
    """
    rows = _primary_reports(reports, stage, budget.minimum_episodes)
    result = dict(action='continue', reason='minimum_exploration_window',
                  evaluated_checkpoints=0, anchor_iteration=-1,
                  allotted_evaluations=budget.block_evaluations,
                  maximum_evaluations=budget.maximum_evaluations, trends=[])
    if not rows:
        return result
    if rows[-1].get('nan_episodes', 0):
        return result | dict(action='diagnose', reason='non_finite_evaluation')
    clean = eligible_stage_reports(reports,stage)
    if not clean:return result | dict(action='diagnose',reason='no_numerically_valid_candidate')
    best = max(clean, key=lambda r: (r['success_rate'], r['mean_best_glide_s']))
    anchor = max(checkpoint_iteration(best['checkpoint']),
                 max((a['iteration'] for a in adaptations if a['stage'] == stage), default=-1))
    window = [r for r in rows if checkpoint_iteration(r['checkpoint']) > anchor]
    count = len(window)
    result.update(anchor_iteration=anchor, evaluated_checkpoints=count)
    if count >= budget.maximum_evaluations:
        return result | dict(action='adapt', reason='exploration_budget_exhausted',
                             allotted_evaluations=budget.maximum_evaluations)
    for boundary in range(budget.block_evaluations, count + 1, budget.block_evaluations):
        block = window[boundary-budget.block_evaluations:boundary]
        trend = _block_trend(block, budget)
        trend['ending_iteration'] = checkpoint_iteration(block[-1]['checkpoint'])
        result['trends'].append(trend)
        if not trend['improving']:
            return result | dict(action='adapt', reason='no_measured_recovery',
                                 allotted_evaluations=boundary)
        result.update(allotted_evaluations=min(boundary+budget.block_evaluations,
                                              budget.maximum_evaluations),
                      reason='recovering_exploration')
    return result


def reviewed_exploration_extension(reports,stage,adaptations=(),budget=ExplorationBudget()):
    """One explicit extra block after a saved review, never an endless extension.

    All completed blocks, including the final one hidden by the normal hard
    limit, must show measured recovery. The original anchor/history is retained.
    The caller must resume the latest saved weights with unchanged exploration
    and stop after this one additional block; physical skill gates are untouched.
    """
    wider=replace(budget,maximum_evaluations=budget.maximum_evaluations+budget.block_evaluations)
    plan=exploration_plan(reports,stage,adaptations,wider)
    if (plan['evaluated_checkpoints']!=budget.maximum_evaluations or
            plan['action']!='continue' or plan['reason']!='recovering_exploration' or
            plan['allotted_evaluations']!=wider.maximum_evaluations):
        raise ValueError('Reviewed extension requires exactly the original limit and recovery in every completed block')
    rows=_primary_reports(reports,stage,budget.minimum_episodes)
    paths={r['checkpoint'] for r in rows if checkpoint_iteration(r['checkpoint'])>plan['anchor_iteration']}
    current=[r for r in reports if r['stage']==stage and r['checkpoint'] in paths]
    valid_paths={r['checkpoint'] for r in eligible_stage_reports(reports,stage)}
    if (not paths<=valid_paths or any(r.get('nan_episodes',0) or r.get('fixed_pose',False) or
            r.get('action_noise_in_evaluation',False) or r.get('memory_reset_each_step',False) for r in current)):
        raise ValueError('Reviewed extension requires numerically valid independent deterministic evaluations')
    return wider,plan


def verified_gate_reports(reports, stage, goal_s, minimum_episodes=256, threshold=.8):
    """Two distinct-seed, same-policy batches; never reward or a fixed pose."""
    if len(reports) < 2 or minimum_episodes < 256 or threshold < .8:
        return False
    rows = reports[-2:]
    try:
        for row in rows:
            p = float(row['success_rate'])
            if (row['stage'] != stage or row['goal_s'] != goal_s or
                    row['episodes'] < minimum_episodes or not threshold <= p <= 1 or
                    row['nan_episodes'] != 0 or row.get('fixed_pose', False) or
                    row.get('action_noise_in_evaluation', False) or row.get('memory_reset_each_step',False)):
                return False
            if stage == 'self-launch' and (row['injected_speed_min'] != 0 or row['injected_speed_max'] != 0):
                return False
        if rows[0]['seed'] == rows[1]['seed'] or rows[0]['checkpoint'] != rows[1]['checkpoint']:
            return False
        hashes = [r.get('checkpoint_sha256') for r in rows]
        if any(h is not None for h in hashes):
            if hashes[0] != hashes[1] or re.fullmatch(r'[0-9a-f]{64}', hashes[0] or '') is None:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def audit_stage_history(state, stage_goals):
    """Read-only proof audit. Historical success is not current-policy retention."""
    issues = []
    passed = state.get('passed_stages', [])
    if len(passed) > len(stage_goals):
        return ['too_many_passed_stages']
    for i, item in enumerate(passed):
        stage, goal = stage_goals[i]
        if (item.get('stage') != stage or
                not verified_gate_reports(item.get('reports', []), stage, goal) or
                any(r.get('checkpoint') != item.get('checkpoint') for r in item.get('reports', [])[-2:])):
            issues.append(f'invalid_stage_proof:{stage}')
    names = [s for s, _ in stage_goals]
    current = state.get('stage')
    if current not in names:
        issues.append('unknown_current_stage')
    elif names.index(current) > len(passed):
        issues.append('skipped_unpassed_stage')
    if state.get('final_self_launch_completed') and (len(passed) != len(stage_goals) or issues):
        issues.append('unverified_completion')
    return issues


FAILURE_CHECK_NAMES = ('finite', 'left_support', 'right_air', 'clearance',
                       'forward_speed', 'lateral_speed', 'heading', 'cross_track',
                       'upright', 'blade_yaw', 'phase', 'no_body_contact')
FAILURE_VALUE_NAMES = ('speed_m_s', 'com_speed_m_s', 'clearance_m', 'upright_cos',
                       'support_yaw_rad', 'capture_error_m', 'left_front_force_n',
                       'left_rear_force_n', 'right_force_n', 'waist_roll_deg',
                       'waist_pitch_deg')


class FirstBreakDiagnostics:
    """First >=0.10-s valid streak break, restricted to first episodes.

    Capture happens in an evaluation-only metric BEFORE automatic resets.
    Reasons can overlap. Failed trials without such a break remain explicit,
    not silently reclassified as successful or dropped from the denominator.
    """
    def __init__(self, episodes, step_dt, device, established_s=.10):
        import torch
        self.torch = torch
        self.step_dt = step_dt
        self.established_steps = round(established_s / step_dt)
        self.streak = torch.zeros(episodes, dtype=torch.long, device=device)
        self.maximum_streak = torch.zeros_like(self.streak)
        self.seen = torch.zeros(episodes, dtype=torch.bool, device=device)
        self.reasons = torch.zeros(episodes, len(FAILURE_CHECK_NAMES), dtype=torch.bool, device=device)
        self.values = torch.zeros(episodes, len(FAILURE_VALUE_NAMES), device=device)
        self.times = torch.zeros(episodes, device=device)
        self.streak_at_break = torch.zeros(episodes, device=device)

    def observe(self, checks, values, alive, step):
        torch = self.torch
        valid = checks.all(1)
        event = alive & ~self.seen & (self.streak >= self.established_steps) & ~valid
        self.reasons = torch.where(event[:, None], ~checks, self.reasons)
        self.values = torch.where(event[:, None], values, self.values)
        self.times = torch.where(event, (step + 1) * self.step_dt, self.times)
        self.streak_at_break = torch.where(event, self.streak * self.step_dt, self.streak_at_break)
        self.seen |= event
        self.streak = torch.where(alive & valid, self.streak + 1, 0)
        self.maximum_streak = torch.maximum(self.maximum_streak, self.streak)

    def report(self, success):
        failed = ~success.bool()
        selected = failed & self.seen
        failures = int(failed.sum())
        observed = int(selected.sum())
        reason_counts = {name: int((self.reasons[:, i] & selected).sum())
                         for i, name in enumerate(FAILURE_CHECK_NAMES)}
        report = dict(version=1, interpretation='first_established_streak_break_in_failed_first_episodes',
                      reason_counts_overlap=True, failed_episodes=failures,
                      recorded_break_episodes=observed,
                      failed_without_recorded_break=int((failed & ~self.seen).sum()),
                      failed_without_established_streak=int((failed & (self.maximum_streak < self.established_steps)).sum()),
                      reason_counts=reason_counts,
                      reason_pct_of_failed={k: 100*v/max(failures, 1) for k, v in reason_counts.items()})
        if observed:
            report['mean_first_break_time_s'] = float(self.times[selected].mean())
            report['mean_streak_before_break_s'] = float(self.streak_at_break[selected].mean())
            # Small per-trial evidence, no full trajectory or privileged policy.
            ids = selected.nonzero().flatten().cpu().tolist()
            reasons = self.reasons.cpu().tolist(); values = self.values.cpu().tolist()
            means={}; nonfinite={}
            for column,name in enumerate(FAILURE_VALUE_NAMES):
                finite=[values[i][column] for i in ids if math.isfinite(values[i][column])]
                means[name]=statistics.mean(finite) if finite else None
                nonfinite[name]=observed-len(finite)
            report['mean_values_at_break']=means
            report['nonfinite_value_counts']=nonfinite
            values=[[v if math.isfinite(v) else None for v in row] for row in values]
            times = self.times.cpu().tolist(); streaks = self.streak_at_break.cpu().tolist()
            report['trials'] = [dict(env_id=i, time_s=times[i], preceding_streak_s=streaks[i],
                                    reasons=[name for j, name in enumerate(FAILURE_CHECK_NAMES) if reasons[i][j]],
                                    values=dict(zip(FAILURE_VALUE_NAMES, values[i]))) for i in ids]
        return report


def cap_scalar_exploration_std(current, maximum):
    """Lower a scalar Gaussian's action-wise noise; never raise a quiet joint."""
    import torch
    if not math.isfinite(maximum) or not 0 < maximum <= 1:
        raise ValueError('Exploration cap must be finite and in (0, 1]')
    if not torch.isfinite(current).all() or (current <= 0).any():
        raise ValueError('Scalar exploration parameters must be finite and positive')
    return current.clamp_max(maximum)


def safe_evaluation_score(reports):
    """Do not promote a candidate if either primary or confirmation has NaNs."""
    if not reports:raise ValueError('No independent evaluation evidence')
    for row in reports:
        if row['nan_episodes']:
            raise ValueError('Non-finite primary or confirmation evaluation; preserve the prior best')
        for key in ('success_rate','mean_best_glide_s','max_glide_s'):
            if not math.isfinite(float(row[key])):
                raise ValueError('Non-finite independent evaluation score')
        if not 0 <= row['success_rate'] <= 1:
            raise ValueError('Invalid independent success rate')
    return reports[0]['success_rate'],reports[0]['mean_best_glide_s']


def eligible_stage_reports(reports, stage):
    """Keep primary candidates only if ALL recorded batches are numerically valid."""
    groups={}
    for row in reports:
        if row['stage']==stage:groups.setdefault(row['checkpoint'],[]).append(row)
    result=[]
    for rows in groups.values():
        valid=True
        for row in rows:
            try:
                p=float(row['success_rate']);duration=float(row['mean_best_glide_s'])
                if row.get('nan_episodes',0) or not math.isfinite(p) or not 0<=p<=1 or not math.isfinite(duration) or duration<0:
                    valid=False
                if 'max_glide_s' in row and (not math.isfinite(float(row['max_glide_s'])) or row['max_glide_s']<duration-1e-5):
                    valid=False
            except (ValueError,TypeError,KeyError):valid=False
        if valid:result.append(rows[0])
    return result
