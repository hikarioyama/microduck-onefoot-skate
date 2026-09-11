"""Export the scalar training trajectories of this work to portable CSV.

The training logs live in the upstream working tree under
`logs/rsl_rl/onefoot/<timestamp>_<recipe>/`. They are 155 MB of 377 raw
TensorBoard event files and are NOT part of the published repository. This tool
reads only the runs that produced the shipped checkpoints and the recorded
negative results, and writes one wide CSV per run plus a machine-readable index.

Design notes
------------
* Runs are named by the ALIAS used throughout REPORT.md / evidence/milestones.json,
  not by timestamp, so the trajectory a reader wants is the one they can find.
* A run directory can contain many event files (the trainer appends one per chunk,
  and a reboot starts a new one). They are merged, and duplicate steps are
  de-duplicated keeping the LAST value, because a restart re-logs the same step.
* Tags ending in `/time` (for example `Train/mean_reward/time`) carry WALL-CLOCK
  seconds in their step column, not the update index, so they are kept in the
  index JSON instead of being forced into the update-indexed CSV.
* Values are written with 6 significant digits. This bounds the file size and is
  far more precision than a learning curve needs; the raw logs remain the
  authority for any number quoted in the report.

Usage
-----
    uv run --locked python local/export-training-trajectories.py \
        --out /path/to/deliverable/trajectories
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# alias -> run directory relative to the onefoot log root. The aliases match
# evidence/milestones.json and REPORT.md. `main-expert` is the upstream
# waist-balance curriculum run whose final checkpoint is the frozen one-foot
# expert the bridge preserves.
RUNS = {
    'main-expert': '2026-09-10_19-55-19_history-pilot-control/balance-100',
    'retained-bridge-pilot': '2026-09-10_12-59-15_retained-skill-bridge-pilot',
    'unload': '2026-09-10_16-46-41_credit-bridge-unload-pilot',
    'unload-extension-01': '2026-09-10_17-28-14_credit-bridge-unload-pilot',
    'unload-extension-02': '2026-09-10_18-12-30_credit-bridge-unload-pilot',
    'unload-extension-04': '2026-09-10_18-51-50_credit-bridge-unload-pilot',
    'transfer-near-extension-04': '2026-09-10_19-11-26_credit-bridge-transfer-near-pilot',
    'transfer-extension-04': '2026-09-10_19-22-07_credit-bridge-transfer-pilot',
    'full-extension-04': '2026-09-10_19-36-35_credit-bridge-full-pilot',
    'full-reboot-01': '2026-09-10_23-11-22_credit-bridge-full-pilot',
    'full-v16': '2026-09-10_23-52-51_credit-bridge-full-pilot',
    'full-v16-02': '2026-09-11_00-30-01_credit-bridge-full-pilot',
    'full-v16b': '2026-09-11_01-13-00_credit-bridge-full-pilot',
    'glide-glide-01': '2026-09-11_01-53-58_credit-bridge-glide-pilot',
    'glide-glide-02': '2026-09-11_03-22-06_credit-bridge-glide-pilot',
}

# alias -> its curriculum state directory. The state file holds the screening
# evaluations (64 first episodes, seed 70601, every 100 updates) that the gate
# and the report judge, which the TensorBoard scalars do NOT contain: the
# Episode_Metrics/* tags are 4096-env training aggregates, not this evaluation.
# `main-expert` and `retained-bridge-pilot` predate the credit-bridge state
# files and have no screening series here.
STATE_DIRS = {
    'unload': 'unload-training',
    'unload-extension-01': 'unload-extension-01-training',
    'unload-extension-02': 'unload-extension-02-training',
    'unload-extension-04': 'unload-extension-04-training',
    'transfer-near-extension-04': 'transfer-near-extension-04-training',
    'transfer-extension-04': 'transfer-extension-04-training',
    'full-extension-04': 'full-extension-04-training',
    'full-reboot-01': 'full-reboot-01-training',
    'full-v16': 'full-v16-training',
    'full-v16-02': 'full-v16-02-training',
    'full-v16b': 'full-v16b-training',
    'glide-glide-01': 'glide-glide-01-training',
    'glide-glide-02': 'glide-glide-02-training',
}


def read_run(directory):
    """Merged scalar series of one run: {tag: {step: value}} plus file provenance."""
    accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0})
    accumulator.Reload()
    series = {}
    for tag in accumulator.Tags()['scalars']:
        # Last write wins: a chunk boundary re-logs the step it stopped on.
        values = {}
        for event in accumulator.Scalars(tag):
            values[event.step] = event.value
        series[tag] = values
    files = sorted(p for p in directory.rglob('events.out.tfevents*') if p.is_file())
    provenance = [{
        'name': str(p.relative_to(directory)),
        'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    } for p in files]
    return series, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path,
                        default=Path('logs/rsl_rl/onefoot'),
                        help='onefoot TensorBoard log root (upstream working tree)')
    parser.add_argument('--out', type=Path, required=True,
                        help='destination directory for the exported trajectories')
    parser.add_argument('--only', nargs='*', default=None,
                        help='restrict to these aliases')
    parser.add_argument('--states', type=Path,
                        default=Path('local/onefoot-bridge-curriculum'),
                        help='curriculum state root holding the screening reports')
    args = parser.parse_args()

    wanted = args.only or list(RUNS)
    unknown = sorted(set(wanted) - set(RUNS))
    if unknown:
        raise SystemExit(f'Unknown aliases: {unknown}')

    args.out.mkdir(parents=True, exist_ok=True)
    index = {'log_root': str(args.logs), 'runs': {}}

    for alias in wanted:
        directory = args.logs / RUNS[alias]
        if not directory.is_dir():
            raise SystemExit(f'Missing run directory for {alias}: {directory}')
        series, provenance = read_run(directory)

        indexed = {tag: values for tag, values in series.items()
                   if not tag.endswith('/time')}
        wall_clock = {tag: sorted(values.items()) for tag, values in series.items()
                      if tag.endswith('/time')}
        if not indexed:
            raise SystemExit(f'No update-indexed scalars in {directory}')

        # The union of steps, so a tag logged at a different cadence (the episode
        # aggregates appear one update later than the training-loop series) is
        # placed at its own step rather than being shifted onto someone else's.
        steps = sorted({step for values in indexed.values() for step in values})
        columns = sorted(indexed)
        path = args.out / f'{alias}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['step', *columns])
            for step in steps:
                writer.writerow([str(step)] + [
                    '' if (value := indexed[column].get(step)) is None else f'{value:.6g}'
                    for column in columns])

        index['runs'][alias] = {
            'run_dir': RUNS[alias],
            'csv': path.name,
            'csv_bytes': path.stat().st_size,
            'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'updates': [steps[0], steps[-1]],
            'rows': len(steps),
            'columns': ['step', *columns],
            'tags': {tag: {'count': len(values), 'first_step': min(values),
                           'last_step': max(values)}
                     for tag, values in sorted(series.items())},
            'wall_clock_tags': {tag: points for tag, points in sorted(wall_clock.items())},
            'event_files': provenance,
        }
        print(f'{alias:28s} rows={len(steps):5d} columns={len(columns):3d} '
              f'{path.stat().st_size/1024:8.1f} KB  {RUNS[alias]}')

    index_path = args.out / 'index.json'
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')

    # The screening series: the numbers the gate and the report actually judge.
    screening = {'state_root': str(args.states), 'runs': {}}
    for alias in STATE_DIRS:
        state_path = args.states / STATE_DIRS[alias] / 'state.json'
        if not state_path.is_file():
            print(f'{alias:28s} no screening state ({state_path})')
            continue
        state = json.loads(state_path.read_text())
        screening['runs'][alias] = {
            'state_dir': STATE_DIRS[alias],
            'state_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
            'status': state.get('status'),
            'stage': state.get('stage'),
            'updates': state.get('updates'),
            'stage_gate_passed': state.get('stage_gate_passed'),
            'initial_assistance': state.get('initial_assistance'),
            'checkpoint': state.get('checkpoint'),
            'checkpoint_sha256': state.get('checkpoint_sha256'),
            'reports': state.get('reports', []),
        }
        print(f'{alias:28s} screening reports={len(state.get("reports", [])):3d}')
    screening_path = args.out / 'screening.json'
    screening_path.write_text(json.dumps(screening, indent=1, sort_keys=True) + '\n')

    total = sum(row['csv_bytes'] for row in index['runs'].values())
    print(f'\n{len(index["runs"])} runs, {total/1048576:.1f} MB of CSV -> {index_path}')
    print(f'{len(screening["runs"])} screening series -> {screening_path}')


if __name__ == '__main__':
    main()
