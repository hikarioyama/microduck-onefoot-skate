"""Read-only discovery of an explicitly selected run's completed checkpoints."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]
LOG_ROOT=(ROOT/'logs/rsl_rl/onefoot').resolve()
DEFAULT_STATE=ROOT/'local/onefoot-live-source.json'
LEGACY_STATE=ROOT/'local/onefoot-curriculum-state.json'
COURSE_STATE=ROOT/'local/onefoot-bridge-curriculum/course.json'
RECIPES=('curriculum','continuous','centroid','memory','lean','waist','credit-bridge')
CREDIT_STAGES=('near','unload','transfer-near','transfer','full','hold-check')
CHECKPOINT_RE=re.compile(r'model_(\d+)\.pt\Z')


@dataclass(frozen=True)
class ActiveSource:
    root: Path
    recipe: str
    stage: str
    flat: bool=False
    initial_iteration: int | None=field(default=None,compare=False)
    state_path: Path | None=field(default=None,compare=False)
    training_status: str=field(default='',compare=False)

    @property
    def directory(self):return self.root if self.flat else self.root/self.stage

    @property
    def display_stage(self):
        if self.recipe=='credit-bridge':
            return f'{self.stage} [{"from rest" if self.stage=="full" else "assisted"}]'
        return self.stage


def project_path(value):
    path=Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def read_source(path=DEFAULT_STATE,log_root=LOG_ROOT,*,state_root=None):
    """Follow a stable live pointer and/or course.active_state, never mtime guesses.

    Indirect state pointers stay inside local/. During partially published stage
    transitions callers retain their previously loaded good model and retry.
    """
    path=Path(path)
    if path==DEFAULT_STATE and not path.exists():
        path=COURSE_STATE if COURSE_STATE.exists() else LEGACY_STATE
    allowed_states=(ROOT/'local').resolve() if state_root is None else Path(state_root).resolve()
    seen=set();expected_stage=None
    for _ in range(5):
        path=path.resolve()
        if path in seen:raise ValueError('Cyclic viewer state pointer')
        seen.add(path)
        state=json.loads(path.read_text())
        pointer=state.get('state_file') or state.get('active_state')
        if pointer:
            target=project_path(pointer)
            if not target.is_relative_to(allowed_states):
                raise ValueError('Viewer state pointer leaves the allowed local directory')
            if state.get('active_state'):
                expected_stage=state.get('stage')
            path=target
            continue
        break
    else:raise ValueError('Viewer state pointer chain is too deep')
    if 'log_root' in state and 'onefoot_source' in state:
        root=project_path(state['log_root']);stage=state['stage'];recipe='credit-bridge';flat=True
        if stage not in CREDIT_STAGES:raise ValueError('Unknown retained-bridge stage')
        initial_iteration=None
        if state.get('initial_checkpoint'):
            if project_path(state['initial_checkpoint'])!=root/'model_initial.pt':
                raise ValueError('Initial checkpoint does not belong to the active run')
            initial_iteration=state.get('source_iteration',0)
            if not isinstance(initial_iteration,int) or initial_iteration<0:
                raise ValueError('Invalid initial checkpoint iteration')
    else:
        root=project_path(state['run_root']);stage=state['stage'];recipe=state['recipe'];flat=False
        initial_iteration=None
        if recipe=='credit-bridge':raise ValueError('Bridge runs require their own flat manifest layout')
    if not root.is_relative_to(Path(log_root).resolve()):
        raise ValueError('Active experiment is outside the allowed log directory')
    if recipe not in RECIPES or not re.fullmatch(r'[a-z]+(?:-[a-z0-9]+)*',stage):
        raise ValueError('Invalid active recipe or stage')
    if expected_stage is not None and expected_stage!=stage:
        raise ValueError('Course/stage publication is not consistent yet')
    directory=root if flat else root/stage
    if directory.resolve()!=directory:
        raise ValueError('Stage directory must not redirect to a different location')
    if state.get('run_dir') and project_path(state['run_dir'])!=directory:
        raise ValueError('Training stage/directory update is not complete yet')
    return ActiveSource(root,recipe,stage,flat,initial_iteration,path,state.get('status',''))


def fingerprint(path):
    stat=Path(path).stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns


@dataclass(frozen=True)
class Checkpoint:
    source: ActiveSource
    path: Path
    iteration: int
    fingerprint: tuple

    @property
    def label(self):
        suffix=f' (iteration {self.iteration})' if self.path.name=='model_initial.pt' else ''
        return f'{self.source.display_stage}/{self.path.name}{suffix}'


def checkpoint_order(checkpoint):
    # A trained model_0 supersedes the untrained model_initial at the same index.
    return checkpoint.iteration,checkpoint.path.name!='model_initial.pt'


class StableCheckpoints:
    """Require age and unchanged fingerprints; ignore incomplete/rejected files.

    Only the explicitly selected directory is scanned. Other experiments, best
    pointers, smoke siblings, symlinks and *.tmp files are never selected.
    """
    def __init__(self,settle_seconds=3.):
        if settle_seconds<0:raise ValueError('Negative settle time')
        self.settle_seconds=settle_seconds
        self.seen={};self.failed={}

    def scan(self,source,*,now=None,monotonic=None):
        now=time.time() if now is None else now
        monotonic=time.monotonic() if monotonic is None else monotonic
        ready=[]
        for path in source.directory.glob('model_*.pt'):
            match=CHECKPOINT_RE.fullmatch(path.name)
            if match:iteration=int(match.group(1))
            elif path.name=='model_initial.pt' and source.initial_iteration is not None:
                iteration=source.initial_iteration
            else:continue
            if path.is_symlink():continue
            try:stamp=fingerprint(path)
            except FileNotFoundError:continue
            if stamp[2]<=0:continue
            previous=self.seen.get(path)
            if previous is None or previous[0]!=stamp:
                self.seen[path]=(stamp,monotonic)
                continue
            if self.failed.get(path)==stamp:continue
            if monotonic-previous[1]<self.settle_seconds:continue
            if now-stamp[3]/1e9<self.settle_seconds:continue
            ready.append(Checkpoint(source,path,iteration,stamp))
        return sorted(ready,key=checkpoint_order)

    def reject(self,checkpoint):self.failed[checkpoint.path]=checkpoint.fingerprint


def needs_reload(current,candidate):
    if current is not None and current.source==candidate.source and checkpoint_order(candidate)<checkpoint_order(current):
        return False
    return current is None or (current.path,current.fingerprint)!=(candidate.path,candidate.fingerprint)
