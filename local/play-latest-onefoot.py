"""Follow active saved policies with mjlab's standard Viser UI and run loop.

Only backend checkpoint discovery/scheduling is added. No custom widgets,
markers, dashboards or simulation loop. Checkpoint updates use the official
CheckpointManager/FETCH_CHECKPOINT path. A stage/recipe change rebuilds the
matching environment and Viser server; browser reconnection is automatic.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
import viser
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager,format_time_ago
from mjlab_microduck.checkpoint_safety import safe_runner_load
from mjlab_microduck.tasks.microduck_onefoot_curriculum_env_cfg import (
    TASK as BASE_TASK,STAGE_BY_NAME,make_curriculum_onefoot_env_cfg,make_curriculum_onefoot_rl_cfg)
from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import (
    TASK as CONTINUOUS_TASK,make_continuous_onefoot_env_cfg,make_continuous_onefoot_rl_cfg)
from mjlab_microduck.tasks.microduck_onefoot_centroid_env_cfg import (
    TASK as CENTROID_TASK,make_centroid_onefoot_env_cfg,make_centroid_onefoot_rl_cfg)
from mjlab_microduck.tasks.microduck_onefoot_memory_env_cfg import (
    TASK as MEMORY_TASK,make_memory_onefoot_env_cfg,make_memory_onefoot_rl_cfg)
from mjlab_microduck.tasks.microduck_onefoot_lean_env_cfg import (
    TASK as LEAN_TASK,make_lean_onefoot_env_cfg,make_lean_onefoot_rl_cfg)
from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import (
    TASK as WAIST_TASK,make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg)
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import (
    TASK as CREDIT_TASK, STAGES as CREDIT_STAGES,
    make_credit_bridge_env_cfg, make_credit_bridge_rl_cfg)
from onefoot_live_source import (ROOT,DEFAULT_STATE,read_source,StableCheckpoints,
                                fingerprint,needs_reload,checkpoint_order)

FACTORIES={
    'curriculum':(BASE_TASK,make_curriculum_onefoot_env_cfg,make_curriculum_onefoot_rl_cfg),
    'continuous':(CONTINUOUS_TASK,make_continuous_onefoot_env_cfg,make_continuous_onefoot_rl_cfg),
    'centroid':(CENTROID_TASK,make_centroid_onefoot_env_cfg,make_centroid_onefoot_rl_cfg),
    'memory':(MEMORY_TASK,make_memory_onefoot_env_cfg,make_memory_onefoot_rl_cfg),
    'lean':(LEAN_TASK,make_lean_onefoot_env_cfg,make_lean_onefoot_rl_cfg),
    'waist':(WAIST_TASK,make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg),
}


class Status:
    def __init__(self,path):
        self.path=path
        self.data=dict(status='starting',pid=os.getpid(),load_count=0,
                       automatic_follow=True,saved_checkpoints_only=True)

    def write(self,**changes):
        self.data.update(changes,updated=datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.data,ensure_ascii=False,indent=2)+'\n')
        temp.replace(self.path)

    def loaded(self,checkpoint,env):
        cfg=env.unwrapped.command_manager.get_term('twist').cfg
        spawn=env.unwrapped.event_manager.get_term_cfg('curriculum_spawn').params
        self.write(status='running',recipe=checkpoint.source.recipe,stage=checkpoint.source.stage,
            run_root=str(checkpoint.source.root),checkpoint=str(checkpoint.path),
            iteration=checkpoint.iteration,fingerprint=list(checkpoint.fingerprint),
            sha256=hashlib.sha256(checkpoint.path.read_bytes()).hexdigest(),
            load_count=self.data['load_count']+1,goal_s=cfg.goal_s,
            injected_speed_range=list(spawn['speed_range']),phase_start=spawn['phase_start'],
            initial_assistance=any(v!=0 for v in spawn['speed_range']) or spawn['phase_start']>0,
            source_state_file=str(checkpoint.source.state_path),training_status=checkpoint.source.training_status,
            display_stage=checkpoint.source.display_stage,display_mode='current_training_stage',
            control_dt=env.unwrapped.step_dt,
            loaded_at=datetime.now(timezone.utc).isoformat(),device=str(env.device),error=None)
        print('LIVE_VIEWER_LOADED:',json.dumps({k:self.data[k] for k in
            ('recipe','stage','iteration','checkpoint','load_count','goal_s','injected_speed_range')}),flush=True)


class EpisodePolicy:
    """Reset official policy memory on automatic as well as manual env resets."""
    def __init__(self,actor,env):self.actor=actor;self.env=env

    def __call__(self,obs):
        self.actor.reset(self.env.episode_length_buf==0)
        return self.actor(obs)

    def reset(self,dones=None):self.actor.reset(dones)


def viewer_configuration(source):
    if source.recipe=='credit-bridge':
        if source.stage not in CREDIT_STAGES:raise ValueError('Unknown bridge stage')
        task,make_env,make_rl=CREDIT_TASK,make_credit_bridge_env_cfg,make_credit_bridge_rl_cfg
        cfg=make_env(play=False,stage=source.stage)  # Match the current training conditions.
    else:
        if source.stage not in STAGE_BY_NAME:raise ValueError('Unknown curriculum stage')
        task,make_env,make_rl=FACTORIES[source.recipe]
        cfg=make_env(play=True,stage=source.stage)
    cfg.scene.num_envs=1;cfg.sim.nan_guard.enabled=True
    # mjviser's camera elevation has the opposite sign to MuJoCo's native
    # camera convention used in these task configs. Keep the camera above ground.
    cfg.viewer.elevation=abs(cfg.viewer.elevation)
    return cfg,make_rl(),task


def build_environment(source,device='cuda:0'):
    cfg,rl,task=viewer_configuration(source)
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device=device),clip_actions=rl.clip_actions)
    assert env.get_observations()['actor'].shape==(1,61) and env.num_actions==14
    return env,rl,load_runner_cls(task)


class PolicyLoader:
    """Load into a fresh actor so a bad checkpoint never damages the shown policy."""
    def __init__(self,env,rl,runner_cls):
        self.env=env;self.rl=rl;self.runner_cls=runner_cls;self.loaded=None

    def load(self,checkpoint):
        if fingerprint(checkpoint.path)!=checkpoint.fingerprint:
            raise ValueError('Checkpoint changed before loading')
        device=str(self.env.device)
        with torch.random.fork_rng(devices=[] if device=='cpu' else [torch.device(device).index or 0]):
            runner=self.runner_cls(self.env,deepcopy(asdict(self.rl)),device=device)
            safe_runner_load(runner,checkpoint.path,load_cfg={'actor':True},strict=True,map_location=device)
            actor=runner.get_inference_policy(device=device)
            if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
                raise ValueError('Nonfinite saved actor; keeping the previous good display')
        if fingerprint(checkpoint.path)!=checkpoint.fingerprint:
            raise ValueError('Checkpoint changed while loading')
        if actor.is_recurrent:actor.rnn.rnn.flatten_parameters()
        self.loaded=checkpoint
        return EpisodePolicy(actor,self.env)


class FollowingViewer(ViserPlayViewer):
    """Use the stock run loop, GUI, reset, and checkpoint-selection behavior."""
    def __init__(self,env,policy,*,server,source,loader,catalog,state_file,status,poll_seconds,stop,frame_rate=60.):
        self.source=source;self.loader=loader;self.catalog=catalog
        self.state_file=state_file;self.status_file=status;self.poll_seconds=poll_seconds
        self.stop=stop;self.next_poll=0.;self.next_source=None;self.last_discovery_error=None
        self.entries={};self.shown=loader.loaded
        manager=CheckpointManager(current_name=loader.loaded.label,
            fetch_available=self.available,load_checkpoint=self.load_named)
        super().__init__(env,policy,viser_server=server,checkpoint_manager=manager,frame_rate=frame_rate)

    def available(self):
        return [(item.label,format_time_ago(max(0,int(time.time()-item.fingerprint[3]/1e9))))
                for item in sorted(self.entries.values(),key=checkpoint_order)]

    def load_named(self,name):
        checkpoint=self.entries[name]
        try:return self.loader.load(checkpoint)
        except Exception:
            self.catalog.reject(checkpoint)
            raise

    def _poll_source(self):
        if self.stop():self._interrupted=True;return
        now=time.monotonic()
        if now<self.next_poll:return
        self.next_poll=now+self.poll_seconds
        try:
            desired=read_source(self.state_file)
            ready=self.catalog.scan(desired)
            self.last_discovery_error=None
        except (OSError,ValueError,KeyError) as error:
            message=str(error)
            if message!=self.last_discovery_error:
                print('LIVE_VIEWER_WAITING:',message,flush=True)
                self.last_discovery_error=message
                self.status_file.write(status='waiting_for_source',error=message)
            return
        if not ready:
            self.status_file.write(status='waiting_for_checkpoint',desired_stage=desired.stage,
                                   desired_state_file=str(desired.state_path),training_status=desired.training_status)
            return
        latest=ready[-1]
        if desired!=self.source:
            self.next_source=desired;self._interrupted=True
            self.status_file.write(status='switching_stage',next_stage=desired.stage,next_recipe=desired.recipe)
            return
        self.entries={item.label:item for item in ready}
        # A current file can be replaced or disappear while its loaded policy is
        # still valid. Keep its standard dropdown entry without rolling back.
        if self.shown.label not in self.entries:self.entries[self.shown.label]=self.shown
        update=dict(latest_available=str(latest.path),latest_iteration=latest.iteration,
                    source_state_file=str(desired.state_path),training_status=desired.training_status)
        if self.status_file.data.get('status') in ('waiting_for_source','waiting_for_checkpoint') and self._last_error is None:
            update.update(status='running',error=None)
        self.status_file.write(**update)
        if needs_reload(self.shown,latest):
            self._actions.append((ViewerAction.FETCH_CHECKPOINT,'latest'))
        else:
            self._actions.append((ViewerAction.FETCH_CHECKPOINT,'refresh'))

    def _process_actions(self):
        self._poll_source()
        super()._process_actions()

    def _handle_custom_action(self,action,payload):
        if action!=ViewerAction.FETCH_CHECKPOINT:
            return super()._handle_custom_action(action,payload)
        old_name=self._ckpt_mgr.current_name
        old_policy=self.policy
        old_loaded=self.loader.loaded
        had_error=self._last_error is not None
        try:
            if payload=='latest' and self.entries:
                latest=max(self.entries.values(),key=checkpoint_order)
                if latest.label==old_name and needs_reload(self.shown,latest):
                    # The official handler normally skips a same-name file. First
                    # refresh its normal dropdown, then request a selected reload.
                    super()._handle_custom_action(action,'refresh')
                    self._ckpt_mgr.current_name=''
                    payload='selected'
            result=super()._handle_custom_action(action,payload)
        except Exception as error:
            self.policy=old_policy;self._ckpt_mgr.current_name=old_name;self.loader.loaded=old_loaded
            self.status_file.write(status='checkpoint_load_failed',error=str(error))
            print('LIVE_VIEWER_LOAD_FAILED:',repr(error),flush=True)
            return True
        if self.loader.loaded!=self.shown:
            self.shown=self.loader.loaded
            if had_error:self.resume()
            self.status_file.loaded(self.shown,self.env)
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--state-file',type=Path,default=DEFAULT_STATE)
    p.add_argument('--status-file',type=Path,default=ROOT/'local/onefoot-live-viewer.json')
    p.add_argument('--port',type=int,default=8086)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--frame-rate',type=float,help='Rendering only; physics/control timestep stays unchanged')
    p.add_argument('--poll-seconds',type=float,default=2.)
    p.add_argument('--settle-seconds',type=float,default=3.)
    a=p.parse_args()
    if a.device=='cpu':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU playback must not initialize a GPU'
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
        assert torch.cuda.device_count()==1 and '5070 Ti' in torch.cuda.get_device_name(0)
    if a.frame_rate is None:a.frame_rate=60.  # Preserve the stock rendering target.
    if not 1<=a.frame_rate<=120:raise ValueError('Invalid render frame rate')
    if not 1024<=a.port<=65535 or a.poll_seconds<1 or a.settle_seconds<1:
        raise ValueError('Invalid viewer port or polling interval')
    configure_torch_backends();torch.set_num_threads(1 if a.device=='cpu' else 2)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    status=Status(a.status_file);status.write(port=a.port,poll_seconds=a.poll_seconds,render_frame_rate=a.frame_rate)
    catalog=StableCheckpoints(a.settle_seconds)
    try:
        while not stopped:
            try:
                source=read_source(a.state_file)
                ready=catalog.scan(source)
            except (OSError,ValueError,KeyError) as error:
                status.write(status='waiting_for_source',error=str(error));time.sleep(a.poll_seconds);continue
            if not ready:time.sleep(a.poll_seconds);continue
            selected=ready[-1];env=None;server=None;viewer=None;loader=None;policy=None
            try:
                env,rl,runner_cls=build_environment(source,device=a.device)
                loader=PolicyLoader(env,rl,runner_cls)
                try:policy=loader.load(selected)
                except Exception:
                    catalog.reject(selected)
                    raise
                server=viser.ViserServer(host='127.0.0.1',port=a.port,label=f'mjlab / Microduck / {source.display_stage}')
                if server.get_port()!=a.port:
                    raise RuntimeError('Requested viewer port is busy; refusing a different port')
                viewer=FollowingViewer(env,policy,server=server,source=source,loader=loader,
                    catalog=catalog,state_file=a.state_file,status=status,
                    poll_seconds=a.poll_seconds,stop=lambda:stopped,frame_rate=a.frame_rate)
                viewer.entries={item.label:item for item in ready}
                status.loaded(selected,env)
                viewer.run(catch_sigint=False)
            except Exception as error:
                status.write(status='viewer_error',error=str(error))
                print('LIVE_VIEWER_ERROR:',repr(error),flush=True)
                time.sleep(a.poll_seconds)
            finally:
                if server is not None:server.stop()
                if env is not None:env.close()
                env=None;server=None;viewer=None;loader=None;policy=None
                gc.collect()
                if a.device!='cpu':torch.cuda.empty_cache()
    finally:
        status.write(status='stopped')


if __name__=='__main__':main()
