"""Finite-budget dynamic-transfer PPO run with an explicit actor-only warm start."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks.microduck_onefoot_dynamic_env_cfg import DYNAMIC_TASK,DYNAMIC_ASSISTED_TASK,SEED_POSE


from mjlab_microduck.tasks.microduck_onefoot_aligned_env_cfg import ALIGNED_TASK,ALIGNED_ASSISTED_TASK


def warm_actor_state(previous,current):
    """Keep the learned policy/normalizer, but restore fresh exploration noise."""
    assert previous.keys()==current.keys(), 'Actor schema mismatch'
    return {key:current[key] if key.startswith('distribution.') else previous[key]
            for key in current}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('assisted','full'),default='assisted')
    p.add_argument('--recipe',choices=('dynamic','aligned'),default='dynamic')
    p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=1500)
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed-probability',type=float,default=.70)
    a=p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert '5070 Ti' in torch.cuda.get_device_name(0)
    configure_torch_backends()
    task=DYNAMIC_ASSISTED_TASK if a.stage=='assisted' else DYNAMIC_TASK
    if a.recipe=='aligned': task=ALIGNED_ASSISTED_TASK if a.stage=='assisted' else ALIGNED_TASK
    cfg=load_env_cfg(task); cfg.scene.num_envs=a.num_envs
    cfg.sim.nan_guard.enabled=True; cfg.seed=42
    assert 0<=a.seed_probability<=1
    cfg.events['onefoot_seed'].params['probability']=a.seed_probability
    rl=load_rl_cfg(task); rl.logger='tensorboard'; rl.save_interval=100
    rl.max_iterations=a.iterations
    name=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+('_smoke-' if a.smoke else '_')+rl.run_name
    run=Path('logs/rsl_rl/onefoot')/name
    run.mkdir(parents=True,exist_ok=False)
    dump_yaml(run/'params'/'env.yaml',asdict(cfg)); dump_yaml(run/'params'/'agent.yaml',asdict(rl))
    # Include untracked experiment files as well as the changed tracked source.
    for source in [Path('local/train-dynamic-onefoot.py'),Path('src/mjlab_microduck/tasks/mdp.py'),
                   Path('src/mjlab_microduck/tasks/microduck_onefoot_env_cfg.py'),
                   Path('src/mjlab_microduck/tasks/microduck_onefoot_dynamic_env_cfg.py'),
                   Path('src/mjlab_microduck/tasks/microduck_onefoot_aligned_env_cfg.py')]:
        target=run/'source_snapshot'/source
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device='cuda:0'),clip_actions=rl.clip_actions)
    try:
        runner=load_runner_cls(task)(env,deepcopy(asdict(rl)),str(run),device='cuda:0')
        if a.warm_start:
            ckpt=torch.load(a.warm_start,map_location='cuda:0',weights_only=False)
            actor=runner.alg.actor
            state=warm_actor_state(ckpt['actor_state_dict'],actor.state_dict())
            actor.load_state_dict(state,strict=True)
            print('Actor/normalizer warm start:',a.warm_start,flush=True)
            print('Fresh critic, optimizer and exploration; iteration resets to 0',flush=True)
        else:
            # Begin near the physically tested balance target, not v1's failed coast.
            robot=env.unwrapped.scene['robot']
            servo_ids,names=robot.find_joints(r'^(?!passive_).*')
            target=torch.tensor([SEED_POSE[name] for name in names],device='cuda:0')
            bias=target-robot.data.default_joint_pos[0,servo_ids]
            last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
            with torch.no_grad(): last.weight.zero_(); last.bias.copy_(bias)
            print('Fresh actor initialized to tested balance joint target; no learned success assumed',flush=True)
        provenance={'task':task,'recipe':a.recipe,'stage':a.stage,'run_dir':str(run),'budget':a.iterations,
                    'seed_probability':a.seed_probability,'initialization':'actor_warm_start' if a.warm_start else 'balance_joint_target',
                    'warm_start':str(a.warm_start) if a.warm_start else None,
                    'warm_start_sha256':hashlib.sha256(a.warm_start.read_bytes()).hexdigest() if a.warm_start else None}
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        if not a.smoke:
            pointer=Path('local/onefoot-active.json')
            temp=pointer.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(provenance,log_file='onefoot-dynamic-training.log',
                label='頭・脚・軸足操舵の練習（初速補助・70%の片足初期姿勢補助）' if a.stage=='assisted' else '自力加速から動的重心移動'),ensure_ascii=False,indent=2)+'\n')
            temp.replace(pointer)
        runner.add_git_repo_to_log(__file__)
        print('RUN_DIRECTORY:',run,flush=True)
        # Start physical episodes at phase zero; do not randomize timeout counters.
        runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    finally:
        env.close()


if __name__=='__main__': main()
