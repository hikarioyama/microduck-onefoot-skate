"""Frozen v15 reward/physical screening, with checkpoint and expert hash guards."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import safe_runner_load
from mjlab_microduck.retained_skill_bridge import retained_skills_digest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('credit_bridge_evaluation_trainer',ROOT/'local/train-credit-bridge.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--stage',choices=(*trainer.BRIDGE_CURRICULUM,'hold-check'),default='full')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--episodes',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert 1<=a.episodes<=256
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
    assert torch.cuda.device_count()==1 and '5070 Ti' in torch.cuda.get_device_name(0)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    main_hash=hashlib.sha256(trainer.MAIN.read_bytes()).hexdigest()
    env,rl=trainer.environment(1,a.seed,a.stage)
    try:
        runner=load_runner_cls(trainer.TASK)(env,deepcopy(asdict(rl)),device='cuda:0')
        safe_runner_load(runner,a.checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
        actor=runner.get_inference_policy(device='cuda:0')
        frozen=retained_skills_digest(actor)
        result=trainer.evaluate(actor,rl,a.seed,a.episodes,a.stage)
        assert retained_skills_digest(actor)==frozen
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==digest
        assert hashlib.sha256(trainer.MAIN.read_bytes()).hexdigest()==main_hash
        result.update(checkpoint=str(a.checkpoint),checkpoint_sha256=digest,
                      frozen_skills_sha256=frozen,main_state_unchanged=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print('CREDIT_FROZEN_EVALUATION',json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':main()
