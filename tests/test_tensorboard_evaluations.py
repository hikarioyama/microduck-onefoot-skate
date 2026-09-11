import importlib.util
import json
from pathlib import Path
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def module():
    path=Path(__file__).parents[1]/'local/tensorboard-evaluations.py'
    spec=importlib.util.spec_from_file_location('tb_evaluations',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def report(**kwargs):
    return dict(dict(stage='balance-100',seed=123,goal_s=1.,episodes=256,
        success_rate=.25,success_050=.99,success_100=.25,success_200=0.,
        mean_best_glide_s=.9,max_glide_s=1.12,p10_best_glide_s=.75,
        mean_support_yaw=.1,injected_speed_min=.28,injected_speed_max=.32,
        nan_episodes=0,termination_counts={'fell_over':64,'replanted_swing':128}),**kwargs)


def test_percentages_units_and_confirmation_are_explicit():
    m=module();values=m.scalar_values(report())
    assert values['Evaluation/success_rate_pct']==25.
    assert values['Evaluation/mean_best_glide_s']==.9
    assert values['Evaluation/gate_threshold_pct']==80.
    assert values['Evaluation/termination_pct/fell_over']==25.
    assert values['Evaluation/support_yaw_deg']==pytest.approx(5.72957795)
    assert 'EvaluationConfirm/success_rate_pct' in m.scalar_values(report(),True)
    with pytest.raises(ValueError):m.scalar_values(report(success_rate=float('nan')))
    with pytest.raises(ValueError):m.scalar_values(report(episodes=0))


def test_backfill_is_readable_by_tensorboard_and_restart_safe(tmp_path):
    m=module();m.LOGS=tmp_path/'logs';root=m.LOGS/'example';stage=root/'balance-100'
    stage.mkdir(parents=True);checkpoint=stage/'model_42.pt';checkpoint.write_bytes(b'placeholder')
    rows=[report(checkpoint=str(checkpoint)),report(checkpoint=str(checkpoint),seed=124,success_rate=.3)]
    for row,suffix in zip(rows,('','_confirm')):
        (stage/f'eval_42{suffix}.json').write_text(json.dumps(row))
    state={'run_root':str(root),'evaluations':rows}
    publisher=m.Publisher()
    try:
        assert publisher.publish(state)==2
        assert publisher.publish(state)==0
    finally:publisher.close()
    publisher=m.Publisher()
    try:assert publisher.publish(state)==0
    finally:publisher.close()
    # No training files are changed or created in the parent run.
    assert not list(stage.glob('events.out.*'))
    events=EventAccumulator(str(stage/'evaluation'));events.Reload()
    assert events.Scalars('Evaluation/success_rate_pct')[0].step==42
    assert events.Scalars('Evaluation/success_rate_pct')[0].value==25.
    assert events.Scalars('EvaluationConfirm/success_rate_pct')[0].value==pytest.approx(30.)
    assert len(events.Scalars('Evaluation/success_rate_pct'))==1


def test_changed_or_mismatched_evidence_is_not_silently_published(tmp_path):
    m=module();m.LOGS=tmp_path/'logs';root=m.LOGS/'example';stage=root/'balance-100'
    stage.mkdir(parents=True);checkpoint=stage/'model_42.pt'
    row=report(checkpoint=str(checkpoint));path=stage/'eval_42.json'
    path.write_text(json.dumps(row))
    publisher=m.Publisher()
    try:
        assert publisher.publish({'run_root':str(root),'evaluations':[row]})==1
        changed=dict(row,success_rate=.5);path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):publisher.publish({'run_root':str(root),'evaluations':[row]})
        with pytest.raises(ValueError):publisher.publish({'run_root':str(root),'evaluations':[changed]})
    finally:publisher.close()
