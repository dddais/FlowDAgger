"""Regression checks for experiment orchestration and paired result integrity."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from flowdagger_pi05.flow_recovery import run_task, schedule, summarize


def test_training_ready_accepts_successful_retry_but_not_eval(tmp_path):
    directory = tmp_path / 'jobs/task_07/offline_attempt03'
    directory.mkdir(parents=True)
    marker = directory / 'COMPLETED.json'
    marker.write_text(json.dumps({'task': 7, 'stage': 'eval'}))
    assert not schedule.training_ready(tmp_path, 7)
    marker.write_text(json.dumps({'task': 7, 'stage': 'offline'}))
    assert schedule.training_ready(tmp_path, 7)
    assert not schedule.training_ready(tmp_path, 6)


def test_scheduler_finishes_without_querying_gpu_after_last_exit(tmp_path, monkeypatch):
    class FinishedChild:
        pid = 123
        returncode = 0
        def poll(self):
            return 0
    calls = []
    def gpu_status(*args, **kwargs):
        calls.append(args)
        assert len(calls) == 1, 'finished jobs must not depend on another GPU query'
        return '0, 24000, 0\n'
    monkeypatch.setattr(schedule.subprocess, 'check_output', gpu_status)
    monkeypatch.setattr(schedule.subprocess, 'Popen', lambda *a, **kw: FinishedChild())
    monkeypatch.setattr(schedule.time, 'sleep', lambda _: None)
    monkeypatch.setattr(schedule.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['schedule', '--root', str(tmp_path), '--config-prefix',
                                    str(tmp_path / 'config'), '--stage', 'offline', '--tasks', '0'])
    schedule.main()
    result = json.loads((tmp_path / 'schedule_offline/summary.json').read_text())
    assert result['tasks'] == [0] and result['failed_tasks'] == []


def test_cancelled_worker_stops_its_running_child(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    real_sleep = run_task.time.sleep
    children = []
    def launch(*args, **kwargs):
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    interrupted = False
    def cancel(delay):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(delay)
    monkeypatch.setattr(run_task.subprocess, 'Popen', launch)
    monkeypatch.setattr(run_task.time, 'sleep', cancel)
    monkeypatch.setattr(run_task.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['run_task', '--root', str(tmp_path), '--config',
                                    str(tmp_path / 'config.json'), '--task', '0', '--gpu', '0',
                                    '--port', '18010', '--stage', 'offline'])
    try:
        with pytest.raises(KeyboardInterrupt):
            run_task.main()
        assert len(children) == 1 and children[0].poll() is not None
        assert not (tmp_path / 'jobs/task_00/offline/COMPLETED.json').exists()
    finally:
        for child in children:
            run_task.stop_process(child)


def test_worker_retry_resumes_existing_training_run(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).parents[1] / 'config/libero_offline.json').read_text())
    config['output_dir'] = str(tmp_path / 'models')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    training_run = tmp_path / 'models/task_00/run.json'
    training_run.parent.mkdir(parents=True)
    training_run.write_text('{}')
    commands = []
    class FinishedChild:
        pid, returncode = 123, 0
        def poll(self):
            return 0
    def launch(command, **kwargs):
        commands.append(command)
        return FinishedChild()
    monkeypatch.setattr(run_task.subprocess, 'Popen', launch)
    monkeypatch.setattr(run_task.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['run_task', '--root', str(tmp_path), '--config', str(config_path),
                                    '--task', '0', '--gpu', '0', '--port', '18010', '--stage', 'offline',
                                    '--job-label', 'offline_retry01'])
    run_task.main()
    assert '--resume' not in commands[0] and '--resume' in commands[1]
    assert (tmp_path / 'jobs/task_00/offline_retry01/COMPLETED.json').is_file()


def test_scheduler_cleans_owned_worker_on_failure(tmp_path, monkeypatch):
    class ActiveChild:
        pid = 123
        def poll(self):
            return None
    child, stopped, commands = ActiveChild(), [], []
    def launch(command, **kwargs):
        commands.append(command)
        assert kwargs['start_new_session']
        return child
    def fail(_):
        raise RuntimeError('scheduler interrupted')
    monkeypatch.setattr(schedule.subprocess, 'Popen', launch)
    monkeypatch.setattr(schedule.subprocess, 'check_output', lambda *a, **kw: '0, 24000, 0\n')
    monkeypatch.setattr(schedule, 'stop_process', lambda p, **kw: stopped.append(p))
    monkeypatch.setattr(schedule.time, 'sleep', fail)
    monkeypatch.setattr(schedule.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['schedule', '--root', str(tmp_path), '--config-prefix',
                                    str(tmp_path / 'config'), '--stage', 'eval', '--tasks', '0',
                                    '--schedule-label', 'eval_retry', '--evaluation-dir', 'eval_retry'])
    with pytest.raises(RuntimeError, match='scheduler interrupted'):
        schedule.main()
    assert stopped == [child]
    assert commands[0][-4:] == ['--job-label', 'eval_retry', '--evaluation-dir', 'eval_retry']


@pytest.mark.parametrize('protocol_drift', [False, True])
def test_summary_handles_task_subset_and_rejects_protocol_drift(tmp_path, monkeypatch, protocol_drift):
    for mode in ('standard', 'recovery'):
        trials = [{'id': i, 'task_index': 0, 'task_key': 'suite/task', 'init_index': i,
                   'bddl': '/scenes/episode_001/step.bddl'} for i in range(2)]
        plan = {'tasks': {'suite/task': {'task_index': 0}}, 'settings': {'query_freq': 10}, 'trials': trials}
        plan_path = tmp_path / f'plans/task_00/{mode}.json'
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan))
        for policy in ('base', 'steering'):
            directory = tmp_path / f'eval_v2/{policy}/task_00/{mode}'
            directory.mkdir(parents=True)
            (directory / 'summary.json').write_text('{}')
            actual = dict(plan, settings={'query_freq': 5 if protocol_drift and policy == 'steering' else 10})
            (directory / 'plan.json').write_text(json.dumps(actual))
            server = dict(mode=policy, checkpoint_identity='same', bridge='same', action_horizon=10, action_dim=7)
            (directory / 'server.json').write_text(json.dumps(server))
            records = [dict(t, status='completed', success=(policy == 'steering' or t['id'] == 0)) for t in trials]
            (directory / 'trials.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    monkeypatch.setattr(summarize, 'bootstrap_interval', lambda _: [0., 1.])
    output = tmp_path / 'analysis'
    monkeypatch.setattr(sys, 'argv', ['summarize', '--root', str(tmp_path), '--output', str(output), '--tasks', '0'])
    if protocol_drift:
        with pytest.raises(AssertionError, match='protocol drift'):
            summarize.main()
        assert not (output / 'comparison.json').exists()
    else:
        summarize.main()
        result = json.loads((output / 'comparison.json').read_text())
        assert len(result['rows']) == 4
        for mode in ('standard', 'recovery'):
            assert result['comparisons'][mode]['macro_base'] == 0.5
            assert result['comparisons'][mode]['macro_steering'] == 1.
