import functools as ft
import random
import time
from unittest.mock import patch, DEFAULT

import pytest

from amlb.job import MultiThreadingJobRunner, State
from amlb.utils import Timeout

from dummy import DummyJob

steps_per_job = 6


@pytest.mark.slow
def test_run_multiple_jobs():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", result=i, steps=seq_steps, verbose=True, duration_secs=0.1)
            for i in range(n_jobs)]
    assert len(seq_steps) == n_jobs
    assert all(step == 'created' for _, step in seq_steps)

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3)
    results = runner.start()
    print(results)

    assert len(results) == n_jobs
    assert len(seq_steps) == n_jobs * steps_per_job


@pytest.mark.slow
def test_run_multiple_jobs_with_delay():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", result=i, steps=seq_steps, verbose=True, duration_secs=1)
            for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)
    results = runner.start()
    print(results)

    assert len(results) == n_jobs
    assert len(seq_steps) == n_jobs * steps_per_job


@pytest.mark.slow
def test_stop_runner_during_job_run():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, result=i, steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)
    with Timeout(timeout_secs=1, on_timeout=runner.stop):
        results = runner.start()
    print(results)

    assert len(results) < n_jobs
    assert len(seq_steps) < n_jobs * steps_per_job
    cancelled_jobs = [j for j, s in seq_steps if s == 'cancelling']
    assert len(cancelled_jobs) > 1
    first_cancelled = cancelled_jobs[0]
    first_cancelled_idx = int(first_cancelled.split('_')[1])
    last_result = results[-1]
    assert last_result.result is None
    assert last_result.duration > 0
    assert cancelled_jobs == [f"job_{i}" for i in range(first_cancelled_idx, n_jobs)]

    for state in ['created', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s[1] == state, seq_steps))) == n_jobs
    assert len(list(filter(lambda s: s[1] == 'starting', seq_steps))) == len(results)
    assert len(list(filter(lambda s: s[1] == 'completing', seq_steps))) + len(cancelled_jobs) == n_jobs


@pytest.mark.slow
def test_reschedule_job_default():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=1, result=i, steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)

    def _run(self, mock, ori):
        if mock.call_count < 3:
            self.reschedule()
            return
        return ori()

    rescheduled_job = jobs[4]
    ori = rescheduled_job._run
    with patch.object(rescheduled_job, "_run") as mock:
        mock.side_effect = ft.partial(_run, rescheduled_job, mock, ori)
        results = runner.start()
    print(results)

    assert len(results) == n_jobs
    assert len(seq_steps) > n_jobs * steps_per_job
    normal_job_steps = [s for n, s in seq_steps if n != rescheduled_job.name]
    rescheduled_job_steps = [s for n, s in seq_steps if n == rescheduled_job.name]

    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - 1

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'rescheduling', rescheduled_job_steps))) == 2
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1

    assert seq_steps.index(('job_4', 'completing')) > seq_steps.index(('job_5', 'completing'))


@pytest.mark.slow
def test_reschedule_job_enforce_job_priority():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=1, result=i, steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2,
                                     queueing_strategy=MultiThreadingJobRunner.QueueingStrategy.enforce_job_priority)

    def _run(self, mock, ori):
        if mock.call_count < 3:
            self.reschedule()
            return
        return ori()
        # return DEFAULT  # ensures that the wrapped function is called after the side effect
                          # (doesn't work as expected on linux, even when using patch.object(job, '_run', wraps=job._run)

    rescheduled_job = jobs[4]
    ori = rescheduled_job._run
    with patch.object(rescheduled_job, "_run") as mock:
        mock.side_effect = ft.partial(_run, rescheduled_job, mock, ori)
        results = runner.start()
    print(results)

    assert len(results) == n_jobs
    assert len(seq_steps) > n_jobs * steps_per_job
    normal_job_steps = [s for n, s in seq_steps if n != rescheduled_job.name]
    rescheduled_job_steps = [s for n, s in seq_steps if n == rescheduled_job.name]

    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - 1

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'rescheduling', rescheduled_job_steps))) == 2
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1

    assert seq_steps.index(('job_4', 'completing')) < seq_steps.index(('job_5', 'completing'))


@pytest.mark.slow
@pytest.mark.stress
def test_reschedule_job_high_parallelism():
    seq_steps = []
    n_jobs = 600
    jobs = [DummyJob(name=f"job_{i}", duration_secs=random.randint(150, 250)/10, result=i,
                     steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=200, delay_secs=0.1,
                                     queueing_strategy=MultiThreadingJobRunner.QueueingStrategy.enforce_job_priority)

    def _run(self, mock, ori):
        if mock.call_count < 3:
            self.reschedule()
            return
        return ori()

    rescheduled_jobs = [j for i, j in enumerate(jobs) if i % 17 == 0]

    for job in rescheduled_jobs:
        ori = job._run
        mock = patch.object(job, "_run").start()
        mock.side_effect = ft.partial(_run, job, mock, ori)
    results = runner.start()

    rescheduled_job_names = [j.name for j in rescheduled_jobs]
    normal_job_steps = [s for n, s in seq_steps if n not in rescheduled_job_names]
    rescheduled_job_steps = [s for n, s in seq_steps if n in rescheduled_job_names]

    assert len(results) == n_jobs
    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - len(rescheduled_job_names)

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == 3 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == 3 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'rescheduling', rescheduled_job_steps))) == 2 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)


@pytest.mark.slow
@pytest.mark.stress
def test_reschedule_does_not_lead_to_runner_cancellation_on_high_parallelism():
    seq_steps = []
    attempts = 5
    n_jobs = 200
    jobs = [DummyJob(name=f"job_{i}", duration_secs=random.randint(10, 20)/10, result=i,
                     steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    def _run(self, ori=None):
        attempt = self.ext['attempt'] = self.ext.get('attempt', 0) + 1
        if attempt < attempts:
            self.reschedule()
            time.sleep(self._duration_secs/10)
            return
        return ori()

    def _on_state(self, state, ori=None):
        ori(state)
        if state is State.rescheduling:
            time.sleep(random.randint(1, 10)/10)
        elif state is State.stopped:
            print(f"#results = {len(runner.results)}")

    rescheduled_jobs = [j for i, j in enumerate(jobs) if i % 5 == 0]
    for job in rescheduled_jobs:
        job._run = ft.partialmethod(_run, ori=job._run).__get__(job, None)
    for job in jobs:
        job._on_state = ft.partialmethod(_on_state, ori=job._on_state).__get__(job, None)

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=50, delay_secs=0.01,
                                     queueing_strategy=MultiThreadingJobRunner.QueueingStrategy.enforce_job_priority)
    results = runner.start()
    # print(results)

    rescheduled_job_names = [j.name for j in rescheduled_jobs]
    normal_job_steps = [s for n, s in seq_steps if n not in rescheduled_job_names]
    rescheduled_job_steps = [s for n, s in seq_steps if n in rescheduled_job_names]

    assert len(results) == n_jobs
    assert len(list(filter(lambda s: s == 'cancelling', rescheduled_job_steps))) == 0
    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - len(rescheduled_job_names)

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == attempts * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == attempts * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'rescheduling', rescheduled_job_steps))) == (attempts-1) * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)

