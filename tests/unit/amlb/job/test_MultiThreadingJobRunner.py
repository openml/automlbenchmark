import functools as ft
import random
from unittest.mock import patch, DEFAULT

import pytest

from amlb.job import MultiThreadingJobRunner
from amlb.utils import Timeout

from dummy import DummyJob

steps_per_job = 6


def test_run_multiple_jobs():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", steps=seq_steps, verbose=True, duration_secs=0.1) for i in range(n_jobs)]
    assert len(seq_steps) == n_jobs
    assert all(step == 'created' for _, step in seq_steps)

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3)
    runner.start()
    assert len(seq_steps) == n_jobs * steps_per_job


def test_run_multiple_jobs_with_delay():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", steps=seq_steps, verbose=True, duration_secs=1) for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)
    runner.start()
    assert len(seq_steps) == n_jobs * steps_per_job


def test_stop_runner_during_job_run():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, steps=seq_steps, verbose=True) for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)
    with Timeout(timeout_secs=1, on_timeout=runner.stop):
        runner.start()
    assert len(seq_steps) < n_jobs * steps_per_job


def test_reschedule_job_default():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=1, result=i, steps=seq_steps, verbose=True) for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)

    def _run(self, mock):
        if mock.call_count < 3:
            runner.reschedule(self)
            return
        return DEFAULT  # ensures that the wrapped function is called after the side effect

    rescheduled_job = jobs[4]
    with patch.object(rescheduled_job, "_run", wraps=rescheduled_job._run) as mock:
        mock.side_effect = ft.partial(_run, rescheduled_job, mock)
        runner.start()

    assert len(seq_steps) > n_jobs * steps_per_job
    normal_job_steps = [s for n, s in seq_steps if n != rescheduled_job.name]
    rescheduled_job_steps = [s for n, s in seq_steps if n == rescheduled_job.name]

    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - 1

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'rescheduled', rescheduled_job_steps))) == 2
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1

    assert seq_steps.index(('job_4', 'completing')) > seq_steps.index(('job_5', 'completing'))


def test_reschedule_job_enforce_job_priority():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=1, result=i, steps=seq_steps, verbose=True) for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2,
                                     queueing_strategy=MultiThreadingJobRunner.QueueingStrategy.enforce_job_priority)
    def _run(self, mock):
        if mock.call_count < 3:
            runner.reschedule(self)
            return
        return DEFAULT  # ensures that the wrapped function is called after the side effect

    rescheduled_job = jobs[4]
    with patch.object(rescheduled_job, "_run", wraps=rescheduled_job._run) as mock:
        mock.side_effect = ft.partial(_run, rescheduled_job, mock)
        runner.start()

    assert len(seq_steps) > n_jobs * steps_per_job
    normal_job_steps = [s for n, s in seq_steps if n != rescheduled_job.name]
    rescheduled_job_steps = [s for n, s in seq_steps if n == rescheduled_job.name]

    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - 1

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == 3
    assert len(list(filter(lambda s: s == 'rescheduled', rescheduled_job_steps))) == 2
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1

    assert seq_steps.index(('job_4', 'completing')) < seq_steps.index(('job_5', 'completing'))


@pytest.mark.no_ci
def test_reschedule_job_high_parallelism():
    seq_steps = []
    n_jobs = 600
    jobs = [DummyJob(name=f"job_{i}", duration_secs=random.randint(150, 250)/10, steps=seq_steps, verbose=True) for i in range(n_jobs)]

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=200, delay_secs=0.1,
                                     queueing_strategy=MultiThreadingJobRunner.QueueingStrategy.enforce_job_priority)

    def _run(self, mock):
        if mock.call_count < 3:
            runner.reschedule(self)
            return
        return DEFAULT  # ensures that the wrapped function is called after the side effect

    rescheduled_jobs = [j for i, j in enumerate(jobs) if i % 17 == 0]
    for job in rescheduled_jobs:
        mock = patch.object(job, "_run", wraps=job._run).start()
        mock.side_effect = ft.partial(_run, job, mock)
    runner.start()

    rescheduled_job_names = [j.name for j in rescheduled_jobs]
    normal_job_steps = [s for n, s in seq_steps if n not in rescheduled_job_names]
    rescheduled_job_steps = [s for n, s in seq_steps if n in rescheduled_job_names]

    for state in ['created', 'starting', 'running', 'completing', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s == state, normal_job_steps))) == n_jobs - len(rescheduled_job_names)

    assert len(list(filter(lambda s: s == 'created', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'starting', rescheduled_job_steps))) == 3 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'running', rescheduled_job_steps))) == 3 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'rescheduled', rescheduled_job_steps))) == 2 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'completing', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'stopping', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)
    assert len(list(filter(lambda s: s == 'stopped', rescheduled_job_steps))) == 1 * len(rescheduled_job_names)

