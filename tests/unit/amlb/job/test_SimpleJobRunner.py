import functools as ft
from unittest.mock import patch, DEFAULT

import pytest

from amlb.job import SimpleJobRunner
from amlb.utils import Timeout

from dummy import DummyJob

steps_per_job = 6


@pytest.mark.slow
def test_run_multiple_jobs():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", result=i, steps=seq_steps, verbose=True)
            for i in range(n_jobs)]
    assert len(seq_steps) == n_jobs
    assert all(step == 'created' for _, step in seq_steps)

    runner = SimpleJobRunner(jobs)
    results = runner.start()
    print(results)

    assert len(results) == n_jobs
    assert [r.result for r in results] == list(range(n_jobs))

    assert len(seq_steps) == n_jobs * steps_per_job
    run_steps = seq_steps[n_jobs:]  # ignoring the created step
    run_steps_per_job = steps_per_job - 1
    for i in range(n_jobs):
        job_steps = run_steps[run_steps_per_job*i : run_steps_per_job*(i+1)]
        assert all(job == f"job_{i}" for job, _ in job_steps)
        assert ['starting', 'running', 'completing', 'stopping', 'stopped'] == [step for _, step in job_steps]


@pytest.mark.slow
def test_stop_runner_during_job_run():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, result=i, steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    runner = SimpleJobRunner(jobs)
    with Timeout(timeout_secs=2, on_timeout=runner.stop):
        results = runner.start()
    print(results)

    assert len(results) < n_jobs
    assert len(seq_steps) < n_jobs * steps_per_job
    cancelled_jobs = [j for j, s in seq_steps if s == 'cancelling']
    assert len(cancelled_jobs) > 1
    first_cancelled = cancelled_jobs[0]
    first_cancelled_idx = int(first_cancelled.split('_')[1])
    last_result = results[-1]
    assert last_result.name == first_cancelled
    assert last_result.result is None
    assert last_result.duration > 0
    assert cancelled_jobs == [f"job_{i}" for i in range(first_cancelled_idx, n_jobs)]

    for state in ['created', 'stopping', 'stopped']:
        assert len(list(filter(lambda s: s[1] == state, seq_steps))) == n_jobs
    assert len(list(filter(lambda s: s[1] == 'starting', seq_steps))) == len(results)
    assert len(list(filter(lambda s: s[1] == 'completing', seq_steps))) + len(cancelled_jobs) == n_jobs


@pytest.mark.slow
def test_reschedule_job():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, result=i, steps=seq_steps, verbose=True)
            for i in range(n_jobs)]

    runner = SimpleJobRunner(jobs)

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

    assert seq_steps.index(('job_4', 'completing')) < seq_steps.index(('job_5', 'completing'))

