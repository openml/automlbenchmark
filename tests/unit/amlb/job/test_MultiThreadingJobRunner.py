from unittest.mock import patch, DEFAULT

from amlb.job import MultiThreadingJobRunner
from amlb.utils import Timeout

from dummy import DummyJob

steps_per_job = 5


def test_run_multiple_jobs():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", steps=seq_steps, verbose=True, duration_secs=0.1) for i in range(n_jobs)]
    assert len(seq_steps) == n_jobs
    assert all(step == 'created' for _, step in seq_steps)
    seq_steps.clear()

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3)
    runner.start()
    assert len(seq_steps) == n_jobs * steps_per_job


def test_run_multiple_jobs_with_delay():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", steps=seq_steps, verbose=True, duration_secs=1) for i in range(n_jobs)]
    seq_steps.clear()

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)
    runner.start()
    assert len(seq_steps) == n_jobs * steps_per_job


def test_stop_runner_during_job_run():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, steps=seq_steps, verbose=True) for i in range(n_jobs)]
    seq_steps.clear()

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=0.2)
    with Timeout(timeout_secs=1, on_timeout=runner.stop):
        runner.start()
    assert len(seq_steps) < n_jobs * steps_per_job


def test_reschedule_job():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=15, steps=seq_steps, verbose=True) for i in range(n_jobs)]
    seq_steps.clear()

    runner = MultiThreadingJobRunner(jobs, parallel_jobs=3, delay_secs=1)
    rescheduled_job = jobs[4]
    with patch.object(rescheduled_job, "_run", wraps=rescheduled_job._run) as mock:
        def _run():
            if mock.call_count < 3:
                runner.reschedule(rescheduled_job)
            return DEFAULT # ensures that the wrapped function is called after the side effect
        mock.side_effect = _run
        runner.start()

