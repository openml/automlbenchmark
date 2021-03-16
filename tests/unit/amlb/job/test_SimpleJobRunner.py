from unittest.mock import patch, DEFAULT

from amlb.job import SimpleJobRunner
from amlb.utils import Timeout

from dummy import DummyJob

steps_per_job = 5


def test_run_multiple_jobs():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", result=i, steps=seq_steps, verbose=True) for i in range(n_jobs)]
    assert len(seq_steps) == n_jobs
    assert all(step == 'created' for _, step in seq_steps)
    seq_steps.clear()

    runner = SimpleJobRunner(jobs)
    results = runner.start()
    print(results)
    assert len(seq_steps) == n_jobs * steps_per_job
    for i in range(n_jobs):
        job_steps = seq_steps[steps_per_job*i : steps_per_job*(i+1)]
        assert all(job == f"job_{i}" for job, _ in job_steps)
        assert ['starting', 'running', 'completing', 'stopping', 'stopped'] == [step for _, step in job_steps]
    assert len(results) == n_jobs
    assert [r.result for r in results] == list(range(n_jobs))


def test_stop_runner_during_job_run():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, result=i, steps=seq_steps, verbose=True) for i in range(n_jobs)]
    seq_steps.clear()

    runner = SimpleJobRunner(jobs)
    with Timeout(timeout_secs=2, on_timeout=runner.stop):
        results = runner.start()
    print(results)
    assert len(seq_steps) < n_jobs * steps_per_job
    # assert results only for started jobs


def test_reschedule_job():
    seq_steps = []
    n_jobs = 10
    jobs = [DummyJob(name=f"job_{i}", duration_secs=0.5, steps=seq_steps, verbose=True) for i in range(n_jobs)]
    seq_steps.clear()

    runner = SimpleJobRunner(jobs)
    rescheduled_job = jobs[4]
    with patch.object(rescheduled_job, "_run", wraps=rescheduled_job._run) as mock:
        def _run():
            if mock.call_count < 3:
                runner.reschedule(rescheduled_job)
            return DEFAULT # ensures that the wrapped function is called after the side effect
        mock.side_effect = _run
        runner.start()

