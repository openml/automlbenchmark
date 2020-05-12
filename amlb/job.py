"""
**job** module handles all the job running logic:

- consistent exception handling and logging
- currently 2 job runners are implemented:
  - SimpleJobRunner runs the jobs sequentially.
  - ParallelJobRunner queues the jobs and run them in a dedicated thread
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum, auto
import logging
import multiprocessing
import queue
import signal
import threading
import time

from .utils import Namespace, Timer, InterruptTimeout

log = logging.getLogger(__name__)


class State(Enum):
    created = auto()
    cancelled = auto()
    running = auto()
    stopping = auto()
    stopped = auto()


class InvalidStateError(Exception):
    pass


class CancelledError(Exception):
    pass


class Job:

    def __init__(self, name="", timeout_secs=None):
        self.name = name
        self.timeout = timeout_secs
        self.state = State.created

    def start(self):
        try:
            start_msg = "Starting job {}.".format(self.name)
            if self.state == State.stopping:
                self.state = State.cancelled
                raise CancelledError("Job was cancelled.")
            elif self.state != State.created:
                self.state = State.cancelled
                raise InvalidStateError("Job can't be started from state `{}`.".format(self.state))
            log.info("\n%s\n%s", '-'*len(start_msg), start_msg)
            self.state = State.running
            with Timer() as t:
                with InterruptTimeout(self.timeout):
                    result = self._run()
            log.info("Job %s executed in %.3f seconds.", self.name, t.duration)
            log.debug("Job %s returned: %s", self.name, result)
            return result, t.duration
        except Exception as e:
            log.error("Job `%s` failed with error: %s", self.name, str(e))
            log.exception(e)
            return None, -1

    def stop(self):
        try:
            self.state = State.stopping
            self._stop()
            return 0
        except Exception as e:
            log.exception(e)
            return 1

    def done(self):
        try:
            if self.state in [State.running, State.stopping]:
                self._on_done()
            else:
                log.warning("Job `%s` done with unexpected state: %s", self.name, self.state)
        except Exception as e:
            log.error("Job `%s` completion failed with error: %s", self.name, str(e))
            log.exception(e)
        finally:
            self.state = State.stopped

    def _run(self):
        """jobs should implement their run logic in this method"""
        pass

    def _stop(self):
        pass

    def _on_done(self):
        """hook to execute logic after job completion in a thread-safe way as this is executed in the main thread"""
        pass


class JobRunner:

    def __init__(self, jobs):
        self.jobs = jobs
        self.results = []
        self.state = State.created

    def start(self):
        if self.state != State.created:
            raise InvalidStateError(self.state)
        self.state = State.running
        with Timer() as t:
            self._run()
        self.state = State.stopped
        log.info("All jobs executed in %.3f seconds.", t.duration)
        return self.results

    def stop(self):
        self.state = State.stopping
        return self._stop()

    def _run(self):
        pass

    def _stop(self):
        for job in self.jobs:
            job.stop()


class SimpleJobRunner(JobRunner):

    def _run(self):
        for job in self.jobs:
            if self.state == State.stopping:
                break
            result, duration = job.start()
            self.results.append(Namespace(name=job.name, result=result, duration=duration))
            job.done()


class MultiThreadingJobRunner(JobRunner):

    def __init__(self, jobs, parallel_jobs=1, done_async=True, delay_secs=0, use_daemons=False):
        super().__init__(jobs)
        self.parallel_jobs = parallel_jobs
        self._done_async = done_async
        self._delay = delay_secs  # short sleep between enqueued jobs to make console more readable
        self._daemons = use_daemons

    def _run(self):
        q = queue.Queue()

        def worker():
            while True:
                job = q.get()
                if job is None or self.state == State.stopping:
                    q.task_done()
                    break
                result, duration = job.start()
                self.results.append(Namespace(name=job.name, result=result, duration=duration))
                if self._done_async:
                    job.done()
                q.task_done()

        threads = []
        for thread in range(self.parallel_jobs):
            thread = threading.Thread(target=worker, daemon=self._daemons)
            thread.start()
            threads.append(thread)

        # previous_handler = signal.signal(signal.SIGINT, self.stop)

        try:
            for job in self.jobs:
                if self.state == State.stopping:
                    break
                q.put(job)     # TODO: timeout
                if self._delay > 0:
                    time.sleep(self._delay)
            q.join()
        finally:
            # signal.signal(signal.SIGINT, previous_handler)
            for _ in range(self.parallel_jobs):
                q.put(None)     # stopping workers
            for thread in threads:
                thread.join()
            if not self._done_async:
                for job in self.jobs:
                    job.done()


class MultiProcessingJobRunner(JobRunner):
    pass


""" Experimental: trying to simplify multi-threading/processing"""


class ExecutorJobRunner(JobRunner):

    def __init__(self, pool_executor_class, jobs, parallel_jobs):
        super().__init__(jobs)
        self.pool_executor_class = pool_executor_class
        self.parallel_jobs = parallel_jobs

    def _run(self):
        def worker(job):
            result, duration = job.start()
            job.done()
            return Namespace(name=job.name, result=result, duration=duration)

        with self.pool_executor_class(max_workers=self.parallel_jobs) as executor:
            self.results.extend(executor.map(worker, self.jobs))
            # futures = []
            # for job in self.jobs:
            #     future = executor.submit(worker, job)
            #    # future.add_done_callback(lambda _: job.done())
            #     futures.append(future)
            # for future in as_completed(futures):
            #     self.results.append(future.result())


class ThreadPoolExecutorJobRunner(ExecutorJobRunner):
    def __init__(self, jobs, parallel_jobs):
        super().__init__(ThreadPoolExecutor, jobs, parallel_jobs)


class ProcessPoolExecutorJobRunner(ExecutorJobRunner):
    def __init__(self, jobs, parallel_jobs):
        super().__init__(ProcessPoolExecutor, jobs, parallel_jobs)

