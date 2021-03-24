"""
**job** module handles all the job running logic:

- consistent exception handling and logging
- currently 2 job runners are implemented:
  - SimpleJobRunner runs the jobs sequentially.
  - ParallelJobRunner queues the jobs and run them in a dedicated thread
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum, auto
import logging
import platform
import queue
import signal
import threading
import time

from .utils import Namespace, Timer, InterruptTimeout, is_main_thread, raise_in_thread, signal_handler

log = logging.getLogger(__name__)


class State(Enum):
    created = auto()
    cancelled = auto()
    running = auto()
    rescheduled = auto()
    stopping = auto()
    stopped = auto()


class JobError(Exception):
    pass


class InvalidStateError(JobError):
    pass


class CancelledError(JobError):
    pass


class Job:

    def __init__(self, name="", timeout_secs=None, priority=None, raise_exceptions=False):
        """

        :param name:
        :param timeout_secs:
        :param priority:
        :param raise_exceptions: bool (default=False)
            If True, log and raise any Exception that caused a job failure.
            If False, only log the exception.
        """
        self.name = name
        self.timeout = timeout_secs
        self.priority = priority
        self.state = State.created
        self.thread_id = None
        self.raise_exceptions = raise_exceptions

    def start(self):
        try:
            start_msg = "Starting job {}.".format(self.name)
            self.thread_id = threading.current_thread().ident
            if self.state == State.stopping:
                self.state = State.cancelled
                raise CancelledError("Job was cancelled.")
            elif self.state != State.created:
                self.state = State.cancelled
                raise InvalidStateError("Job can't be started from state `{}`.".format(self.state))
            log.info("\n%s\n%s", '-'*len(start_msg), start_msg)
            self.state = State.running
            self._prepare()

            interruption_sequence = [
                dict(sig=None),
                # first trying sig=None to avoid propagation of the interruption error: this way we can collect the timeout in the result
                dict(sig=signal.SIGINT if is_main_thread() else signal.SIGTERM),
                # if main thread, try a graceful interruption.
            ]
            if platform.system() != "Windows":
                interruption_sequence.append(dict(sig=signal.SIGQUIT))
                interruption_sequence.append(dict(sig=signal.SIGKILL))

            with Timer() as t:
                with InterruptTimeout(
                        self.timeout,
                        interruptions=interruption_sequence,
                        wait_retry_secs=60
                ):  # escalates every minute if the previous interruption was ineffective
                    result = self._run()
            log.info("Job %s executed in %.3f seconds.", self.name, t.duration)
            log.debug("Job %s returned: %s", self.name, result)
            return result, t.duration
        except Exception as e:
            log.exception("Job `%s` failed with error: %s", self.name, str(e))
            if self.raise_exceptions:
                raise
            return None, -1

    def stop(self):
        try:
            self.state = State.stopping
            self._stop()
            return 0
        except Exception as e:
            log.exception("Job `%s` did not stop gracefully: %s", self.name, str(e))
            return 1

    def done(self):
        try:
            if self.state in [State.rescheduled, State.running, State.stopping]:
                self._on_done()
        except Exception as e:
            log.exception("Job `%s` completion failed with error: %s", self.name, str(e))
        finally:
            if self.state is State.rescheduled:
                self.reset()
            else:
                self.reset(State.stopped)

    def reschedule(self):
        self.state = State.rescheduled
        self.thread_id = None

    def reset(self, state=State.created):
        self.state = state
        self.thread_id = None

    def _prepare(self):
        """hood to execute pre-run logic: this is executed in the same thread as the run logic."""
        pass

    def _run(self):
        """jobs should implement their run logic in this method."""
        pass

    def _stop(self):
        if self.thread_id is not None:
            raise_in_thread(self.thread_id, CancelledError)

    def _on_done(self):
        """hook to execute logic after job completion in a thread-safe way as this is executed in the main thread."""
        pass


class JobRunner:

    def __init__(self, jobs):
        self.jobs = jobs
        self.results = []
        self.state = State.created
        self._queue = None
        self._last_priority = 0

    def start(self):
        if self.state != State.created:
            raise InvalidStateError(self.state)
        self._init_queue()
        self.state = State.running
        with Timer() as t:
            self._run()
        self.state = State.stopped
        log.info("All jobs executed in %.3f seconds.", t.duration)
        return self.results

    def stop(self):
        if self.state in [State.stopping, State.stopped]:
            return
        self.state = State.stopping
        if self._queue:
            self._queue.put((-1, None))
        return self._stop()

    def stop_if_complete(self):
        if 0 < len(self.jobs) == len(self.results):
            self.stop()

    def put(self, job, priority=None):
        if priority is None:
            if job.priority is None:
                job.priority = self._last_priority = self._last_priority+1
        else:
            job.priority = priority
        if self._queue:
            self._queue.put((job.priority, job))
        else:
            log.warning("Ignoring job `%s`. Runner state: `%s`", job.name, self.state)

    def _init_queue(self):
        self._queue = queue.PriorityQueue(maxsize=len(self.jobs))
        for job in self.jobs:
            self.put(job)

    def __iter__(self):
        return self

    def __next__(self):
        if self._queue is None:
            return
        _, job = self._queue.get()
        self._queue.task_done()
        if job is None:
            self._queue = None
        return job

    def _run(self):
        pass

    def _stop(self):
        for job in self.jobs:
            job.stop()


class SimpleJobRunner(JobRunner):

    def _run(self):
        for job in self:
            if job is None or self.state == State.stopping:
                break
            result, duration = job.start()
            if job.state is not State.rescheduled:
                self.results.append(Namespace(name=job.name, result=result, duration=duration))
            job.done()
            self.stop_if_complete()


class MultiThreadingJobRunner(JobRunner):

    def __init__(self, jobs, parallel_jobs=1, done_async=True, delay_secs=0, use_daemons=False):
        super().__init__(jobs)
        self.parallel_jobs = parallel_jobs
        self._done_async = done_async
        self._delay = delay_secs  # short sleep between enqueued jobs to make console more readable
        self._daemons = use_daemons

    def _run(self):
        signal_handler(signal.SIGINT, self.stop)
        signal_handler(signal.SIGTERM, self.stop)
        q = queue.Queue(1)  # block one at a time to allow prioritization of rescheduled jobs

        def worker():
            while True:
                job = q.get()
                if job is None or self.state == State.stopping:
                    q.task_done()
                    break
                try:
                    result, duration = job.start()
                    if job.state is not State.rescheduled:
                        self.results.append(Namespace(name=job.name, result=result, duration=duration))
                    if self._done_async:
                        job.done()
                    self.stop_if_complete()
                finally:
                    q.task_done()

        threads = []
        for thread in range(self.parallel_jobs):
            thread = threading.Thread(target=worker, daemon=self._daemons)
            thread.start()
            threads.append(thread)

        try:
            for job in self:
                if self.state == State.stopping:
                    break
                print(f"queueing job {job.name}")
                q.put(job)
                print(f"queued job {job.name}")
                if self._delay > 0:
                    time.sleep(self._delay)
            q.join()
        finally:
            q.maxsize = self.parallel_jobs  # resize to ensure that all workers can get a None job
            for _ in range(self.parallel_jobs):
                try:
                    q.put_nowait(None)     # stopping workers
                except:
                    pass
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

