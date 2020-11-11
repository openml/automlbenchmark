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

from .utils import Namespace, Timer, InterruptTimeout, raise_in_thread, signal_handler

log = logging.getLogger(__name__)


class State(Enum):
    created = auto()
    cancelled = auto()
    running = auto()
    rescheduled = auto()
    stopping = auto()
    stopped = auto()


class InvalidStateError(Exception):
    pass


class CancelledError(Exception):
    pass


class Job:

    def __init__(self, name="", timeout_secs=None, priority=None):
        self.name = name
        self.timeout = timeout_secs
        self.priority = priority
        self.state = State.created
        self.thread_id = None

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
            with Timer() as t:
                # don't propagate interruption error here (sig=None) so that we can collect the timeout in the result
                with InterruptTimeout(self.timeout, sig=None):
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
            if self.state in [State.rescheduled, State.running, State.stopping]:
                self._on_done()
        except Exception as e:
            log.error("Job `%s` completion failed with error: %s", self.name, str(e))
            log.exception(e)
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

    def _run(self):
        """jobs should implement their run logic in this method"""
        pass

    def _stop(self):
        if self.thread_id is not None:
            raise_in_thread(self.thread_id, CancelledError)

    def _on_done(self):
        """hook to execute logic after job completion in a thread-safe way as this is executed in the main thread"""
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
        self.state = State.stopping
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
        self._queue.put((job.priority, job))

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
        self._queue.task_done();
        if job is None:
            self._queue = None
            return
        return job

    def _run(self):
        pass

    def _stop(self):
        for job in self.jobs:
            job.stop()


class SimpleJobRunner(JobRunner):

    def _run(self):
        for job in self:
            if self.state == State.stopping:
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
        q = queue.Queue()

        def worker():
            while True:
                job = q.get()
                if job is None or self.state == State.stopping:
                    q.task_done()
                    break
                result, duration = job.start()
                if job.state is not State.rescheduled:
                    self.results.append(Namespace(name=job.name, result=result, duration=duration))
                if self._done_async:
                    job.done()
                self.stop_if_complete()
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
                q.put(job)     # TODO: timeout
                if self._delay > 0:
                    time.sleep(self._delay)
            q.join()
        finally:
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

