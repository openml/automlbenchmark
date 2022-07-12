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
import pprint
import queue
import signal
import threading

from .utils import Namespace, Timer, ThreadSafeCounter, InterruptTimeout, is_main_thread, raise_in_thread, signal_handler

log = logging.getLogger(__name__)


class State(Enum):
    created = auto()
    starting = auto()
    running = auto()
    completing = auto()
    rescheduling = auto()
    cancelling = auto()
    stopping = auto()
    stopped = auto()


class JobError(Exception):
    pass


class InvalidStateError(JobError):
    pass


class CancelledError(JobError):
    pass


class Job:

    state_machine = [
        (None,                  [State.created]),
        (State.created,         [State.starting, State.cancelling]),
        (State.starting,        [State.running, State.rescheduling, State.cancelling]),
        (State.running,         [State.completing, State.rescheduling, State.cancelling]),
        (State.completing,      [State.stopping]),
        (State.rescheduling,    [State.starting, State.stopping]),
        (State.cancelling,      [State.stopping, State.stopped]),
        (State.stopping,        [State.stopped]),
        (State.stopped,         None)
    ]

    @classmethod
    def is_state_transition_ok(cls, old_state: State, new_state: State):
        allowed = next((head for tail, head in cls.state_machine if tail == old_state), None)
        return allowed and new_state in allowed

    printer = pprint.PrettyPrinter(indent=2, compact=True)

    def __init__(self, name="", timeout_secs=None, priority=None, raise_on_failure=False):
        """

        :param name:
        :param timeout_secs:
        :param priority:
        :param raise_on_failure: bool (default=False)
            If True, log and raise any Exception that caused a job failure.
            If False, only log the exception, and produce a None result.
        """
        self.name = name
        self.timeout = timeout_secs
        self.priority = priority
        self.state = None
        self.thread_id = None
        self.raise_on_failure = raise_on_failure
        self.set_state(State.created)

    def start(self):
        t = None
        try:
            self.thread_id = threading.current_thread().ident
            if self.set_state(State.starting):
                start_msg = f"Starting job {self.name}."
                log.info("\n%s\n%s", '-'*len(start_msg), start_msg)
                self._setup()

            if not self.is_state_transition_ok(self.state, State.running):
                raise CancelledError(f"Job `{self.name}` was interrupted during setup")

            interruption_sequence = [
                # first trying sig=None to avoid propagation of the interruption error:
                # this way we can collect the timeout in the result
                dict(sig=None),
                # if main thread, try a graceful interruption.
                dict(sig=signal.SIGINT if is_main_thread() else signal.SIGTERM),
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
                    if self.set_state(State.running):
                        result = self._run()
            log.info("Job `%s` executed in %.3f seconds.", self.name, t.duration)
            log.debug("Job `%s` returned: %s", self.name, result)
            return Namespace(name=self.name, result=result, duration=t.duration)
        except Exception as e:
            log.exception("Job `%s` failed with error: %s", self.name, str(e))
            if self.raise_on_failure:
                raise
            return Namespace(name=self.name, result=None, duration=t.duration if t else -1)

    def reschedule(self):
        """Called  when the runner plans to restart the job at a later time."""
        if self.is_state_transition_ok(self.state, State.rescheduling):
            self.reset(State.rescheduling)

    def done(self):
        """Happy ending scenario."""
        if self.is_state_transition_ok(self.state, State.completing):
            self.set_state(State.completing)
            self.set_state(State.stopping)
            self.reset(State.stopped)

    def stop(self):
        """Sad ending scenario."""
        if self.is_state_transition_ok(self.state, State.cancelling):
            try:
                if self.set_state(State.cancelling):
                    self._cancel()
                self.set_state(State.stopping)
                return 0
            except Exception as e:
                log.exception("Job `%s` did not stop gracefully: %s", self.name, str(e))
                return 1
            finally:
                self.reset(State.stopped)

    def set_state(self, state: State):
        assert self.is_state_transition_ok(self.state, state), f"Illegal job transition from state {self.state} to {state}"
        old_state = self.state
        self.state = state
        log.debug("Changing job `%s` from state %s to %s.", self.name, old_state, state)
        skip_default = False
        try:
            skip_default = bool(self._on_state(state))
        except Exception as e:
            log.exception("Error when handling state change to %s for job `%s`: %s", state, self.name, str(e))
        return not skip_default

    def reset(self, state=State.created):
        self.thread_id = None
        self.set_state(state)

    def _setup(self):
        """hook to execute pre-run logic: this is executed in the same thread as the run logic."""
        pass

    def _run(self):
        """jobs should implement their run logic in this method."""
        pass

    def _cancel(self):
        """hook executed on the job once it's being cancelled by the runner:
        this is called only once on the job, and only if it didn't complete"""
        if self.thread_id is not None:
            raise_in_thread(self.thread_id, CancelledError(f"Job `{self.name}` was interrupted."))

    def _on_state(self, state: State):
        pass

    def __str__(self):
        return Job.printer.pformat(self.__dict__)


class JobRunner:

    state_machine = [
        (None,              [State.created]),
        (State.created,     [State.starting, State.stopping]),
        (State.starting,    [State.running, State.stopping]),
        (State.running,     [State.stopping]),
        (State.stopping,    [State.stopped]),
        (State.stopped,     None)
    ]

    @classmethod
    def is_state_transition_ok(cls, old_state: State, new_state: State):
        allowed = next((head for tail, head in cls.state_machine if tail == old_state), None)
        return allowed and new_state in allowed

    def __init__(self, jobs):
        self.jobs = jobs
        self.results = []
        self.state = None
        self._queue = None
        self._last_priority = 0
        self.set_state(State.created)

    def start(self):
        t = None
        try:
            if self.set_state(State.starting):
                self._setup()
            with Timer() as t:
                if self.set_state(State.running):
                    self._run()
        finally:
            self.stop()
            if t is not None:
                log.info("All jobs executed in %.3f seconds.", t.duration)
        return self.results

    def stop(self):
        if self.is_state_transition_ok(self.state, State.stopping):
            try:
                if self.set_state(State.stopping):
                    return self._stop()
            finally:
                self.set_state(State.stopped)

    def stop_if_complete(self):
        if 0 < len(self.jobs) == len(self.results):
            self.stop()

    def put(self, job, priority=None):
        if self.state in [State.stopping, State.stopped]:
            return
        if priority is None:
            if job.priority is None:
                job.priority = self._last_priority = self._last_priority+1
        else:
            job.priority = priority
        if self._queue:
            self._queue.put((job.priority, job))
        else:
            log.warning("Ignoring job `%s`. Runner state: `%s`", job.name, self.state)

    def reschedule(self, job, priority=None):
        if self.state not in [State.running]:
            return
        job.reschedule()
        if job.state is State.rescheduling:
            self.put(job, priority)

    def set_state(self, state: State):
        assert self.is_state_transition_ok(self.state, state), f"Illegal job runner transition from state {self.state} to {state}"
        old_state = self.state
        self.state = state
        log.debug("Changing job runner from state %s to %s.", old_state, state)
        skip_default = False
        try:
            skip_default = bool(self._on_state(state))
        except Exception as e:
            log.exception("Error when handling state change to %s for job runner: %s", state, str(e))
        return not skip_default

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

    def _setup(self):
        self._queue = queue.PriorityQueue(maxsize=len(self.jobs))
        for job in self.jobs:
            self.put(job)

    def _run(self):
        pass

    def _stop(self):
        if self._queue:
            self._queue.put((-1, None))
        jobs = self.jobs.copy()
        self.jobs.clear()
        for job in jobs:
            job.stop()

    def _on_state(self, state: State):
        pass


class SimpleJobRunner(JobRunner):

    def __init__(self, jobs):
        super().__init__(jobs)
        self._interrupt = threading.Event()

    def _run(self):
        for job in self:
            if job is None or self._interrupt.is_set():
                break
            result = job.start()
            if job.state is State.rescheduling:
                self.reschedule(job)
            else:
                self.results.append(result)
                job.done()
            self.stop_if_complete()

    def _on_state(self, state: State):
        if state is State.stopping:
            self._interrupt.set()


class MultiThreadingJobRunner(JobRunner):

    class QueueingStrategy:
        keep_queue_full = 0
        enforce_job_priority = 1

    def __init__(self, jobs, parallel_jobs=1, done_async=True, delay_secs=0,
                 queueing_strategy: QueueingStrategy = QueueingStrategy.keep_queue_full,
                 use_daemons=False):
        super().__init__(jobs)
        self.parallel_jobs = parallel_jobs
        self._done_async = done_async
        self._delay = delay_secs  # short sleep between enqueued jobs to make console more readable
        self._daemons = use_daemons
        self._queueing_strategy = queueing_strategy
        self._interrupt = threading.Event()

    def _run(self):
        q = queue.Queue()
        available_workers = ThreadSafeCounter(self.parallel_jobs)
        wc = threading.Condition()

        def worker():
            while True:
                job = q.get()
                available_workers.dec()
                try:
                    if job is None or self._interrupt.is_set():
                        break
                    result = job.start()
                    if job.state is State.rescheduling:
                        self.reschedule(job)
                    else:
                        self.results.append(result)
                        if self._done_async:
                            job.done()
                    self.stop_if_complete()
                finally:
                    q.task_done()
                    available_workers.inc()
                    with wc:
                        wc.notify_all()

        threads = []
        for t in range(self.parallel_jobs):
            thread = threading.Thread(target=worker, daemon=self._daemons)
            thread.start()
            threads.append(thread)

        try:
            while not self._interrupt.is_set():
                if self._queueing_strategy == MultiThreadingJobRunner.QueueingStrategy.enforce_job_priority:
                    with wc:
                        wc.wait_for(lambda: available_workers.value > 0)
                job = next(self, None)
                if self._interrupt.is_set() or job is None:
                    break
                q.put(job)
                if self._delay > 0:
                    self._interrupt.wait(self._delay)
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

    def _on_state(self, state: State):
        if state is State.stopping:
            self._interrupt.set()


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

