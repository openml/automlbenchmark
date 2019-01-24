"""
**job** module handles all the job running logic:

- consistent exception handling and logging
- currently 2 job runners are implemented:
  - SimpleJobRunner runs the jobs sequentially.
  - ParallelJobRunner queues the jobs and run them in a dedicated thread
"""
from enum import Enum
import logging
import queue
import random as rnd
import threading
import time

from .utils import Namespace

log = logging.getLogger(__name__)


class Job:

    def __init__(self, name=""):
        self.name = name

    def run(self):
        try:
            beep = time.time()
            result = self._run()
            duration = time.time() - beep
            log.info("Job %s executed in %.3f seconds", self.name, duration)
            log.debug("Job %s returned: %s", self.name, result)
            return result, duration
        except Exception as e:
            log.error("Job `%s` failed with error %s", self.name, str(e))
            log.exception(e)
            return None, -1

    def done(self):
        try:
            self._on_done()
        except Exception as e:
            log.error("Job `%s` completion failed with error %s", self.name, str(e))
            log.exception(e)

    def _run(self):
        """jobs should implement their run logic in this method"""
        pass

    def _on_done(self):
        """hook to execute logic after job completion in a thread-safe way as this is executed in the main thread"""
        pass


class JobRunner:

    State = Enum('State', 'created running stopping stopped')

    def __init__(self, jobs):
        self.jobs = jobs
        self.results = []
        self.state = JobRunner.State.created

    def start(self):
        start_time = time.time()
        self.state = JobRunner.State.running
        self._run()
        self.state = JobRunner.State.stopped
        stop_time = time.time()
        log.info("All jobs executed in %.3f seconds", stop_time-start_time)
        return self.results

    def stop(self):
        self.state = JobRunner.State.stopping

    def _run(self):
        pass


class SimpleJobRunner(JobRunner):

    def _run(self):
        for job in self.jobs:
            if self.state == JobRunner.State.stopping:
                break
            result, duration = job.run()
            self.results.append(Namespace(name=job.name, result=result, duration=duration))
            job.done()


class ParallelJobRunner(JobRunner):

    def __init__(self, jobs, parallel_jobs, done_async=False, delay_secs=0):
        super().__init__(jobs)
        self.parallel_jobs = parallel_jobs
        self._done_async = done_async
        self._delay = delay_secs  # short sleep between enqueued jobs to make console more readable

    def _run(self):
        q = queue.Queue()

        def worker():
            while True:
                job = q.get()
                if job is None or self.state == JobRunner.State.stopping:
                    break
                result, duration = job.run()
                self.results.append(Namespace(name=job.name, result=result, duration=duration))
                if self._done_async:
                    job.done()
                q.task_done()

        threads = []
        for thread in range(self.parallel_jobs):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        try:
            for job in self.jobs:
                q.put(job)     # TODO: timeout
                if self._delay > 0:
                    time.sleep(self._delay)
                if self.state == JobRunner.State.stopping:
                    break
            q.join()
        finally:
            for _ in range(self.parallel_jobs):
                q.put(None)     # stopping workers
            for thread in threads:
                thread.join()
            if not self._done_async:
                for job in self.jobs:
                    job.done()

