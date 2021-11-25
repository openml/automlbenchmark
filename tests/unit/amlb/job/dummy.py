import time

from amlb.job import Job, State as JobState


class DummyJob(Job):

    def __init__(self, name="", timeout_secs=None, priority=None, raise_exceptions=False,
                 duration_secs=0, result=None, steps=None, verbose=False):
        self.steps = [] if steps is None else steps
        self._verbose = verbose
        self._duration_secs = duration_secs
        self._result = result
        self.ext = {}
        super().__init__(name, timeout_secs, priority, raise_exceptions)

    def _add_step(self, step):
        self.steps.append((self.name, step))
        if self._verbose:
            print(f"job {self.name} in step {step}")

    def _run(self):
        super()._run()
        if self._duration_secs > 0:
            time.sleep(self._duration_secs)
        return self._result

    def _on_state(self, state: JobState):
        self._add_step(state.name)

