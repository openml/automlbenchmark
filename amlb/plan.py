from dataclasses import dataclass
import logging
from typing import List

from .benchmark import Benchmark

log = logging.getLogger(__name__)


@dataclass
class RunTask:
    framework: str
    benchmark: str
    constraint: str
    tasks: [str]
    folds: [int]


class ExecutionPlan:

    def __init__(self, benchmark_cls, plan: List[RunTask]):
        self.benchmark_cls = benchmark_cls
        self.plan = plan

    def run(self):
        main_instance = Benchmark(None, None, None)
        bench_instances = [self.benchmark_cls(rt.framework, rt.benchmark, rt.constraint) for rt in self.plan]
        jobs = []
        for instance, rt in zip(bench_instances, self.plan):
            jobs.extend(instance.create_jobs(tasks=rt.tasks, folds=rt.folds))
        return main_instance.run_jobs(jobs)
