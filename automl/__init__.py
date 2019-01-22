"""
automl entrypoint package.
"""

from .logger import logger as log
from .resources import Resources
from .benchmark import Benchmark
from .docker import DockerBenchmark
from .aws import AWSBenchmark, AWSRemoteBenchmark
from .results import TaskResult

__all__ = (
    log,
    Resources,
    Benchmark,
    DockerBenchmark,
    AWSBenchmark,
    AWSRemoteBenchmark,
    TaskResult,
)

# TODO:
#   global:
#     README.md
#     update usage documentation
#     pydoc
#     unit tests
#  features:
#    AWS: reuse instances for faster startup, at least during a single benchmark, we could limit #instances = #parallel jobs
#    meta-benchmark? benchmark a subset of configured frameworks: runbenchmark.py frameworks.yaml test -m aws -p 4
#    timeouts (esp. for AWS, but also for all jobs in general: global timeout + job timeout?)
#    progress bar?? fancy useless stuff
#    visualizations for results.csv
#    search for todos in code
