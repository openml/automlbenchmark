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
#    AWS: docker support, currently AWS support is completely benchmark/framework agnostic and uses a generic image
#         but using docker inside AWS could improve reproducibility, although it requires building+publishing+maintaining multiple images
#         Note that current generic docker support allows running multiple docker instances in parallel, so we could
#    meta-benchmark? benchmark a subset of configured frameworks: runbenchmark.py frameworks.yaml test -m aws -p 4
#    small tool/command to rebuild scores from predictions
#    timeouts (esp. for AWS, but also for all jobs in general: global timeout + job timeout?)
#    progress bar?? fancy useless stuff
#    search for todos in code
