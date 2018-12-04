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
#    AWS support (in progress)
#    group score files into global one
#    timeouts (esp. for AWS, but also for all jobs in general: global timeout + job timeout?)
#    progress bar?? fancy useless stuff
#    search for todos in code
