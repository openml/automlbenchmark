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
#    search for todos in code
