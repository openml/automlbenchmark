"""
amlb entrypoint package.
"""

from .logger import app_logger as log
from .errors import AutoMLError
from .resources import Resources
from .benchmark import Benchmark, SetupMode
from .docker import DockerBenchmark
from .singularity import SingularityBenchmark
from .aws import AWSBenchmark, AWSRemoteBenchmark
from .results import TaskResult

__all__ = (
    log,
    AutoMLError,
    Resources,
    Benchmark,
    DockerBenchmark,
    SingularityBenchmark,
    AWSBenchmark,
    AWSRemoteBenchmark,
    SetupMode,
    TaskResult,
)
