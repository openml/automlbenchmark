"""
amlb entrypoint package.
"""

from amlb.logger import app_logger as log
from amlb.errors import AutoMLError
from amlb.resources import Resources
from amlb.benchmark import Benchmark, SetupMode
from amlb.docker import DockerBenchmark
from amlb.singularity import SingularityBenchmark
from amlb.aws import AWSBenchmark, AWSRemoteBenchmark
from amlb.results import TaskResult

__all__ = [
    "log",
    "AutoMLError",
    "Resources",
    "Benchmark",
    "DockerBenchmark",
    "SingularityBenchmark",
    "AWSBenchmark",
    "AWSRemoteBenchmark",
    "SetupMode",
    "TaskResult",
]
