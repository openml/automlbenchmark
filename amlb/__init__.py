"""
amlb entrypoint package.
"""

from .__version__ import __version__
from .benchmark import Benchmark, SetupMode
from .errors import AutoMLError
from .logger import app_logger as log
from .resources import Resources
from .results import TaskResult
from .runners import AWSBenchmark, DockerBenchmark, SingularityBenchmark

__all__ = [
    "AWSBenchmark",
    "AutoMLError",
    "Benchmark",
    "DockerBenchmark",
    "Resources",
    "SetupMode",
    "SingularityBenchmark",
    "TaskResult",
    "__version__",
    "log",
]
