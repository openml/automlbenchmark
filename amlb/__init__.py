"""
amlb entrypoint package.
"""

from .logger import app_logger as log
from .errors import AutoMLError
from .resources import Resources
from .benchmark import Benchmark, SetupMode
from .runners import AWSBenchmark, DockerBenchmark, SingularityBenchmark
from .results import TaskResult
from .__version__ import __version__

__all__ = [
    "log",
    "AutoMLError",
    "Resources",
    "Benchmark",
    "DockerBenchmark",
    "SingularityBenchmark",
    "AWSBenchmark",
    "SetupMode",
    "TaskResult",
    "__version__"
]
