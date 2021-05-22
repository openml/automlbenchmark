"""
amlb entrypoint package.
"""

from .benchmark import SetupMode
from .logger import app_logger as log
from .errors import AutoMLError
from .resources import Resources
from .runners import LocalBenchmark, AWSBenchmark, DockerBenchmark, SingularityBenchmark
from .results import TaskResult
from .__version__ import __version__

__all__ = [
    "log",
    "AutoMLError",
    "Resources",
    "LocalBenchmark",
    "AWSBenchmark",
    "DockerBenchmark",
    "SingularityBenchmark",
    "SetupMode",
    "TaskResult",
    "__version__"
]
