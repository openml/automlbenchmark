import os
from .run import run

def setup():
    load_log = os.popen("./setup.sh").read()

__all__ = (setup, run)