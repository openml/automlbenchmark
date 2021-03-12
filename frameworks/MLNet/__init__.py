from amlb.utils import call_script_in_same_dir
import os

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)



def run(dataset, config):
    from .exec import run
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DOTNET_INSTALL_DIR = os.path.join(dir_path, '.dotnet')
    MLNET = os.path.join(DOTNET_INSTALL_DIR, 'mlnet')
    return run(dataset, config, mlnet=MLNET)
