import os

import pytest
from amlb import Resources, resources
from amlb.defaults import default_dirs
from amlb.utils import Namespace, config_load


@pytest.fixture
def load_default_resources(tmp_path):
    config_default = config_load(
        os.path.join(default_dirs.root_dir, "resources", "config.yaml")
    )
    config_default_dirs = default_dirs
    config_test = Namespace(
        frameworks=Namespace(
            definition_file=[
                "{root}/resources/frameworks.yaml",
                "{root}/tests/resources/frameworks.yaml",
            ]
        )
    )
    # allowing config override from user_dir: useful to define custom benchmarks and frameworks for example.
    config_user = Namespace()
    # config listing properties set by command line
    config_args = Namespace.parse(
        {"results.global_save": False},
        output_dir=str(tmp_path),
        script=os.path.basename(__file__),
        run_mode="local",
        parallel_jobs=1,
        sid="pytest.session",
        exit_on_error=True,
        test_server=False,
        tag=None,
        command="pytest invocation",
    )
    config_args = Namespace({k: v for k, v in config_args if v is not None})
    # merging all configuration files and saving to the global variable
    resources.from_configs(
        config_default, config_test, config_default_dirs, config_user, config_args
    )


@pytest.fixture
def simple_resource():
    return Resources(
        Namespace(
            input_dir="my_input",
            output_dir="my_output",
            user_dir="my_user_dir",
            root_dir="my_root_dir",
            docker=Namespace(
                image_defaults=Namespace(
                    author="author",
                    image=None,
                    tag=None,
                )
            ),
            frameworks=Namespace(
                root_module="frameworks",
                definition_file=[],
                allow_duplicates=False,
                tags=[],
            ),
        )
    )
