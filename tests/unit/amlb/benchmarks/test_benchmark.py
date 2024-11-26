from amlb import Benchmark, SetupMode, resources
import os

from amlb.defaults import default_dirs
from amlb.utils import config_load
from amlb.utils import Namespace as ns
from tests.conftest import tmp_output_directory


def test_benchmark(tmp_path) -> None:
    config_default = config_load(
        os.path.join(default_dirs.root_dir, "resources", "config.yaml")
    )
    config_default_dirs = default_dirs
    # allowing config override from user_dir: useful to define custom benchmarks and frameworks for example.
    config_user = ns()
    # config listing properties set by command line
    config_args = ns.parse(
        {"results.global_save": False},
        output_dir=str(tmp_output_directory),
        script=os.path.basename(__file__),
        run_mode="local",
        parallel_jobs=1,
        sid="pytest.session",
        exit_on_error=True,
        test_server=False,
        tag=None,
        command="pytest invocation",
    )
    config_args = ns({k: v for k, v in config_args if v is not None})
    # merging all configuration files and saving to the global variable
    resources.from_configs(
        config_default, config_default_dirs, config_user, config_args
    )
    benchmark = Benchmark(
        framework_name="constantpredictor",
        benchmark_name="test",
        constraint_name="test",
        job_history=None,
    )
    benchmark.setup(SetupMode.force)
    results = benchmark.run()
    assert len(results) == 6
    assert not results["result"].isna().any()
