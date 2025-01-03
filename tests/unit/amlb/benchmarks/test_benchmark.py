from pathlib import Path
from subprocess import SubprocessError

import pytest

from amlb import Benchmark, SetupMode, resources, DockerBenchmark, SingularityBenchmark
from amlb.job import JobError
from amlb.utils import Namespace


def test_benchmark(load_default_resources) -> None:
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


@pytest.mark.parametrize(
    ("framework_name", "tag", "expected"),
    [
        ("constantpredictor", "latest", "automlbenchmark/constantpredictor:stable"),
        ("flaml", "2023Q2", "automlbenchmark/flaml:1.2.4"),
        ("autosklearn", "stable", "automlbenchmark/autosklearn:stable"),
    ],
)
def test_docker_image_name(
    framework_name, tag, expected, load_default_resources
) -> None:
    framework_def, _ = resources.get().framework_definition(
        framework_name,
        tag=tag,
    )
    # The docker image name is based entirely on configuration (i.e. configured branch, not checked out branch).
    # If there is a different branch checked out locally and you run the benchmark,
    # you will get a prompt to verify you want to build anyway
    result = DockerBenchmark.image_name(
        framework_def,
    )
    assert result == expected


@pytest.mark.parametrize("branch", ["master", "foo"])
def test_docker_image_name_uses_branch(branch, mocker, load_default_resources) -> None:
    mocker.patch(
        "amlb.runners.container.rget",
        return_value=Namespace(project_info=Namespace(branch=branch)),
    )
    framework_def, _ = resources.get().framework_definition("constantpredictor")
    result = DockerBenchmark.image_name(framework_def)
    assert result == f"automlbenchmark/constantpredictor:stable-{branch}"


@pytest.mark.parametrize("label", [None, "master", "foo"])
def test_docker_image_name_uses_label(label, mocker, load_default_resources) -> None:
    branch = "bar-branch"
    mocker.patch(
        "amlb.runners.container.rget",
        return_value=Namespace(project_info=Namespace(branch=branch)),
    )

    framework_def, _ = resources.get().framework_definition("constantpredictor")
    result = DockerBenchmark.image_name(framework_def, label=label)

    used_label = label or branch
    assert result == f"automlbenchmark/constantpredictor:stable-{used_label}"


@pytest.mark.parametrize(
    ("framework_name", "tag", "expected"),
    [
        ("constantpredictor", "latest", "constantpredictor_stable"),
        ("flaml", "2023Q2", "flaml_1.2.4"),
        ("autosklearn", "stable", "autosklearn_stable"),
    ],
)
def test_singularity_image_name(
    framework_name, tag, expected, load_default_resources
) -> None:
    benchmark = SingularityBenchmark(
        framework_name=f"{framework_name}:{tag}",
        benchmark_name="test",
        constraint_name="test",
    )
    image_path = benchmark._container_image_name(
        as_docker_image=False,
    )
    image_name = Path(image_path).stem
    assert image_name == expected


@pytest.mark.parametrize(
    ("framework_name", "tag", "expected"),
    [
        ("constantpredictor", "latest", "automlbenchmark/constantpredictor:stable"),
        ("flaml", "2023Q2", "automlbenchmark/flaml:1.2.4"),
        ("autosklearn", "stable", "automlbenchmark/autosklearn:stable"),
    ],
)
def test_singularity_image_name_as_docker(
    framework_name, tag, expected, load_default_resources
) -> None:
    benchmark = SingularityBenchmark(
        framework_name=f"{framework_name}:{tag}",
        benchmark_name="test",
        constraint_name="test",
    )
    result = benchmark._container_image_name(
        as_docker_image=True,
    )
    assert result == expected


def test_benchmark_setup_errors_if_framework_does_not_install(
    load_default_resources,
) -> None:
    benchmark = Benchmark(
        framework_name="setup_fail",
        benchmark_name="test",
        constraint_name="test",
        job_history=None,
    )

    with pytest.raises(JobError) as exc_info:
        benchmark.setup(SetupMode.force)
    assert "setup" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, SubprocessError)
    assert "command_that_fails" in exc_info.value.__cause__.stderr
