import pandas as pd
import pytest

from amlb.benchmarks.openml import (
    is_openml_benchmark,
    load_openml_task_as_definition,
    load_oml_benchmark,
)
from amlb.utils import Namespace


def test_is_openml_benchmark_valid():
    assert is_openml_benchmark("openml/s/271")
    assert is_openml_benchmark("test.openml/s/271")
    assert is_openml_benchmark("openml/t/41")


def test_is_openml_benchmark_bad_input():
    assert not is_openml_benchmark("openml/d/271"), "datasets are not valid targets"
    assert not is_openml_benchmark(
        "api.openml/t/271"
    ), "only openml and test.openml are valid prefixes"
    assert not is_openml_benchmark("validation"), "this is a file-defined benchmark"


@pytest.mark.parametrize(
    ("oml_task", "oml_dataset"),
    [
        (Namespace(id=42, dataset_id=21), Namespace(name="foo", description="bar")),
        (Namespace(id=20, dataset_id=8), Namespace(name="baz", description="coo")),
    ],
)
def test_load_openml_task(mocker, oml_task, oml_dataset):
    mocker.patch("openml.tasks.get_task", new=mocker.Mock(return_value=oml_task))
    mocker.patch(
        "openml.datasets.get_dataset", new=mocker.Mock(return_value=oml_dataset)
    )
    [task] = load_openml_task_as_definition("openml", oml_task.id)
    assert task.name == oml_dataset.name
    assert task.description == oml_dataset.description
    assert task.openml_task_id == oml_task.id
    assert task.id == f"openml.org/t/{oml_task.id}"


@pytest.mark.parametrize(
    ("oml_task", "oml_dataset"),
    [
        (Namespace(id=42, dataset_id=21), Namespace(name="foo", description="bar")),
        (Namespace(id=20, dataset_id=8), Namespace(name="baz", description="coo")),
    ],
)
@pytest.mark.parametrize("domain", ["openml", "test.openml"])
def test_load_oml_benchmark_from_task(mocker, oml_task, oml_dataset, domain):
    benchmark_name = f"{domain}/t/{oml_task.id}"

    mocker.patch(
        "openml.config"
    )  # Required to avoid load_oml_benchmark affecting global state
    mocker.patch("openml.tasks.get_task", new=mocker.Mock(return_value=oml_task))
    mocker.patch(
        "openml.datasets.get_dataset", new=mocker.Mock(return_value=oml_dataset)
    )

    benchmark, file, [task] = load_oml_benchmark(benchmark=benchmark_name)

    assert benchmark == benchmark_name
    assert file is None, "OpenML benchmark definitions do not reference files on disk"
    assert task.name == oml_dataset.name
    assert task.description == oml_dataset.description
    assert task.openml_task_id == oml_task.id
    assert task.id == f"{domain}.org/t/{oml_task.id}"


@pytest.mark.parametrize("domain", ["openml", "test.openml"])
def test_load_oml_benchmark_from_suite(mocker, domain):
    mock_suite = Namespace(id=271, data=[21], tasks=[42])
    mock_datasets = pd.DataFrame(
        [[21, "foo", "bar"]], columns=["did", "name", "description"]
    )
    benchmark_name = f"{domain}/s/{mock_suite.id}"

    mocker.patch(
        "openml.config"
    )  # required to avoid load_oml_benchmark affecting global state
    mocker.patch("openml.study.get_suite", new=mocker.Mock(return_value=mock_suite))
    mocker.patch(
        "openml.datasets.list_datasets", new=mocker.Mock(return_value=mock_datasets)
    )

    benchmark, file, [task] = load_oml_benchmark(benchmark=benchmark_name)

    assert benchmark == benchmark_name
    assert file is None, "OpenML benchmark definitions do not reference files on disk"
    assert task.name == "foo"
    assert task.description == f"{domain}/d/21"
    assert task.openml_task_id == 42
    assert task.id == f"{domain}.org/t/42"
