import os
from collections import OrderedDict
import pathlib

import openml
import pandas as pd
import pytest

from amlb.uploads import _load_predictions, _load_fold, _get_flow, _load_task_data, \
    _extract_and_format_hyperparameter_configuration, _upload_results

here = pathlib.Path(__file__).parent.absolute()
resources = here / 'resources'

iris_constant = resources / "iris_constant"
iris_h2o = resources / "iris_h2o"


@pytest.fixture(scope="module")
def with_oml_test_server():
    openml.config.start_using_configuration_for_example()
    yield
    openml.config.stop_using_configuration_for_example()


@pytest.mark.use_web
@pytest.mark.use_disk
def test__load_fold_loads_prediction_file(with_oml_test_server):
    iris_task = openml.tasks.get_task(1196)
    predictions = _load_fold(iris_constant, 0, iris_task)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (15, 8)
    assert set(predictions.columns) == {"repeat", "fold", "index", "iris-setosa",
                                        "iris-versicolor", "iris-virginica",
                                        "predictions", "truth"}


@pytest.mark.use_web
@pytest.mark.use_disk
def test__load_fold_attaches_correct_index_information(with_oml_test_server):
    iris_task = openml.tasks.get_task(1196)
    predictions = _load_fold(iris_constant, 0, iris_task)
    assert list(predictions["index"]) == [43, 14, 37, 23, 10, 99, 87, 97, 62, 92, 119,
                                          111, 144, 116, 125]
    assert all(predictions["fold"] == 0)
    predictions = _load_fold(iris_constant, 1, iris_task)
    assert list(predictions["index"]) == [49, 15, 47, 0, 44, 76, 89, 58, 54, 53, 149,
                                          114, 112, 115, 139]
    assert all(predictions["fold"] == 1)


@pytest.mark.use_disk
def test__load_predictions_loads_all_files():
    predictions = _load_predictions(iris_constant)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (150, 8)
    assert set(predictions.columns) == {"repeat", "fold", "index", "iris-setosa",
                                        "iris-versicolor", "iris-virginica",
                                        "predictions", "truth"}
    assert set(predictions["fold"]) == set(range(10))


@pytest.mark.use_disk
def test__extract_flow_hyperparameter_configuration():
    metadata = _load_task_data(iris_constant)
    flow = _get_flow(metadata, sync_with_server=False)
    parameters = _extract_and_format_hyperparameter_configuration(metadata, flow)

    expected_parameters = [
        OrderedDict([("oml:name", "max_runtime_seconds"), ("oml:value", 3600),
                     ("oml:component", flow.id)]),
        OrderedDict([("oml:name", "max_mem_size_mb"), ("oml:value", 2465),
                     ("oml:component", flow.id)]),
        OrderedDict(
            [("oml:name", "cores"), ("oml:value", 8), ("oml:component", flow.id)]),
        OrderedDict([("oml:name", "seed"), ("oml:value", 1362575752),
                     ("oml:component", flow.id)]),
    ]
    assert parameters == expected_parameters


@pytest.mark.openml_upload
@pytest.mark.use_web
@pytest.mark.use_disk
def test__upload_results_constant(with_oml_test_server):
    run = _upload_results(iris_constant)
    assert isinstance(run.id, int)


@pytest.mark.openml_upload
@pytest.mark.use_web
@pytest.mark.use_disk
def test__upload_results_h2o(with_oml_test_server):
    run = _upload_results(iris_h2o)
    assert isinstance(run.id, int)


@pytest.mark.skip(
    "I can't think of a meaningful test that doesn't just verify that openml-python works as expected."
)
def test__get_flow(with_oml_test_server):
    metadata = _load_task_data(iris_constant)
    flow = _get_flow(metadata)
