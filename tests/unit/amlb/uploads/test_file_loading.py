import os

import openml
import pandas as pd

from amlb.uploads import _load_predictions, _load_fold

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')

iris_complete = f"{res}/iris/"


def test__load_fold_loads_prediction_file():
    iris_task = openml.tasks.get_task(59)
    predictions = _load_fold(iris_complete, 0, iris_task)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (15, 8)
    assert set(predictions.columns) == {"repeat", "fold", "index", "iris-setosa", "iris-versicolor", "iris-virginica", "predictions", "truth"}


def test__load_fold_attaches_correct_index_information():
    iris_task = openml.tasks.get_task(59)
    predictions = _load_fold(iris_complete, 0, iris_task)
    assert list(predictions["index"]) == [43, 14, 37, 23, 10, 99, 87, 97, 62, 92, 119, 111, 144, 116, 125]
    assert all(predictions["fold"] == 0)
    predictions = _load_fold(iris_complete, 1, iris_task)
    assert list(predictions["index"]) == [49, 15, 47,  0, 44, 76, 89, 58, 54, 53, 149, 114, 112, 115, 139]
    assert all(predictions["fold"] == 1)


def test__load_predictions_loads_all_files():
    predictions = _load_predictions(iris_complete)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (150, 8)
    assert set(predictions.columns) == {"repeat", "fold", "index", "iris-setosa", "iris-versicolor", "iris-virginica", "predictions", "truth"}
    assert set(predictions["fold"]) == set(range(10))
