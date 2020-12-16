import os

import pandas as pd
import pytest

from amlb.uploads import _load_predictions

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')

iris_complete = f"{res}/iris/"


def test__load_predictions_loads_all_files():
    prediction_files = [os.path.join(iris_complete, f"{i}/predictions.csv") for i in range(10)]
    task = None
    predictions = _load_predictions(None, prediction_files)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (150, 5)
    assert list(predictions.columns) == ["iris-setosa", "iris-versicolor", "iris-virginica", "predictions", "truth"]


def test__load_predictions_reorders():
    pass
