import logging
import math
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from .data import Dataset, Feature
from .resources import Resources
from .utils import memoize

log = logging.getLogger(__name__)


class Results:

    def __init__(self, task_name: str, fold: int, resources: Resources):
        self.task = task_name
        self.fold = fold
        self.resources = resources

    @memoize
    def get_result(self, framework):
        predictions_file = os.path.join(self.resources.config.output_dir, "{framework}_{task}_{fold}.pred").format(
            framework=framework.lower(),
            task=self.task,
            fold=self.fold
        )
        log.info("Loading predictions from %s", predictions_file)
        df = pd.read_csv(predictions_file)
        log.debug("Predictions preview:\n %s\n", df.head(10).to_string())
        if df.shape[1] > 2:
            return ClassificationResult(df)
        else:
            return RegressionResult(df)


class Result:

    def __init__(self, predictions_df):
        self.df = predictions_df
        self.truth = self.df.iloc[:, -1].values
        self.predictions = self.df.iloc[:, -2].values
        self.target = None
        self.type = None

    def acc(self):
        return accuracy_score(self.truth, self.predictions)

    def logloss(self):
        return log_loss(self.truth, self.predictions)

    def mse(self):
        return mean_squared_error(self.truth, self.predictions)

    def rmse(self):
        return math.sqrt(self.mse())

    def evaluate(self, metric):
        if hasattr(self, metric):
            return getattr(self, metric)()
        raise ValueError("Metric {metric} is not supported for {type}".format(metric=metric, type=self.type))


class ClassificationResult(Result):

    def __init__(self, predictions_df):
        super().__init__(predictions_df)
        self.classes = self.df.columns[:-2].values
        self.probabilities = self.df.iloc[:, :-2].values.astype(float)
        self.target = Feature(0, 'class', 'categorical', self.classes)
        self.type = 'binomial' if len(self.classes) == 2 else 'multinomial'
        self.truth = self._autoencode(self.truth)
        self.predictions = self._autoencode(self.predictions)

    def auc(self):
        if self.type != 'binomial':
            raise ValueError("AUC metric is only supported for binary classification: {}".format(self.classes))
        return roc_auc_score(self.truth, self.probabilities[:, 1])

    def logloss(self):
        # truth_enc = self.target.label_binarizer.transform(self.truth)
        return log_loss(self.truth, self.probabilities)

    def _autoencode(self, vec):
        needs_encoding = isinstance(vec[0], str) and not vec[0].isdigit()
        return self.target.encode(vec) if needs_encoding else vec


class RegressionResult(Result):

    def __init__(self, predictions_df):
        super().__init__(predictions_df)
        self.truth = self.truth.astype(float)
        self.target = Feature(0, 'target', 'real')
        self.type = 'regression'


def save_predictions_to_file(dataset: Dataset, output_file, class_probabilities=None, class_predictions=None, class_truth=None, encode_classes=False):
    """ Save class probabilities and predicted labels to file in csv format.

    :param dataset:
    :param task:
    :param class_probabilities:
    :param class_predictions:
    :param class_truth:
    :param encode_labels:
    :return: None
    """
    file_path = output_file if re.search(r'\.pred$', output_file) else output_file + '.pred'
    log.info("Saving predictions to %s", file_path)
    df = pd.DataFrame(class_probabilities, columns=dataset.target.values)
    df = df.assign(predictions=class_predictions if not encode_classes else dataset.target.encode(class_predictions))
    truth = class_truth if class_truth is not None else dataset.test.y
    df = df.assign(truth=truth if not encode_classes else dataset.target.encode(truth))
    log.info("Predictions preview:\n %s\n", df.head(20).to_string())
    df.to_csv(file_path, index=False)
    log.debug("Predictions successfully saved to %s", file_path)
