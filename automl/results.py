import logging
import math
import os
import re

from numpy import NaN, ndarray, sort
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

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
        predictions_file = os.path.join(self.resources.config.predictions_dir, "{framework}_{task}_{fold}.csv").format(
            framework=framework.lower(),
            task=self.task,
            fold=self.fold
        )
        log.info("Loading predictions from %s", predictions_file)
        if os.path.isfile(predictions_file):
            df = pd.read_csv(predictions_file)
            log.debug("Predictions preview:\n %s\n", df.head(10).to_string())
            if df.shape[1] > 2:
                return ClassificationResult(df)
            else:
                return RegressionResult(df)
        else:
            log.warning("Predictions file {file} is missing: {framework} either failed or could not produce any prediction".format(
                file=predictions_file,
                framework=framework
            ))
            return NoResult()

    def compute_scores(self, framework, metrics):
        scores = dict(
            framework=framework,
            task=self.task,
            fold=self.fold,
        )
        result = self.get_result(framework)
        for metric in metrics:
            score = result.evaluate(metric)
            scores[metric] = score

        log.info("metric scores: %s", scores)
        return scores


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

    def auc(self):
        return NaN

    def evaluate(self, metric):
        if hasattr(self, metric):
            return getattr(self, metric)()
        raise ValueError("Metric {metric} is not supported for {type}".format(metric=metric, type=self.type))


class NoResult(Result):

    def __init__(self):
        self.missing_result = 'NA'

    def acc(self):
        return self.missing_result

    def logloss(self):
        return self.missing_result

    def mse(self):
        return self.missing_result

    def rmse(self):
        return self.missing_result

    def auc(self):
        return self.missing_result


class ClassificationResult(Result):

    def __init__(self, predictions_df):
        super().__init__(predictions_df)
        self.classes = self.df.columns[:-2].values
        self.probabilities = self.df.iloc[:, :-2].values.astype(float)
        self.target = Feature(0, 'class', 'categorical', self.classes, is_target=True)
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
        self.target = Feature(0, 'target', 'real', is_target=True)
        self.type = 'regression'


def save_predictions_to_file(dataset: Dataset, output_file: str,
                             class_probabilities: ndarray=None, class_predictions: ndarray=None, class_truth: ndarray=None,
                             class_probabilities_labels=None,
                             encode_classes=False):
    """ Save class probabilities and predicted labels to file in csv format.

    :param dataset:
    :param task:
    :param class_probabilities:
    :param class_predictions:
    :param class_truth:
    :param encode_labels:
    :return: None
    """
    file_path = output_file if re.search(r'\.csv$', output_file) else output_file + '.csv'
    log.info("Saving predictions to %s", file_path)
    prob_cols = class_probabilities_labels if class_probabilities_labels else dataset.target.label_encoder.classes
    df = pd.DataFrame(class_probabilities, columns=prob_cols)
    if class_probabilities_labels:
        df = df[sort(prob_cols)]  # reorder columns alphabetically: necessary to match label encoding
    df = df.assign(predictions=class_predictions if not encode_classes else dataset.target.label_encoder.transform(class_predictions))
    truth = class_truth if class_truth is not None else dataset.test.y
    df = df.assign(truth=truth if not encode_classes else dataset.target.label_encoder.transform(truth))
    log.info("Predictions preview:\n %s\n", df.head(20).to_string())
    df.to_csv(file_path, index=False)
    log.debug("Predictions successfully saved to %s", file_path)


def scores_as_df(scores, index=None):
    index = index if index else ['task', 'fold', 'framework']
    return pd.DataFrame.from_records(scores, index=index)


def save_scores_to_file(scores, output_file: str):
    scores_df = scores if isinstance(scores, pd.DataFrame) else scores_as_df(scores)
    scores_df.to_csv(output_file)
