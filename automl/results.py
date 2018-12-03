import logging
import math
import os

from numpy import NaN, sort
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

from .data import Dataset, Feature
from .resources import get as rget, config as rconfig
from .utils import Namespace, memoize, now_iso

log = logging.getLogger(__name__)

# TODO: reconsider organisation of output files:
#   predictions: add framework version to name, timestamp? group into subdirs?
#   gather scores in one single file?


class Scoreboard:

    @staticmethod
    def for_all():
        #todo: list benchmarks from resources and load all boards
        scores = []
        board = Scoreboard(scores)
        return board

    @staticmethod
    def for_benchmark(benchmark_name, framework_name=None):
        pass

    @staticmethod
    def for_task(task_name, framework_name=None):
        pass

    def __init__(self, scores, framework_name=None, benchmark_name=None, task_name=None, scores_dir=None):
        self.scores = scores
        self.framework_name = framework_name
        self.benchmark_name = benchmark_name
        self.task_name = task_name
        self.scores_dir = scores_dir if scores_dir else rconfig().scores_dir

    @memoize
    def as_data_frame(self, index=None):
        # index = index if index else ['task', 'framework', 'fold']
        df = self.scores if isinstance(self.scores, pd.DataFrame) \
            else pd.DataFrame.from_records([sc.as_dict() for sc in self.scores], index=index)
        index = index if index else []
        # fixed_cols = ['result', 'mode', 'version', 'utc']
        fixed_cols = ['task', 'framework', 'fold', 'result', 'mode', 'version', 'utc']
        fixed_cols = [col for col in fixed_cols if col not in index]
        dynamic_cols = [col for col in df.columns if col not in index and col not in fixed_cols]
        dynamic_cols.sort()
        df = df.reindex(columns=[]+fixed_cols+dynamic_cols)
        log.debug("scores columns: %s", df.columns)
        return df

    def save(self, append=False, data_frame=None):
        if data_frame is None:
            data_frame = self.as_data_frame()
        exists = os.path.isfile(self._score_file())
        new_format = False
        if exists:
            # todo: detect format change, i.e. data_frame columns are different or different order from existing file
            pass
        if new_format or (exists and not append):
            # todo: backup existing file, i.e. rename to {file_name}_{last_write_time}.ext
            pass
        new_file = not exists or not append or new_format
        is_default_index = data_frame.index.name is None and not any(data_frame.index.names)
        data_frame.to_csv(self._score_file(),
                          header=new_file,
                          index=not is_default_index,
                          mode='w' if new_file else 'a')

    def _score_file(self):
        if self.framework_name:
            if self.task_name:
                file_name = "{framework}_task_{task}.csv".format(framework=self.framework_name, task=self.task_name)
            elif self.benchmark_name:
                file_name = "{framework}_benchmark_{benchmark}.csv".format(framework=self.framework_name, benchmark=self.benchmark_name)
            else:
                file_name = "{framework}.csv".format(framework=self.framework_name)
        else:
            if self.task_name:
                file_name = "task_{task}.csv".format(task=self.task_name)
            elif self.benchmark_name:
                file_name = "benchmark_{benchmark}.csv".format(benchmark=self.benchmark_name)
            else:
                file_name = "all_results.csv"

        return os.path.join(self.scores_dir, file_name)


class TaskResult:

    @staticmethod
    def load_predictions(predictions_file):
        log.info("Loading predictions from %s", predictions_file)
        if os.path.isfile(predictions_file):
            df = pd.read_csv(predictions_file)
            log.debug("Predictions preview:\n %s\n", df.head(10).to_string())
            if df.shape[1] > 2:
                return ClassificationResult(df)
            else:
                return RegressionResult(df)
        else:
            log.warning("Predictions file {file} is missing: framework either failed or could not produce any prediction".format(
                file=predictions_file,
            ))
            return NoResult()

    @staticmethod
    def save_predictions(dataset: Dataset, predictions_file: str,
                         class_probabilities=None, class_predictions=None, class_truth=None,
                         class_probabilities_labels=None,
                         classes_are_encoded=False):
        """ Save class probabilities and predicted labels to file in csv format.

        :param dataset:
        :param predictions_file:
        :param class_probabilities:
        :param class_predictions:
        :param class_truth:
        :param class_probabilities_labels:
        :param classes_are_encoded:
        :return: None
        """
        log.info("Saving predictions to %s", predictions_file)
        prob_cols = class_probabilities_labels if class_probabilities_labels else dataset.target.label_encoder.classes
        df = pd.DataFrame(class_probabilities, columns=prob_cols)
        if class_probabilities_labels:
            df = df[sort(prob_cols)]  # reorder columns alphabetically: necessary to match label encoding

        predictions = class_predictions
        truth = class_truth if class_truth is not None else dataset.test.y
        if not encode_predictions_and_truth and classes_are_encoded:
            predictions = dataset.target.label_encoder.inverse_transform(class_predictions)
            truth = dataset.target.label_encoder.inverse_transform(truth)
        if encode_predictions_and_truth and not classes_are_encoded:
            predictions = dataset.target.label_encoder.transform(class_predictions)
            truth = dataset.target.label_encoder.transform(truth)

        df = df.assign(predictions=predictions)
        df = df.assign(truth=truth)
        log.info("Predictions preview:\n %s\n", df.head(20).to_string())
        df.to_csv(predictions_file, index=False)
        log.debug("Predictions successfully saved to %s", predictions_file)

    def __init__(self, task_name: str, fold: int, predictions_dir=None):
        self.task = task_name
        self.fold = fold
        self.predictions_dir = predictions_dir if predictions_dir else rconfig().predictions_dir

    @memoize
    def get_result(self, framework_name):
        return self.load_predictions(self._predictions_file(framework_name))

    def compute_scores(self, framework_name, metrics):
        framework_def, _ = rget().framework_definition(framework_name)
        scores = Namespace(
            framework=framework_name,
            version=framework_def.version,
            task=self.task,
            fold=self.fold,
            mode=rconfig().run_mode,    # fixme: at the end, we're always running in local mode!!!
            utc=now_iso()
        )
        result = self.get_result(framework_name)
        for metric in metrics:
            score = result.evaluate(metric)
            scores[metric] = score
        scores.result = scores[metrics[0]]
        log.info("metric scores: %s", scores)
        return scores

    def _predictions_file(self, framework_name):
        return os.path.join(self.predictions_dir, "{framework}_{task}_{fold}.csv").format(
            framework=framework_name.lower(),
            task=self.task,
            fold=self.fold
        )


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
        self.classes = self.df.columns[:-2].values.astype(str)
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
        needs_encoding = not encode_predictions_and_truth or (isinstance(vec[0], str) and not vec[0].isdigit())
        return self.target.label_encoder.transform(vec) if needs_encoding else vec


class RegressionResult(Result):

    def __init__(self, predictions_df):
        super().__init__(predictions_df)
        self.truth = self.truth.astype(float)
        self.target = Feature(0, 'target', 'real', is_target=True)
        self.type = 'regression'


encode_predictions_and_truth = False


def save_predictions_to_file(dataset: Dataset, output_file: str,
                             class_probabilities=None, class_predictions=None, class_truth=None,
                             class_probabilities_labels=None,
                             classes_are_encoded=False):
    TaskResult.save_predictions(dataset, predictions_file=output_file,
                                class_probabilities=class_probabilities, class_predictions=class_predictions, class_truth=class_truth,
                                class_probabilities_labels=class_probabilities_labels,
                                classes_are_encoded=classes_are_encoded)
