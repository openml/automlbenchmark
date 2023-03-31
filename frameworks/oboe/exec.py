import logging
import os
import sys

from sklearn.model_selection import StratifiedKFold
import numpy as np

sys.path.append("{}/lib/oboe/automl".format(os.path.realpath(os.path.dirname(__file__))))
from oboe import AutoLearner

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def kfold_fit_validate(self, x_train, y_train, n_folds, random_state=None):
    """Performs k-fold cross validation on a training dataset. Note that this is the function used to fill entries
    of the error matrix.
    Args:
        x_train (np.ndarray): Features of the training dataset.
        y_train (np.ndarray): Labels of the training dataset.
        n_folds (int):        Number of folds to use for cross validation.
    Returns:
        float: Mean of k-fold cross validation error.
        np.ndarray: Predictions on the training dataset from cross validation.
    """
    y_predicted = np.empty(y_train.shape)
    cv_errors = np.empty(n_folds)
    kf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(kf.split(x_train, y_train)):
        x_tr = x_train[train_idx, :]
        y_tr = y_train[train_idx]
        x_te = x_train[test_idx, :]
        y_te = y_train[test_idx]
        model = self.instantiate()
        if len(np.unique(y_tr)) > 1:
            model.fit(x_tr, y_tr)
            y_predicted[test_idx] = np.expand_dims(model.predict(x_te), axis=1)
        else:
            y_predicted[test_idx] = y_tr[0]
        cv_errors[i] = self.error(y_te, y_predicted[test_idx])
    self.cv_error = cv_errors.mean()
    self.cv_predictions = y_predicted
    self.sampled = True
    if self.verbose:
        print("{} {} complete.".format(self.algorithm, self.hyperparameters))
    return cv_errors, y_predicted


def run(dataset, config):
    log.info(f"\n**** Applying monkey patch ****\n")
    from oboe.model import Model
    Model.kfold_fit_validate = kfold_fit_validate

    log.info(f"\n**** Oboe [{config.framework_version}] ****\n")
    is_classification = config.type == 'classification'
    if not is_classification:
        # regression currently fails (as of 26.02.2019: still under development state by oboe team)
        raise ValueError('Regression is not yet supported (under development).')

    X_train = dataset.train.X
    y_train = dataset.train.y

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_cores = config.framework_params.get('_n_cores', config.cores)

    log.info('Running oboe with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, n_cores))
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    aml = AutoLearner(p_type='classification' if is_classification else 'regression',
                      n_cores=n_cores,
                      runtime_limit=config.max_runtime_seconds,
                      **training_params)
    aml.error_matrix = aml.error_matrix.to_numpy()

    aml_models = lambda: [aml.ensemble, *aml.ensemble.base_learners] if len(aml.ensemble.base_learners) > 0 else []

    with Timer() as training:
        try:
            aml.fit(X_train, y_train)
        except IndexError as e:
            if len(aml_models()) == 0:  # incorrect handling of some IndexError in oboe if ensemble is empty
                raise ValueError("Oboe could not produce any model in the requested time.")
            raise e

    log.info('Predicting on the test set.')
    X_test = dataset.test.X
    y_test = dataset.test.y
    with Timer() as predict:
        predictions = aml.predict(X_test)
    predictions = predictions.reshape(len(X_test))

    if is_classification:
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`
    else:
        probabilities = None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(aml_models()),
                  training_duration=training.duration,
                  predict_duration=predict.duration)


if __name__ == '__main__':
    call_run(run)
