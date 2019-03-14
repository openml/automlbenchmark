"""
This 'system' first determines the best value for `max_features` for the Random Forest,
by trying up to 10 (uniformly distributed) values of 1..sqrt(p)...p. (p = number of features of the data).
It produces predictions based on a model trained with all of the data for the best found `max_features` value.
"""
import logging
import math

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import impute
from automl.results import save_predictions_to_file
from automl.utils import Timer, translate_dict

log = logging.getLogger(__name__)


def pick_values_uniform(start: int, end: int, length: int):
    d = (end - start) / (length - 1)
    uniform_floats = [start + i * d for i in range(length)]
    return list(set([int(f) for f in uniform_floats]))


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Tuned Random Forest (sklearn) ****\n")

    is_classification = config.type == 'classification'

    # Impute any missing data (can test using -t 146606)
    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y, dataset.test.y

    log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, config.cores))
    log.warning("We completely ignore the requirement to stay within the time limit.")

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor
    metric = dict(auc='roc_auc', logloss='neg_log_loss')[config.metric]

    n_features = X_train.shape[1]
    default_value = max(1, int(math.sqrt(n_features)))
    below_default = pick_values_uniform(start=1, end=default_value, length=6)[:-1]
    above_default = pick_values_uniform(start=default_value, end=n_features, length=11 - len(below_default))[1:]
    max_features_values = below_default + [default_value] + above_default

    log.info("Evaluating multiple values for `max_features`.")
    log.warning("TODO: Incorporate imputation in fold evaluations.")
    max_feature_scores = []
    for max_features_value in max_features_values:
        rf = estimator(n_jobs=config.cores,
                       random_state=config.seed,
                       max_features=max_features_value,
                       **config.framework_params)
        score = cross_val_score(rf, X_train, y_train, scoring=metric, cv=5)
        max_feature_scores.append((score, max_features_value))

    best_score, best_max_features_value = max(max_feature_scores)
    rf = estimator(n_jobs=config.cores,
                   random_state=config.seed,
                   max_features=best_max_features_value,
                   **config.framework_params)

    log.info("Training final model with `max_features={}`.".format(best_max_features_value))
    with Timer() as training:
        rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False)

    return dict(
        models_count=len(rf),
        training_duration=training.duration
    )
