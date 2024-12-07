import logging
import os
import tempfile as tmp

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer
from sapientml import SapientML
from sklearn.preprocessing import OneHotEncoder

os.environ["JOBLIB_TEMP_FOLDER"] = tmp.gettempdir()
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


log = logging.getLogger(__name__)


def run(dataset, config):
    import re

    import pandas as pd

    log.info("\n**** Sapientml ****\n")

    is_classification = config.type == "classification"

    train_path, test_path = dataset.train.path, dataset.test.path
    target_col = dataset.target.name

    # Read parquet using pandas
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    # Removing unwanted sybols from column names (exception case)
    X_train.columns = [re.sub("[^A-Za-z0-9_.]+", "", col) for col in X_train.columns]
    X_test.columns = [re.sub("[^A-Za-z0-9_.]+", "", col) for col in X_test.columns]
    target_col = re.sub("[^A-Za-z0-9_.]+", "", target_col)

    # y_train and y_test
    y_test = X_test[target_col].reset_index(drop=True)

    # Drop target col from X_test
    X_test.drop([target_col], axis=1, inplace=True)

    # Sapientml
    output_dir = (
        config.output_dir + "/" + "outputs" + "/" + config.name + "/" + str(config.fold)
    )
    predictor = SapientML(
        [target_col], task_type="classification" if is_classification else "regression"
    )

    # Fit the model
    with Timer() as training:
        predictor.fit(X_train, output_dir=output_dir)
    log.info(f"Finished fit in {training.duration}s.")

    # predict
    with Timer() as predict:
        predictions = predictor.predict(X_test)
    log.info(f"Finished predict in {predict.duration}s.")

    if is_classification:
        predictions[target_col] = predictions[target_col].astype(str)
        predictions[target_col] = predictions[target_col].str.lower()
        predictions[target_col] = predictions[target_col].str.strip()
        y_test = y_test.to_frame()
        y_test[target_col] = y_test[target_col].astype(str)
        y_test[target_col] = y_test[target_col].str.lower()
        y_test[target_col] = y_test[target_col].str.strip()

    if is_classification:
        probabilities = OneHotEncoder(handle_unknown="ignore").fit_transform(
            predictions.to_numpy()
        )
        probabilities = pd.DataFrame(
            probabilities.toarray(), columns=dataset.target.classes
        )

        return result(
            output_file=config.output_predictions_file,
            predictions=predictions,
            truth=y_test,
            probabilities=probabilities,
            training_duration=training.duration,
            predict_duration=predict.duration,
        )
    else:
        return result(
            output_file=config.output_predictions_file,
            predictions=predictions,
            truth=y_test,
            training_duration=training.duration,
            predict_duration=predict.duration,
        )


if __name__ == "__main__":
    call_run(run)
