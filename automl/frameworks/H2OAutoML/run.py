import time

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.utils.multiclass import type_of_target

import h2o
from h2o.automl import H2OAutoML

from automl.utils import save_predictions_to_file


def run(dataset, config):
    # Mapping of benchmark metrics to H2O metrics
    if config.metric == "acc":
        h2o_metric = "mean_per_class_error"
    elif config.metric == "auc":
        h2o_metric = "AUC"
    elif config.metric == "logloss":
        h2o_metric = "logloss"
    else:
        # TO DO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        print('Performance metric, {}, not supported, using AUTO.'.format(config.metric))
        # performance_metric = None

    print('Starting H2O cluster')
    print('cores {}, memory {}mb'.format(config.cores, config.max_mem_size))
    h2o.init(nthreads=config.cores, max_mem_size=str(config.max_mem_size) + 'M')

    print('Loading data.')
    # Load train as an H2O Frame, but test as a Pandas DataFrame
    train = h2o.import_file(dataset.train)
    test = h2o.import_file(dataset.test)

    print('Running H2O AutoML with a maximum time of {}s on {} cores, optimizing {}.'
          .format(config.max_runtime, config.cores, h2o_metric))

    print('Running model on task.')
    start_time = time.time()

    aml = H2OAutoML(max_runtime_secs=config.max_runtime, sort_metric=h2o_metric)
    aml.train(y=train.ncol-1, training_frame=train)
    actual_runtime_min = (time.time() - start_time)/60.0
    print('Requested training time (minutes): ' + str((config.max_runtime/60.0)))
    print('Actual training time (minutes): ' + str(actual_runtime_min))

    print('Predicting on the test set.')
    predictions = aml.predict(test).as_data_frame()

    preview_size = 20
    # truth_df = test[:, -1].as_data_frame(header=False)
    truth_df = test[:, dataset.y].as_data_frame(header=False)
    predictions.insert(0, 'truth', truth_df)
    print("Predictions sample:")
    print(predictions.head(preview_size))
    print()

    y_pred = predictions.iloc[:, 1]
    y_true = predictions.iloc[:, 0]
    print("test target type: "+type_of_target(y_true))
    accuracy = accuracy_score(y_true, y_pred)
    print('Optimization was towards metric, but following score is always accuracy:')
    print("Accuracy: "+str(accuracy))
    print()

    # TO DO: See if we can use the h2o-sklearn wrappers here instead
    class_predictions = y_pred.values
    class_probabilities = predictions.iloc[:, 2:].values

    # TO DO: Change this to roc_curve, auc
    if type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OBinomialModelMetrics:
        y_scores = predictions.iloc[:, -1]
        auc = roc_auc_score(y_true=y_true, y_score=y_scores)
        print("AUC: " + str(auc))
    elif type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OMultinomialModelMetrics:
        logloss = log_loss(y_true=y_true, y_pred=class_probabilities)
        print("Log Loss: " + str(logloss))

    dest_file = "{folder}/predictions_h2o_{task_id}_{fold}".format(folder=config.folder, task_id=config.task_id, fold=config.fold)
    save_predictions_to_file(class_probabilities, class_predictions.astype(str), dest_file)

