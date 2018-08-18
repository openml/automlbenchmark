import sys
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import accuracy_score
sys.path.append('/bench/common')
import common_code


if __name__ == '__main__':

    # Some assumptions are made (TO DO: this should be generalized in the future)
    # Binary classification (need to update y_pred code to extend to multiclass & regression)
    # The data is loaded from an ARFF file and the response is designated as a categorical
    # The response column is the last column and called 'class' (all OpenML datasets have this)

    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    # Harcoded for testing
    # TO DO: un-hardcode this and grab args from input
    runtime_seconds = 30
    number_cores = -1
    performance_metric = "AUC"

    # TO DO: Add a mapping of performance metrics from benchmark name -> H2O name
    # e.g. "acc" -> "mean_per_class_error"

    print('Starting H2O cluster')
    # TO DO: Pass in a memory size as an argument to use here
    #h2o.init(ncores=number_cores, max_mem_size=16). #ncores not working if set to -1 (need to check)
    h2o.init()

    print('Loading data.')
    # Load train as an H2O Frame, but test as a Pandas DataFrame
    train = h2o.upload_file(common_code.TRAIN_DATA_PATH)
    test = h2o.upload_file(common_code.TEST_DATA_PATH)
    y_test = test[:, -1].as_data_frame()

    print('Running H2O AutoML with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('Running model on task.')
    runtime_min = (int(runtime_seconds)/60)
    print('ignoring runtime.')
    #aml = H2OAutoML(max_runtime_secs=runtime_seconds, sort_metric=performance_metric)
    #aml = H2OAutoML(max_models=2). # For testing only
    aml = H2OAutoML(max_runtime_secs=runtime_seconds)
    aml.train(y=train.ncol-1, training_frame=train)
    y_pred_df = aml.predict(test).as_data_frame()
    # TO DO: this will only work for binary classification, need to extend to any task
    #y_pred = y_pred_df.iloc[:, -1]

    #if type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OBinomialModelMetrics:
    #    y_pred = y_pred_df.iloc[:, 0]
    #elif type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OMultinomialModelMetrics:
    #    y_pred = y_pred_df.iloc[:, 0]
    #else:
    #    print('The benchmarks do not yet support Regression tasks')

    y_classpred = y_pred_df.iloc[:, 0]
    print('Optimization was towards metric, but following score is always accuracy:')
    print("THIS_IS_A_DUMMY_TOKEN " + str(accuracy_score(y_test, y_classpred)))

    # TO DO: See if we can use the h2o-sklearn wrappers here instead
    class_predictions = y_pred_df.iloc[:, 0].values #TO DO: should this be dtype=object instead?
    class_probabilities = y_pred_df.iloc[:, 1:].values

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
