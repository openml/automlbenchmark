import sys
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
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

    # Mapping of benchmark metrics to H2O metrics
    if performance_metric == "acc":
        performance_metric = "mean_per_class_error"
    elif performance_metric == "auc":
        performance_metric = "AUC"
    else:
        # TO DO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        print('Performance metric, {}, not supported.'.format(performance_metric))    

    # Harcoded for testing
    # TO DO: un-hardcode this and grab args from input
    runtime_seconds = 30
    number_cores = -1
    performance_metric = "AUC"


    print('Starting H2O cluster')
    # TO DO: Pass in a memory size as an argument to use here
    #h2o.init(nthreads=number_cores, max_mem_size=16). #ncores not working if set to -1 (need to check)
    h2o.init(nthreads=number_cores)

    print('Loading data.')
    # Load train as an H2O Frame, but test as a Pandas DataFrame
    train = h2o.upload_file(common_code.TRAIN_DATA_PATH)
    test = h2o.upload_file(common_code.TEST_DATA_PATH)
    y_test = test[:, -1].as_data_frame()

    print('Running H2O AutoML with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('Running model on task.')
    print('ignoring performance_metric (always optimizes AUC)')
    #aml = H2OAutoML(max_runtime_secs=runtime_seconds, sort_metric=performance_metric) #Add this
    aml = H2OAutoML(max_runtime_secs=runtime_seconds, sort_metric=performance_metric)
    aml.train(y=train.ncol-1, training_frame=train)

    print('Predicting on the test set.')
    y_pred_df = aml.predict(test).as_data_frame()

    if type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OBinomialModelMetrics:
        y_scores = y_pred_df.iloc[:, -1]
        auc = roc_auc_score(y_true=y_test, y_score=y_scores)
        print("AUC: " + str(auc))

    y_classpred = y_pred_df.iloc[:, 0]
    print('Optimization was towards metric, but following score is always accuracy:')
    print("Accuracy: " + str(accuracy_score(y_test, y_classpred)))

    # TO DO: See if we can use the h2o-sklearn wrappers here instead
    class_predictions = y_pred_df.iloc[:, 0].values.astype(np.str_)
    class_probabilities = y_pred_df.iloc[:, 1:].values

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
