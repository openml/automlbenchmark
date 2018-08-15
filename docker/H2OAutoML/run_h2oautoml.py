import sys


from sklearn.metrics import accuracy_score
import h2o
from h2o.automl import H2OAutoML

if __name__ == '__main__':
    train_data_path = "../common/train.arff"
    test_data_path = "../common/test.arff"

    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    # Harcoded for testing
    runtime_seconds = 30
    number_cores = -1
    performance_metric = "AUC"

    print('Starting H2O cluster')
    # TO DO: Maybe pass in a memory size as an argument to use here
    #h2o.init(ncores=number_cores, max_mem_size=16)
    h2o.init(max_mem_size=16)           

    print('Loading data.')
    train = h2o.upload_file(train_data_path)
    test = h2o.upload_file(test_data_path)

    print('Running H2O AutoML with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('Running model on task.')
    runtime_min = (int(runtime_seconds)/60)
    print('ignoring runtime.')
    aml = H2OAutoML(max_runtime_secs=runtime_seconds, sort_metric=performance_metric)
    aml.train(y = 0, training_frame = train)
    #y_pred = aml.predict(X_test)
    # TO DO: Convert to pandas df so we can use accuracy_score
    #print('Optimization was towards metric, but following score is always accuracy:')
    #print("THIS_IS_A_DUMMY_TOKEN " + str(accuracy_score(y_test, y_pred)))
