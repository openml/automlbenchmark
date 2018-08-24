""" Calculates a score for the predictions given the true values.
input:
 - path to original test data arff
 - path to a csv file containing predictions.
   For classification, it should be class probabilities as well as predicted class
 - metric to calculate.
   Valid values: 'acc', 'log_loss', 'mse', 'rmse'
"""
import sys
from math import sqrt

import arff
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

test_data_filepath = sys.argv[1]
predictions_filepath = sys.argv[2]
metric = sys.argv[3]

token = '6744dfceeb4d2b4a9e60874bcd46b3a1'
metric_mapping = dict(
    acc=accuracy_score,
    log_loss=log_loss,
    mse=mean_squared_error,
    rmse=lambda y1, y2: sqrt(mean_squared_error(y1, y2))
)

with open(test_data_filepath, 'r') as arff_data_file:
    data_arff = arff.load(arff_data_file)

target_name, target_type = data_arff['attributes'][-1]
y_true = np.asarray(data_arff['data'])[:, -1]
y_pred = np.loadtxt(predictions_filepath, dtype='str', delimiter=',')

if isinstance(target_type, list):
    # targets are classes, so we must convert the target labels to a one-hot encoding to calculate the metric.
    labelEncoder = LabelEncoder().fit(target_type)
    y_true_le = labelEncoder.transform(y_true).reshape(-1, 1)
    y_true_ohe = OneHotEncoder().fit_transform(y_true_le)

    class_probabilities, class_predictions = y_pred[:, :-1].astype(float), y_pred[:, -1]

    if metric == 'acc':
        score = accuracy_score(y_true, class_predictions)
    elif metric == 'log_loss':
        score = log_loss(y_true_ohe, class_probabilities)
    else:
        raise ValueError("Predictions determined to be classification, but {} is not a known classification metric."
                         .format(metric))
elif target_type == 'REAL':
    y_true = y_true.astype(float)
    score = metric_mapping[metric](y_true, y_pred)
else:
    raise ValueError("Unexpected attribute type for target: {}.".format(target_type))

print(token, score)
