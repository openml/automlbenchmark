import arff
import numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Relative directories as if from automl folder.
TRAIN_DATA_PATH = "/bench/common/train.arff"
TEST_DATA_PATH = "/bench/common/test.arff"
OUTPUT_PATH = "/bench/common/predictions.csv"


def get_class_names_from_arff(arff_file_path):
    """ Retrieves the class names (possible attribute values of last attribute) from the arff file.

    :param arff_file_path: string. path to the arff file.
    :return: a list of class names
    """
    with open(arff_file_path, 'r') as arff_data_file:
        data_arff = arff.load(arff_data_file)
    attribute_name, attribute_values = data_arff['attributes'][-1]
    return attribute_values


def get_X_y_from_arff(arff_file_path):
    """ Read data from the ARFF file as X and y, where y is the last column and X all other data.

    :param arff_file_path: string. path to the arff file.
    :return: a tuple of two numpy arrays.
    """
    with open(arff_file_path, 'r') as arff_data_file:
        data_arff = arff.load(arff_data_file)
        data = numpy.asarray(data_arff['data'])
        return data[:, :-1], data[:, -1]


def one_hot_encode_predictions(predictions, reference_file=TRAIN_DATA_PATH):
    """ Performs one-hot encoding on predictions, order of column depends on reference file.

    :param predictions: vector of target label predictions
    :param reference_file: reference arff file which defines the target as last attribute.
      This is used to order the columns of the one-hot encoding.
    :return: a one hot encoding of the class predictions as numpy array.
    """
    with open(reference_file, 'r') as arff_data_file:
        arff_data = arff.load(arff_data_file)
        target_name, target_type = arff_data['attributes'][-1]

    le = LabelEncoder().fit(target_type)
    class_predictions_le = le.transform(predictions).reshape(-1, 1)
    class_probabilities = OneHotEncoder().fit_transform(class_predictions_le)
    return class_probabilities.todense()


def save_predictions_to_file(class_probabilities, class_predictions, file_path=OUTPUT_PATH):
    """ Save class probabilities and predicted labels to file in csv format.

    :param class_probabilities: (N,K)-matrix describing class probabilities.
    :param class_predictions:  (N,) or (N,1) vector.
    :param file_path: string. File to save the predictions to.
    :return: None
    """
    if class_predictions.ndim == 1:
        class_predictions = class_predictions.reshape(-1, 1)
    combined_predictions = numpy.hstack((class_probabilities, class_predictions)).astype(str)
    numpy.savetxt(file_path, combined_predictions, delimiter=',', fmt="%s")
