import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
sys.path.append("/home/sharmila/Desktop/Ann-Predictive-Analysis")

train_path = "../data/train/train_numpy.npz"
val_path = "../data/validation/validation_numpy.npz"
test_path= "../data/test/test_numpy2.npz"
def load_numpy_data(path):
    """
    A method to load train data first and split it into either test or validation data on the basis of \
    availability of test and validation data.

    Returns: return numpy ndarray of train and validation data and save the test data for future use.

    """
    with np.load(path) as data:
        features = data["arr_0"]
        labels = data["arr_1"]

        # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    return features,labels

train_data, train_label = load_numpy_data(train_path)
validation_data, validation_label = load_numpy_data(val_path)
test_data = np.concatenate((train_data[0:6], validation_data),axis = 0)
test_label = np.concatenate((train_label[0:6], validation_label),axis = 0)
np.savez(test_path, test_data, test_label)