import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from configs import configs
from shutil import copy


class DataGenerator():
    def __init__(self, datapaths, mode, config):
        self.__dict__.update(datapaths)
        self.mode = mode
        self.config = config
        self.test, self.validation = self.split_if_needed()

    def load(self, type):
        """
        A placeholder method for generating the batch of data.

        Parameters
        ----------
        :return: dataset iterator - https://www.tensorflow.org/api_docs/python/tf/data/Iterator,
        :rtype: tf.data.iterator

        Example
        -------
        datagen = DataGenerator(datapaths = {"train":"../data/w2vec/TRAIN_0.npz", "validation:"", "test":"")
        iterator = datagen.load()
        train_data, validation_data = ds.load_npy_to_tensor_slices()
        with tf.Session() as sess:
            tf.global_variables_initializer()

            for i in range(1):
                sess.run(iterator.initializer, feed_dict={feature_placeholder: train_data[0], label_placeholder: train_data[1]})
                next_element = iterator.get_next()
                datapoint = sess.run(next_element)

        """

        if type == "train":
            train = self.load_numpy_data(self.train)
            dataset = tf.data.Dataset.from_tensor_slices((train[0], train[1]))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(configs.config["batch_size"])
        elif type == "val":
            validation = self.load_numpy_data(self.validation)
            dataset = tf.data.Dataset.from_tensor_slices((validation[0], validation[1]))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(configs.config["val_batch_size"])

        iterator = dataset.make_initializable_iterator()
        return iterator

    def load_numpy_data(self, path):
        """
        A method to load train data first and split it into either test or validation data on the basis of \
        availability of test and validation data.

        Returns: return numpy ndarray of train and validation data and save the test data for future use.

        """
        with np.load(path) as data:
            self.features = data["arr_0"]
            self.labels = data["arr_1"]

            # Assume that each row of `features` corresponds to the same row as `labels`.
        assert self.features.shape[0] == self.labels.shape[0]

        return self.features, self.labels

    def split_if_needed(self):
        return "", ""
