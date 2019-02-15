import sys
import os

from collections import defaultdict
import tensorflow as tf
from base.base_model import BaseModel
from configs import configs


class AnnModel(BaseModel):

    def __init__(self, config):
        """

        :param config:
        """
        self.layer_details = config["hidden_layer"]
        self.config = config
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=list(self.config["input_shape"]),
                                    name="input")

        self.target = tf.placeholder(dtype=tf.int8,
                                     shape=[None, 1],
                                     name="label")

        self.model_params = defaultdict()

        self.build_model()
        self.init_saver()
        self.__dict__.update(self.model_params)

    def build_model(self):



        activated_output = self.add_hidden_layer("first_layer")

    def add_hidden_layer(self, layer_name):
        """

        :param layer_name:
        :return:
        """
        with tf.variable_scope(name_or_scope=layer_name + "_hidden"):
            weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01
                                                            )
            # step 2: create the weight variable with proper initialization
            W = tf.get_variable(name=layer_name + "_weights", dtype=tf.float32,
                                shape=[self.config["input_shape"][-1],self.layer_details[layer_name]["number"]],
                                initializer=weight_initer)

            bias = tf.get_variable(name=layer_name + "_b", shape=self.layer_details[layer_name]["number"],
                                   initializer=tf.zeros_initializer(), dtype=tf.float32)

            output = tf.nn.xw_plus_b(self.input, W, bias, name=layer_name + "output")

            activated_output = tf.nn.sigmoid(output)

            return activated_output

    def output_layer(self, hidden_output):
        """

        :param hidden_output:
        :return:
        """


