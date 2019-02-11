"""
Controller model architecture
"""
import numpy as np
import tensorflow as tf

class SimpleNet:
    def __init__(self, input_shape, output_shape):
        self.layers = []
        self.network = {}
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(self):
        self.input = tf.placeholder(tf.float32, shape = self.input_shape,
                                    name = 'input_tensor')
        self._build_network()
        self.output = tf.identity(self.network['logits'], name ='output_tensor')
        self.model = {'input': self.input, 'output': self.output}
        return self.model

    def _build_network(self):
        with tf.name_scope("layer_1"):
            shape = int(self.input.get_shape()[1])
            W_fc1 = self.weight_variable([shape, 32])
            b_fc1 = self.bias_variable([32])
            self.network['h_fc1'] = tf.nn.relu(tf.matmul(
                                self.input, W_fc1) + b_fc1)

        with tf.name_scope("layer_2"):
            W_fc2 = self.weight_variable([32, 64])
            b_fc2 = self.bias_variable([64])
            self.network['h_fc2'] = tf.nn.relu(tf.matmul(
                                self.network['h_fc1'], W_fc2) + b_fc2)

        with tf.name_scope("layer_3"):
            W_fc3 = self.weight_variable([64, 16])
            b_fc3 = self.bias_variable([16])
            self.network['h_fc3'] = tf.nn.relu(tf.matmul(
                                self.network['h_fc2'], W_fc3) + b_fc3)

        with tf.name_scope("layer_4"):
            W_fc4 = self.weight_variable([16, self.output_shape[-1]])
            b_fc4 = self.bias_variable([self.output_shape[-1]])
            self.network['logits'] = tf.matmul(
                                        self.network['h_fc3'], W_fc4) + b_fc4

    def weight_variable(self, shape):
        with tf.name_scope("weight"):
            initial = tf.truncated_normal(shape, stddev=0.1)
            var = tf.Variable(initial)
            variable_summaries(var)
        return var

    def bias_variable(self, shape):
        with tf.name_scope("bias"):
            initial = tf.constant(0.1, shape=shape)
            var = tf.Variable(initial)
            variable_summaries(var)
        return var


def variable_summaries(var):
    return
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
