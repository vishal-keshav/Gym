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
        # Uncomment the conv layers if environment requires so
        """with tf.name_scope("layer_1"):
            W_conv1 = self.weight_variable([3, 3, self.input_shape[-1], 8])
            b_conv1 = self.bias_variable([8])
            self.network['h_conv1'] = tf.nn.relu(self.conv2d(
                                            self.input, W_conv1) + b_conv1)
            self.network['h_pool1'] = self.max_pool_2x2(self.network['h_conv1'])

        with tf.name_scope("layer_2"):
            W_conv2 = self.weight_variable([3, 3, 8, 16])
            b_conv2 = self.bias_variable([16])
            conv = self.conv2d(self.network['h_pool1'], W_conv2)
            self.network['h_conv2'] = tf.nn.relu( conv + b_conv2 )
            self.network['h_pool2'] = self.max_pool_2x2(self.network['h_conv2'])"""

        with tf.name_scope("layer_1"):
            shape = self.input_shape[-1]
            W_fc1 = self.weight_variable([shape, 64])
            b_fc1 = self.bias_variable([64])
            self.network['h_fc1'] = tf.nn.relu(tf.matmul(
                                        self.input, W_fc1) + b_fc1)
        with tf.name_scope("layer_2"):
            W_fc2 = self.weight_variable([64, 64])
            b_fc2 = self.bias_variable([64])
            self.network['h_fc2'] = tf.nn.relu(tf.matmul(
                                        self.network['h_fc1'], W_fc2) + b_fc2)

        with tf.name_scope("layer_3"):
            shape = self.output_shape[-1]
            W_fc3 = self.weight_variable([64, shape])
            b_fc3 = self.bias_variable([shape])
            self.network['logits'] = tf.matmul(
                                        self.network['h_fc2'], W_fc3) + b_fc3

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

    def conv2d(self, x, W):
        with tf.name_scope("conv"):
            conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def max_pool_2x2(self, x):
        with tf.name_scope("max_pool"):
            pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        return pool


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
