"""
CNN graph generator module
"""

import tensorflow as tf

def build_graph(input_observation):
    layer = tf.layers.conv2d(inputs = input_observation, filters=8, kernel_size=[3,3],
                                padding = "same", activation = tf.nn.relu)
    layer = tf.layers.max_pooling2d(inputs = layer, pool_size=[2,2], strides = 2)
    layer = tf.layers.conv2d(inputs = layer, filters=32, kernel_size=[3,3],
                                padding = "same", activation=tf.nn.relu)
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides = 2)
    layer = tf.contrib.layers.flatten(inputs = layer)
    layer = tf.layers.dense(inputs=layer, units = 1024, activation = tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units = 128, activation = tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units = 6)
    return layer
