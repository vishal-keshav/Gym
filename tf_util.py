"""
CNN graph generator module
"""

import tensorflow as tf

def build_graph(input_observation):
    layer1 = tf.layers.conv2d(inputs = input_observation, filters=6, kernel_size=[3,3],
    padding = "same", activation = tf.nn.relu)
    layer2 = tf.layers.max_pooling2d(inputs = layer1, pool_size=[3,3], strides = 3)
    layer3 = tf.layers.conv2d(inputs = layer2, filters=12, kernel_size=[3,3],
    padding = "same", activation=tf.nn.relu)
    layer4 = tf.layers.max_pooling2d(inputs=layer3, pool_size=[5,5], strides = 5)
    print(tf.shape(layer4))
    flat = tf.reshape(layer4, [1, 21*16*12])
    dense1 = tf.layers.dense(inputs=flat, units = 1024, activation = tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units = 6)
    return dense2
