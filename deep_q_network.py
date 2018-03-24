"""
Deep Q Network module
"""
import tf_util
import tensorflow as tf
import numpy as np

class deep_q_network(object):
    def __init__(self, env, lr, g, input_dim, output_dim):
        self.learning_rate = lr
        self.gamma = g
        self.input_dim = [None, input_dim[0], input_dim[1], input_dim[2]]
        self.output_dim = [None, output_dim]
        self.env = env

    def initialize(self):
        print("Initializing Q networks...")
        self.observation = tf.placeholder(tf.float32, self.input_dim)
        self.reward = tf.placeholder(tf.float32, [None, ])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.next_observation = tf.placeholder(tf.float32, self.input_dim)
        with tf.variable_scope('local_net'):
            self.local_net = tf_util.build_graph(self.observation)
        with tf.variable_scope('global_net'):
            self.global_net = tf_util.build_graph(self.next_observation)

        #Define loss for local network
        self.q_label = tf.stop_gradient(self.reward + self.gamma*tf.reduce_max(self.global_net, axis=1))
        action_index = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        self.q_prediction =tf.gather_nd(params=self.local_net, indices=action_index)
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_label, self.q_prediction))
        self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #Define copy constructor for network parameters as tf op that can be executed
        local_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local_net')
        global_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global_net')
        self.param_copy_op = [tf.assign(dest, src) for dest, src in zip(global_params, local_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def predict_action(self,s):
        #return self.env.action_space.sample()
        local_q_value = self.sess.run(self.local_net, feed_dict={self.observation: s})
        return np.argmax(local_q_value)

    def update_local_net(self,s, a, r, _s):
        #print("Updating local net")
        self.sess.run(self.train_ops, feed_dict = {
            self.observation: s,
            self.action: a,
            self.reward: r,
            self.next_observation: _s,
        })
        return

    def update_global_net(self):
        #print("Updating global net")
        self.sess.run(self.param_copy_op)
        return
