"""
Deep Q Network module
"""
import tf_util
import tensorflow as tf
import numpy as np

class deep_q_network(object):
    def __init__(self, env, param):
        self.learning_rate = param['learning_rate']
        self.gamma = param['gamma']
        self.env = env
        self.mem_index = 0

    def set_param(self, param):
        self.learning_rate = param['learning_rate']
        self.gamma = param['gamma']

    def initialize(self, input_dim, output_dim):
        print("Initializing Q networks...")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.observation = tf.placeholder(tf.float32, self.input_dim)
        self.reward = tf.placeholder(tf.float32, [None, ])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.next_observation = tf.placeholder(tf.float32, self.input_dim)
        #Episodic memory initialization
        self.observation_memory = np.zeros([8, self.input_dim[1], self.input_dim[2], self.input_dim[3]])
        self.next_observation_mem = np.zeros([8, self.input_dim[1], self.input_dim[2], self.input_dim[3]])
        self.action_mem = np.zeros([8, ])
        self.reward_mem = np.zeros([8, ])

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
        print("Q networks initialized.")

    def predict_action(self,s):
        #return self.env.action_space.sample()
        local_q_value = self.sess.run(self.local_net, feed_dict={self.observation: s})
        pred = np.argmax(local_q_value)
        print("Action predicted from local network ", pred)
        return pred

    def store_transition(self, s, a, r, _s):
        print("Storing the transition")
        self.observation_memory[self.mem_index] = s
        self.action_mem[self.mem_index] = a
        self.reward_mem[self.mem_index] = r
        self.next_observation_mem[self.mem_index] = _s
        if self.mem_index == 7:
            self.update_local_net(self.observation_memory, self.action_mem,
                            self.reward_mem, self.next_observation_mem)
        self.mem_index = (self.mem_index + 1)%8

    def update_local_net(self, s, a, r, _s):
        print("Updating local net")
        self.sess.run(self.train_ops, feed_dict = {
            self.observation: s,
            self.action: a,
            self.reward: r,
            self.next_observation: _s,
        })
        return

    def update_global_net(self):
        print("Updating global net")
        self.sess.run(self.param_copy_op)
        return
