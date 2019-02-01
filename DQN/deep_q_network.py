import tensorflow as tf
import numpy as np

import model_builder

class Buffer:
    def __init__(self, batch_size, observation_shape):
        self.batch_size = batch_size
        self.observation_shape = observation_shape
        self.action_shape = 1
        self.reward_shape = 1
        self.can_train = False
        self.initialize_buffer()

    def initialize_buffer(self):
        observation_memory_shape = [self.batch_size, self.observation_shape[1],
                            self.observation_shape[2],self.observation_shape[3]]
        self.current_observation = np.zeros(observation_memory_shape)
        self.next_observation = np.zeros(observation_memory_shape)
        self.action = np.zeros([self.batch_size, self.action_shape])
        self.reward = np.zeros([self.batch_size, self.reward_shape])
        self.index = 0

    def set_buffer(self, current_observation, action, next_observation, reward):
        self.can_train = False
        self.current_observation[self.index] = current_observation
        self.action[self.index] = action
        self.next_observation[self.index] = next_observation
        self.reward[self.index] = reward
        self.index = self.index + 1
        if self.index >= self.batch_size:
            self.index = 0
            self.can_train = True

    def data_set_is_ready(self):
        return self.can_train

    ## This should be called only if data_set_is_ready
    def get_buffer(self):
        return (np.array(self.current_observation), np.array(self.action),
               np.array(self.next_observation), np.array(self.reward))

"""
Important note: our Q network does not take state and actions as inputs, instead
we input only states and require output to be of action_space dimention. Each of
the output from the network is a Q value for the actions in actions space. Since
the actions are discreet, we choose to output Q values for each action. By this,
we need not to execute the Q network actions_space number of times. Thank you!!!
"""

class deep_q_network:
    def __init__(self, env, observation_shape, action_shape, args):
        self.learning_rate = args.lr_rate
        self.gamma = args.discount_factor
        self.env = env
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        # Buffer is initialized at the construction
        self.buffer = Buffer(args.buffer_size, observation_shape)
        self._build_training_graph()

    def _build_training_graph(self):
        inference_graph_builder = model_builder.SimpleNet(
                                self.observation_shape, self.action_shape)
        ## We train the local_net with a batch of (x,y)
        ## x == (s,a) where state is sampled and action is calculated with
        ## argmax on global_net
        ## y == r + gamma*(maximum value on observed state) where r is
        ## environment observed and max value is calculated from global_net
        with tf.variable_scope('local_net'):
            self.model_local = inference_graph_builder.build_model()
        ## Once we sufficiently train the local_net, we copy the it's weights
        ## to global_net
        ## Global_net has its own set of weights (in it's variable scope)
        with tf.variable_scope('global_net'):
            self.model_global = inference_graph_builder.build_model()

        # For defining loss, actual label y
        # This is define by r + max(global_net_out(next_observation))
        self.reward = tf.placeholder(tf.float32, shape = [None],
                                     name = 'reward')
        global_output = self.model_global['output']
        self.next_observation = self.model_global['input']
        max_value = self.reward+self.gamma*tf.reduce_max(global_output, axis=1)
        # We want to stop the gradient flow when minimizing using this variable
        self.label = tf.stop_gradient(max_value)

        # For defining loss, the output from local_net is also required
        # This requires an action and the current state, see the notes above
        self.action = tf.placeholder(tf.int32, shape = [None], name = 'action')
        self.current_observation = self.model_local['input']
        predicted_q_values = self.model_local['output']
        # From this, we want to have that value which corresponds to the action
        # Our predicted is of type [None, nr_actions], so to each batch, we
        # want to index the actions predicted by max of global_net
        # We need indexing vector like [[0, max_actions], [1, max_action], ...]
        action_index = tf.stack([tf.range(tf.shape(self.action)[0],
                                           dtype=tf.int32),
                                  self.action], axis=1)
        self.prediction =tf.gather_nd(params=predicted_q_values,
                                      indices=action_index)

        ## Now create a loss function
        self.loss = tf.reduce_mean(tf.squared_difference(self.label,
                                                         self.prediction))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_ops = optimizer.minimize(self.loss)

        ## Define the global updatation operations also
        ## This can be easily done by assigning all the variables from local_net
        ## scope to global_net scope
        local_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope = 'local_net')
        global_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope = 'global_net')
        self.assign_op = [tf.assign(dest, src)
                            for dest, src in zip(global_params, local_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("Q networks initialized.")

    def predict_action(self, s):
        local_q_value = self.sess.run(self.model_local['output'],
                        feed_dict={self.current_observation: s})
        pred = np.argmax(local_q_value)
        return pred

    def train_on_transition(self, s, a, r, _s):
        self.buffer.set_buffer(s, a, _s, r)
        if self.buffer.data_set_is_ready():
            buffer = self.buffer.get_buffer()
            self.update_local_net(buffer)

    def update_local_net(self, buffer):
        self.sess.run(self.train_ops, feed_dict = {
            self.current_observation: buffer[0],
            self.action: buffer[1],
            self.reward: buffer[3],
            self.next_observation: buffer[2],
        })
        return

    def update_global_net(self):
        self.sess.run(self.assign_op)
        return
