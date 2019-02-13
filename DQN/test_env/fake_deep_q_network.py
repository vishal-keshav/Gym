import os
import random
from collections import deque
from collections import namedtuple

import tensorflow as tf
import numpy as np

import fake_model_builder as model_builder

class Buffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_data = namedtuple("buffer_datum", field_names =
                                     ["current_state", "action","next_state",
                                      "reward", "is_terminal"])
        self.can_train = False
        self.initialize_buffer()

    def initialize_buffer(self):
        self.buffer_memory = deque(maxlen = self.buffer_size)

    def set_buffer(self, current_state, action, next_state, reward,is_terminal):
        self.can_train = False
        data = self.buffer_data(current_state, action, next_state, reward,
                    int(is_terminal))
        self.buffer_memory.append(data)
        if len(self.buffer_memory) >= self.batch_size:
            self.can_train = True

    def data_set_is_ready(self):
        return self.can_train

    ## This should be called only if data_set_is_ready
    def get_buffer(self):
        random_data = random.sample(self.buffer_memory, self.batch_size)
        current_state = np.array([data.current_state for data in random_data])
        action = np.array([data.action for data in random_data])
        next_state = np.array([data.next_state for data in random_data])
        reward = np.array([data.reward for data in random_data])
        is_terminal = np.array([data.is_terminal for data in random_data])
        return (current_state, action, next_state, reward, is_terminal)

def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

"""
Important note: our Q network does not take state and actions as inputs, instead
we input only states and require output to be of action_space dimention. Each of
the output from the network is a Q value for the actions in actions space. Since
the actions are discreet, we choose to output Q values for each action. By this,
we need not to execute the Q network actions_space number of times. Thank you!!!
"""

class deep_q_network:
    def __init__(self, env, observation_shape, action_shape, args):
        # Params for learning the policy and calculate reward
        self.learning_rate = args.lr_rate
        self.gamma = args.discount_factor
        self.weighting_factor = args.weighting_factor
        # Envoroment parameters
        self.env = env
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.args = args
        self.nr_episode = 0
        self.nr_step = 0
        self.update_frequency = args.local_update_frequency
        # Buffer is initialized at the construction
        self.buffer = Buffer(args.buffer_size, args.batch_size)
        self._build_training_graph()
        self.sess.graph.finalize()

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
        self.is_terminal = tf.placeholder(tf.float32, shape = [None],
                                     name = 'is_terminal')
        global_output = self.model_global['output']
        self.next_observation = self.model_global['input']
        max_value = self.reward+self.gamma*(1-self.is_terminal)* \
                                tf.reduce_max(global_output, axis=1)
        #tf.summary.scalar('cummulative_reward', tf.reduce_mean(max_value))
        # We want to stop the gradient flow when minimizing using this variable
        self.label = tf.stop_gradient(max_value)

        # For defining loss, the output from local_net is also required
        # This requires an action and the current state, see the notes above
        # Actions is integral, staring from 0 (not one-hot encoded)
        self.action = tf.placeholder(tf.int32, shape = [None], name = 'action')
        self.current_observation = self.model_local['input']
        predicted_q_values = self.model_local['output']
        # From this, we want to have that value which corresponds to the action
        # Our predicted is of type [None, nr_actions], so to each batch, we
        # want to index the actions predicted by max of global_net
        # We need indexing vector like [[0, max_actions], [1, max_action], ...]
        action_index = tf.stack([tf.range(tf.shape(self.action)[0],
                                           dtype=tf.int32),self.action], axis=1)
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
        self.assign_op = [tf.assign(dest, self.weighting_factor*src +
                                    (1-self.weighting_factor)*dest)
                          for dest, src in zip(global_params, local_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.chk_name = os.path.join(self.args.log_path, 'model.ckpt')
        self.writer = tf.summary.FileWriter(self.args.log_path)
        self.summary_op = tf.summary.merge_all()
        mk_dir(self.args.log_path)
        self.saver = tf.train.Saver()
        if tf.train.checkpoint_exists(self.chk_name) and self.args.restore:
            self.saver.restore(self.sess, self.chk_name)
            print("session restored")
        else:
            self.writer.add_graph(self.sess.graph)
        print("Q networks initialized.")

    # Follow epsilon-greedy method
    def predict_action(self, s, epsilon):
        local_q_value = self.sess.run(self.model_local['output'],
                        feed_dict={self.current_observation: s})
        pred = np.argmax(local_q_value)
        if random.random() > epsilon:
            pred = np.argmax(local_q_value)
        else:
            pred = np.random.randint(self.action_shape[1],
                                     size=len(local_q_value))
        return pred

    def train_on_transition(self, s, a, r, _s, t):
        self.nr_step = self.nr_step + 1
        self.buffer.set_buffer(s, a, _s, r, t)
        train_cond = (self.nr_step % self.update_frequency == 0)
        if self.buffer.data_set_is_ready() and train_cond:
            s, a, s_, r, t = self.buffer.get_buffer()
            self.update_local_net(s, a, s_, r, t)

    def update_local_net(self, s, a, s_, r, t):
        self.sess.run(self.train_ops, feed_dict = {
            self.current_observation: s,
            self.action: np.squeeze(a),
            self.reward: np.squeeze(r),
            self.next_observation: s_,
            self.is_terminal: t,
        })
        #print("updating local")
        return

    def update_global_net(self):
        self.sess.run(self.assign_op)
        return

    def write_summary(self):
        s, a, s_, r, t = self.buffer.get_buffer()
        feed_dict = {
            self.current_observation: s,
            self.action: np.squeeze(a),
            self.reward: np.squeeze(r),
            self.next_observation: s_,
            self.is_terminal: t,
        }
        summary = self.sess.run(self.summary_op, feed_dict = feed_dict)
        self.writer.add_summary(summary, self.nr_episode)
        self.nr_episode = self.nr_episode + 1

    def save_checkpoint(self):
        save_path = self.saver.save(self.sess, self.chk_name)
        print("Checkpoint saved.")
