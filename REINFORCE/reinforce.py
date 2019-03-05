import os
import random
from collections import deque
from collections import namedtuple

import tensorflow as tf
import numpy as np

import model_builder

class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer_data = namedtuple("buffer_datum", field_names =
                                     ["current_state", "action","next_state",
                                      "reward", "is_terminal"])
        self.initialize_buffer()

    def initialize_buffer(self):
        self.buffer_memory = deque()

    def set_buffer(self, current_state, action, next_state, reward,is_terminal):
        data = self.buffer_data(current_state, action, next_state, reward,
                    int(is_terminal))
        self.buffer_memory.append(data)

    def data_set_is_ready(self):
        return True

    ## Here, we need to return full buffer
    def get_buffer(self):
        current_state = np.array([data.current_state
                                       for data in list(self.buffer_memory)])
        action = np.array([data.action for data in list(self.buffer_memory)])
        next_state = np.array([data.next_state
                                       for data in list(self.buffer_memory)])
        reward = np.array([data.reward for data in list(self.buffer_memory)])
        is_terminal = np.array([data.is_terminal
                                       for data in list(self.buffer_memory)])
        return (current_state, action, next_state, reward, is_terminal)

def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

"""
Important note: our policy network does not take state and actions as inputs,
instead we input only states and require output to be of action_space dimention.
Each of the output from the network is a probability of taking an actions. Since
the actions are discreet, we choose to output values for each action. By this,
we need not to execute the network actions_space number of times. Thank you!!!
"""

class reinforce:
    def __init__(self, env, observation_shape, action_shape, args):
        # Params for learning the policy and calculate reward
        self.learning_rate = args.lr_rate
        self.gamma = args.discount_factor
        # Envoroment parameters
        self.env = env
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.args = args
        self.nr_episode = 0
        self.nr_step = 0
        # Buffer is initialized at the construction and after each episode
        self.buffer = Buffer(args.buffer_size)
        self._build_training_graph()
        self.sess.graph.finalize()

    def _build_training_graph(self):
        inference_graph_builder = model_builder.SimpleNet(
                                self.observation_shape, self.action_shape)
        with tf.variable_scope('policy_net'):
            self.model_policy = inference_graph_builder.build_model()
        self.state_placeholder = self.model_policy['input']
        self.prob_out = self.model_policy['prob']
        self.distribution = Categorical(probs = self.prob_out)
        self.action = self.distribution.sample()
        self.action_placeholder = tf.placeholder(tf.float32, shape = (None,1))
        ## If these action probability is fetching us less reward, then these
        ## will be penalized and network parameters will update accordingly.
        self.action_prob = self.distribution.log_prob(self.action_placeholder)

        # The reward will be cummulative rewards for whole episode.
        # We will calculate this reward outside tensorflow graph and feed it.
        self.reward = tf.placeholder(tf.float32, shape = [None], name ='reward')
        # Instead of computing and applying the gradient seperately (accent),
        # We do gradient decent on the negative log(p(a/s))*R. This translates
        # to gradient accent on the probability mass function approximator param
        #self.loss = tf.reduce_mean(-tf.log(self.action_pred)*self.reward)
        self.loss = tf.math.reduce_sum(
                                    -tf.diag_part(self.action_prob)*self.reward)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_ops = optimizer.minimize(self.loss)

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

    def predict_action(self, s):
        action = self.sess.run(self.action,
                        feed_dict={self.state_placeholder: s})
        return action

    def update_on_transition(self, s, a, r, _s, t):
        self.nr_step = self.nr_step + 1
        self.buffer.set_buffer(s, a, _s, r, t)

    def train(self):
        """
        Here, we have the buffer (state action reward etc) for one full episode.
        We use this to train the policy network with the formulae:
        grad = Cummulative_reward*sum(log(policy_distribuiton))
        The grad will be used for gradient accent on the policy parameter.
        Here, instead of gradient accent, we do gradient decent on negative loss
        Basically, we have assumed our loss function to be -log(probability)*R
        Now, minimizing this has the effect of applying the gradient defined by
        gard(log(probability))*R. This is cool. No need to use apply grad in TF.
        """
        # We get all data from deque
        states, actions, s_, rewards, terminal = self.buffer.get_buffer()
        # Gamma weighting is to reduce the variance
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        feed_dict = {self.action_placeholder: np.array(actions),
                     self.state_placeholder: np.array(states),
                     self.rewards: rewards}
        _ = sess.run(self.train_ops, feed_dict = feed_dict)
        self.buffer = Buffer(self.args.buffer_size)
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
