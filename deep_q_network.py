"""
Deep Q Network module
"""
import tensorflow as tf

class deep_q_network(object):
    def __init__(self, env, lr, g, input_dim, output_dim):
        self.learning_rate = lr
        self.gamma = g
        self.input_dim = [None, input_dim[0], input_dim[1], input_dim[2]]
        self.output_dim = [None, output_dim]
        self.env = env

    def initialize(self):
        print("Initialization done")
        print(self.input_dim)
        print(self.output_dim)

    def predict_action(self,s):
        return self.env.action_space.sample()

    def update_local_net(self,s, a, r, _s):
        print("Updating local net")
        return

    def update_global_net(self):
        print("Updating global net")
        return
