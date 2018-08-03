"""
Environment class definition.
Reference: https://gym.openai.com/docs/
Author: bulletcross@gmail.com (Vishal Keshav)
"""

import gym
import numpy as np

class gym_env(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.seed(10)
        self.observation_shape = self.env.reset().shape
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
		#Output the observation to monitor
        self.env = gym.wrappers.Monitor(self.env, "./", force = True)

    def reset(self):
        """
		Reset the environment and returns array the observations
        """
        observation = self.env.reset()
        return observation

    def get_observation_shape(self):
        self.observation_shape

    def next(self,action):
        """
		Given an action, step up the environment
        """
        if not self.env.action_space.contains(action):
            print("Invalid action", action)
        self.env.render()

        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def get_action_space(self):
        return self.action_space

    def print_observation_space(self):
        print(self.observation_space)
