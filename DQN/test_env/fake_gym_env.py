"""
A gym environment should have these capabilities
1. make(env_name): makes and returns an object
2. env objects should have reset() function that returns the observation
3. env object should have action_space and observation_space
    Action space should contain "contains" check
4. A wrapper object which will have Monitor function
5. A render function
"""

import fake_gym as gym
import numpy as np

class gym_env:
    def __init__(self, args):
        self.env_name = args.env_name
        ## gym.make(environment_name) creates and returns an environment
        ## On top of this environment, we will create some wrappers
        self.env = gym.make(self.env_name)
        self.env.seed(10)
        ## Observation shape is important, becase based on this shape, the
        ## agent will define the Q-Value network
        self.observation_shape = self.env.reset().shape
        # On this environment, what actions can be taken
        self.action_space = self.env.action_space
        # What are the possible observations
        self.observation_space = self.env.observation_space
		#Output the observation to monitor
        if args.monitor_training:
            self.env = gym.wrappers.Monitor(self.env, "./", force = True)

        self.args = args

    def reset_environment(self):
        """
		Reset the environment and returns resetted observations
        """
        observation = self.env.reset()
        return observation

    def take_action(self,action):
        """
		Given an action, step up the environment
        """
        if not self.env.action_space.contains(action):
            print("Invalid action", action)
        if self.args.monitor_training:
            self.env.render()

        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
    ################## Non-essential functions ####################
    def get_observation_shape(self):
        return self.observation_shape

    def get_action_space(self):
        return self.action_space

    def print_observation_space(self):
        print(self.observation_space)
