import numpy as np

"""
Fake Gym is a simulation of Gym
It just mimics the APIs provided by Gym, but most of them is non-functional
"""

# Wrappers class for gym
class wrappers:
    def __init__(self):
        pass
    def Monitor(self, env, path, force):
        return env

# A mimic of action space, that has contains functionality
class action_space:
    def __init__(self, arr):
        self.space = arr
        self.n = len(arr)

    def __get__(self):
        return self.space
    # This is why this class exists
    def contains(self, action):
        if action in self.space:
            return True
        else:
            return False

# Sample environment for fake gym
# This is where works is needed to be done
class sample_env:
    def __init__(self, name):
        self.name = name
        self.initial_observation = np.array([0,2])
        self.observation = np.array([0,2])
        self.action_space = action_space(np.array([0,1,2,3]))
        # Observation space is not correct, it should reflects
        # how many values the observation can take
        # But for now, it has been defined such that it should have a shape
        self.observation_space = np.array([0,2])

    def seed(self, seed):
        self.seed = seed

    def reset(self):
        self.observation = np.array([0,2])
        return self.observation

    # We dont render out environment
    def render(self):
        pass

    # This is an important function to define
    def step(self, action):
        new_state = self._take_action(action)
        done = self._terminal_state(new_state)
        reward = self._get_reward(new_state)
        self.observation = new_state
        return new_state, reward, done, {}

    def _take_action(self, action):
        new_state = self.observation
        if action == 0:# Left
            new_state[0] = new_state[0] -1
        if action == 1: # Right
            new_state[0] = new_state[0] + 1
        if action == 2: # Down
            new_state[1] = new_state[1] + 1
        if action == 3: # Up
            new_state[1] = new_state[1] - 1
        return new_state

    def _terminal_state(self, state):
        cond1 = self._out_of_bound(state)
        cond2 = self._dollar_state(state)
        cond3 = self._block_state(state)
        #print(state)
        if cond1 or cond2 or cond3:
            return True
        else:
            #print("Not Terminal")
            return False

    def _get_reward(self, state):
        INF = 0
        positive_reward = 6
        if self._out_of_bound(state):
            return -INF
        if self._dollar_state(state):
            return positive_reward
        if self._block_state(state):
            return -INF
        return 0

    def _out_of_bound(self, state):
        if state[0] < 0 or state[0] >= 5:
            return True
        if state[1] < 0 or state[1] >= 5:
            return True
        return False

    def _dollar_state(self, state):
        cond1 = (state[0] == 3 and state[1] == 3)
        cond2 = (state[0] == 3 and state[1] == 0)
        if cond1 or cond2:
            return True
        return False

    def _block_state(self, state):
        if state[0] == 3 and state[1] == 1:
            return True
        return False

def make(name):
    env = sample_env(name)
    return env

wrappers = wrappers()
