import numpy as np

class action_space:
    def __init__(self, arr):
        self.space = arr

    @property
    def __call__(self):
        return self.space

    def contains(self, action):
        if action in self.space:
            return True
        else:
            return False

class sample_env:
    def __init__(self):
        self.initial_observation = np.array([1,2,3,4])
        self.observation = self.initial_observation
        self.action_space = action_space(np.array([-1,1]))
        self.observation_space = np.array([0,0,0,0])

    def seed(self, seed):
        self.seed = seed

    def reset(self):
        self.observation = self.initial_observation
        return self.observation

    def render(self):
        pass

    def step(self, action):
        return self.observation, 0, False, {}

class wrappers:
    def __init__(self):
        pass
    def Monitor(self, env, path, force):
        pass

class gym:
    def __init__(self):
        self.env_name = "fake gym"
        self.env = sample_env()
        self.wrappers = wrappers()

    def make(self, name):
        return env
