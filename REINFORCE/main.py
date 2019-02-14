import numpy as np
import argparse
from collections import deque

import gym_env
import reinforce

def argument_parser():
    parser = argparse.ArgumentParser(description = "Testing DQN networks")
    parser.add_argument('--env_name', default = 'CartPole-v0', type = str,
                    help = 'agent from gym')
    parser.add_argument('--buffer_size', default = 0, type = int,
                    help = 'size of circuler buffer storing (s,a,r,s_,done) \
                            tuple')
    parser.add_argument('--batch_size', default = 0, type = int,
                    help = 'number of elements to train on at once')
    parser.add_argument('--min_explore_prob', default = 0.01,
                    type = float, help = 'minimum probability to randomly take \
                                          action in epsilon-greedy search')
    parser.add_argument('--max_explore_prob', default = 1.0,
                    type = float, help = 'maximum probability to randomly take \
                                          action in epsilon-greedy search')
    parser.add_argument('--explore_prob_decay', default = 0.995,
                    type = float, help = 'decay in probability to randomly \
                                          take action in epsilon-greedy search')
    parser.add_argument('--max_train_steps', default = 1000, type = int,
                    help = 'maximum number of observation, -1 for infinity')
    parser.add_argument('--nr_episodes', default = 1000, type = int,
                    help = 'number of episodes to train the agent')
    parser.add_argument('--global_update_frequency', default = 1, type = int,
                    help = 'n_steps to take before updating the global network')
    parser.add_argument('--local_update_frequency', default = 1, type = int,
                    help = 'n_steps to take before updating the local network')
    parser.add_argument('--monitor_training', default = False, type = bool,
                    help = 'display the training environment')
    parser.add_argument('--lr_rate', default = 0.01, type = float,
                    help = 'learning rate for training the local DQN network')
    parser.add_argument('--discount_factor', default = 1.0, type = float,
                    help = 'discount factor for reward calculation')
    parser.add_argument('--weighting_factor', default = 0.0, type = float,
                    help = 'weighting factor for reward calculation w.r.t. \
                            pas reward')
    parser.add_argument('--log_path', default = 'checkpoint', type = str,
                    help = 'Path to save the logs and trained weights')
    parser.add_argument('--restore', default = True, type = bool,
                    help = 'Weather to restore the checkpoint or not')
    parser.add_argument('--checkpoint_freq', default = 1000, type = int,
                    help = 'Number of episodes after which to save checkpoints')
    parser.add_argument('--test', default = False, type = bool,
                    help = 'Testing the agent')
    args = parser.parse_args()
    return args

class update_condition:
    def __init__(self, update_frequency):
        self._update_frequency = update_frequency
        self._inbuilt_timer = 0

    def __call__(self):
        self._inbuilt_timer = self._inbuilt_timer + 1
        if self._inbuilt_timer%self._update_frequency == 0:
            self._reset()
            return True
        else:
            return False

    def _reset(self):
        self._inbuilt_timer = 0

def main():
    args = argument_parser()
    env = gym_env.gym_env(args)
    observation_shape = [None] + list(env.get_observation_shape())
    action_shape = [None] + [env.get_action_space().n]
    REIN = reinforce.reinforce(env, observation_shape, action_shape, args)

    while args.test:
        observation = env.reset_environment()
        while True:
            action = REIN.predict_action(np.expand_dims(observation, axis=0))
            new_observation, reward, done, _ = env.take_action(np.array(action))
            if done:
                break
            observation = new_observation

    non_train_condition = update_condition(args.nr_episodes)
    checkpoint_save_condition = update_condition(args.checkpoint_freq)

    scores_window = deque(maxlen=100)
    i_episode = 0
    while not non_train_condition():
        i_episode = i_episode + 1
        observation = env.reset_environment()
        cummulative_reward = 0
        non_step_condition = update_condition(args.max_train_steps)
        ## We run full episode once and collect the data (sequentially)
        while not non_step_condition():
            action = REIN.predict_action(np.expand_dims(observation, axis=0))
            print(action)
            new_observation, reward, done, _ = env.take_action(np.array(action))
            cummulative_reward = cummulative_reward + reward
            REIN.update_on_transition(observation, np.array(action),
                                    reward, new_observation, done)
            if done:
                scores_window.append(cummulative_reward)
                break
            observation = new_observation
        REIN.train()
        #if checkpoint_save_condition():
        #    REIN.save_checkpoint()
        print("Episode " + str(i_episode) + " Average_score " + str(np.mean(scores_window)))

if __name__ == "__main__":
    main()
