import numpy as np
import argparse
from collections import deque

import gym_env
import deep_q_network

def argument_parser():
    parser = argparse.ArgumentParser(description = "Testing DQN networks")
    parser.add_argument('--env_name', default = 'LunarLander-v2', type = str,
                    help = 'agent from gym')
    parser.add_argument('--buffer_size', default = 100000, type = int,
                    help = 'size of circuler buffer storing (s,a,r,s_,done) \
                            tuple')
    parser.add_argument('--batch_size', default = 64, type = int,
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
    parser.add_argument('--nr_episodes', default = 2000, type = int,
                    help = 'number of episodes to train the agent')
    parser.add_argument('--global_update_frequency', default = 4, type = int,
                    help = 'n_steps to take before updating the global network')
    parser.add_argument('--local_update_frequency', default = 4, type = int,
                    help = 'n_steps to take before updating the local network')
    parser.add_argument('--monitor_training', default = False, type = bool,
                    help = 'display the training environment')
    parser.add_argument('--lr_rate', default = 0.0005, type = float,
                    help = 'learning rate for training the local DQN network')
    parser.add_argument('--discount_factor', default = 0.99, type = float,
                    help = 'discount factor for reward calculation')
    parser.add_argument('--weighting_factor', default = 0.001, type = float,
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
    DQN = deep_q_network.deep_q_network(env, observation_shape,
                                        action_shape, args)

    while args.test:
        observation = env.reset_environment()
        while True:
            action = DQN.predict_action(np.expand_dims(observation, axis=0))
            new_observation, reward, done, _=env.take_action(np.squeeze(action))
            if done:
                break
            observation = new_observation

    non_train_condition = update_condition(args.nr_episodes)
    global_update_condition = update_condition(args.global_update_frequency)
    checkpoint_save_condition = update_condition(args.checkpoint_freq)
    epsilon = args.max_explore_prob
    scores = deque(maxlen=100)
    episode = 0
    while not non_train_condition():
        episode = episode + 1
        observation = env.reset_environment()
        cummulative_reward = 0
        non_step_condition = update_condition(args.max_train_steps)
        while not non_step_condition():
            action = DQN.predict_action(np.expand_dims(observation, axis=0),
                                        epsilon)
            new_observation, reward, done, _=env.take_action(np.squeeze(action))
            cummulative_reward = cummulative_reward + reward
            DQN.train_on_transition(observation, np.squeeze(action),
                                    np.squeeze(reward), new_observation, done)
            if global_update_condition():
                #print("updating global")
                DQN.update_global_net()
            if done:
                #print(cummulative_reward)
                scores.append(cummulative_reward)
                #DQN.write_summary()
                break
            observation = new_observation
        #if checkpoint_save_condition():
        #    DQN.save_checkpoint()
        epsilon = max(epsilon*args.explore_prob_decay, args.min_explore_prob)
        print(epsilon)
        print("Episode " + str(episode) + "  avg_score " + str(np.mean(scores)))

if __name__ == "__main__":
    main()
