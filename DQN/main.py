import numpy as np
import argparse

import gym_env
import deep_q_network

#210,160,3
#0,1,2,3,4,5

def argument_parser():
    parser = argparse.ArgumentParser(description = "Testing DQN networks")
    parser.add_argument('--env_name', default = 'SpaceInvaders-v0', type = str,
                    help = 'agent from gym')
    parser.add_argument('--buffer_size', default = 8, type = int,
                    help = 'size of circuler buffer storing (s,a,r,s) tuple')
    parser.add_argument('--exploration_probability', default = 0.1,
                    type = float, help = 'probability to randomly take action')
    parser.add_argument('--max_train_steps', default = -1, type = int,
                    help = 'maximum number of observation, -1 for infinity')
    parser.add_argument('--nr_episodes', default = 100, type = int,
                    help = 'number of episodes to train the agent')
    parser.add_argument('--monitor_training', default = True, type = bool,
                    help = 'display the training environment')
    parser.add_argument('--lr_rate', default = 0.001, type = float,
                    help = 'learning rate for training the local DQN network')
    parser.add_argument('--discount_factor', default = 0.5, type = float,
                    help = 'discount factor for reward calculation')
    args = parser.parse_args()
    return args

class train_condition:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self):
        if self.steps != 0:
            self.steps = self.steps -1
            return True
        else:
            return False

def main():
    args = argument_parser()
    env = gym_env.gym_env(args)
    observation_shape = [None] + list(env.get_observation_shape())
    action_shape = [None] + [env.get_action_space().n]
    DQN = deep_q_network.deep_q_network(env, observation_shape,
                                        action_shape, args)
    condition = train_condition(args.max_train_steps)
    observation = env.reset_environment()
    while condition():
        observation = env.reset_environment()
        while True:
            #observation = observation.reshape([1, 210, 160, 3])
            action = DQN.predict_action(np.expand_dims(observation, axis=0))

            new_observation, reward, done, _ = env.take_action(np.squeeze(action))
            #new_observation = new_observation.reshape([1, 210, 160, 3])
            DQN.train_on_transition(observation, np.squeeze(action),
                                    np.squeeze(reward), new_observation)
            """if current_step%learning_step == 0:
                DQN.update_global_net()
            if done:
                print("*************")
                break
            current_step = current_step + 1"""
            observation = new_observation

if __name__ == "__main__":
    main()
