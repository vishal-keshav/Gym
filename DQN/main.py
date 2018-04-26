"""
Crated by: bulletcross@gmail.com (Vishal Keshav)
"""
import gym_env
import net_param
import deep_q_network
import numpy as np
#210,160,3
#0,1,2,3,4,5

def main():
    env = gym_env.gym_env('SpaceInvaders-v0')
    param = net_param.get_param()
    DQN = deep_q_network.deep_q_network(env, param["learning_rate"], param["gamma"], env.reset().shape, 6)
    DQN.initialize()

    for e in range(2):
        observation = env.reset()
        current_step = 0
        #print(observation.shape)

        while True:
            observation = observation.reshape([1, 210, 160, 3])
            #print(observation.shape)
            action = DQN.predict_action(observation)

            #action = env.get_action_space()
            #print(action.sample())
            new_observation, reward, done, _ = env.next(action)
            new_observation = new_observation.reshape([1, 210, 160, 3])
            reward = np.array([reward])
            action = action.reshape([1,])
            #DQN.update_local_net(observation, action, reward, new_observation)
            DQN.store_transition(observation, action, reward, new_observation)
            if current_step%param["learn_step"] == 0:
                DQN.update_global_net()
            if done:
                print("*************")
                break
            current_step = current_step + 1
            observation = new_observation

if __name__ == "__main__":
    main()