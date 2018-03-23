"""
Crated by: bulletcross@gmail.com (Vishal Keshav)
"""
import gym_env
import net_param
import deep_q_network
#210,160,3
#0,1,2,3,4,5

def main():
    env = gym_env.gym_env('SpaceInvaders-v0')
    param = net_param.get_param()
    DQN = deep_q_network(param["learning_rate"], param["gamma"], env.reset().shape,6)
    for e in range(10):
        observation = env.reset()
        #print(observation.shape)
        while True:
            action = DQN.predict_action(observation)
            #action = env.get_action_space()
            #print(action.sample())
            new_observation, reward, done, _ = env.next(action)
            DQN.update_local_net(observation, action, reward, new_observation)
            if current_step > param["learn_step"]:
                DQN.update_global_net()
            if done:
                print("*************")
                break
            #print(observation)

if __name__ == "__main__":
    main()
