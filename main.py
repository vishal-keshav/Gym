"""
Crated by: bulletcross@gmail.com (Vishal Keshav)
"""
import gym_env

def main():
    env = gym_env.gym_env('SpaceInvaders-v0')
    for e in range(10):
        observation = env.reset()
        for t in range(1000):
            action = env.get_action_space()
            observation, reward, done, _ = env.next(action.sample())
            if done:
                print("*************")
                break
            print(observation)

if __name__ == "__main__":
    main()
