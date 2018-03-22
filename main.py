"""
Crated by: bulletcross@gmail.com (Vishal Keshav)
"""
import gym_env

def main():
    env = gym_env.gym_env('CartPole-v0')
    env.reset()
    for _ in range(100000):
        action = env.get_action_space()
        observation, reward, done, _ = env.next(action.sample())
        if done:
            break
        print(observation)

if __name__ == "__main__":
    main()
