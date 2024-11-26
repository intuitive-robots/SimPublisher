import fancy_gym
import time

from simpub.sim.fancy_gym import FancyGymPublisher

env_name = "BoxPushingDense-v0"

env = fancy_gym.make(env_name, seed=1)
obs = env.reset()

publisher = FancyGymPublisher(env, "127.0.0.1")

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if done:
        obs = env.reset()
