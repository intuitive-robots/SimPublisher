import metaworld
import random
from simpub.sim.mj_publisher import MujocoPublisher


print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

env_name = "basketball-v2"

ml1 = metaworld.ML1(env_name) # Construct the benchmark, sampling tasks

env = ml1.train_classes[env_name]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

publisher = MujocoPublisher(env.model, env.data)


obs = env.reset()  # Reset environment
while True:
    continue
    a = env.action_space.sample()  # Sample an action
    env.step(a)  # Step the environment with the sampled random action