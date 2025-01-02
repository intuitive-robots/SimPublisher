import argparse
import random
import metaworld
from simpub.sim.mj_publisher import MujocoPublisher


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--env_name", type=str, default="basketball-v2")
    args = parser.parse_args()

    # Initialize MetaWorld Benchmark
    ml1 = metaworld.ML1(args.env_name)

    # # Create Environment
    if args.env_name not in ml1.train_classes:
        raise ValueError(
            f"Environment '{args.env_name}' is not available."
            f"Available environments: {list(ml1.train_classes.keys())}")

    env = ml1.train_classes[args.env_name]()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    # Initialize MujocoPublisher
    MujocoPublisher(
        env.model,
        env.data,
        host=args.host,
        visible_geoms_groups=list(range(3))
    )

    # Main Loop
    env.reset()
    while True:
        continue


if __name__ == "__main__":
    main()
