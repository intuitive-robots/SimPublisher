import argparse

from simpub.core.net_manager import init_node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    node = init_node(args.host, "MasterNode")
    node.start_node_broadcast()
    node.spin()
