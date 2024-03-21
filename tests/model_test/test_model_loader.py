import mujoco
import os

from sim_pub.model_loader import MJCFLoader
from sim_pub.mujoco.mujoco_streamer import MujocoStreamer


def load_model(model_path: str) -> MJCFLoader:
    return MJCFLoader(model_path)

def test_connect(model_path: str):
    streamer = MujocoStreamer(model_path=model_path)
    streamer.start_server_thread(block=True)

def test_mesh_loader(model_path: str):
    loader = load_model(model_path)
    loader.asset_lib.include_mesh("test_mesh", "test_mesh_path")
    print(loader.asset_lib.asset_path)

# model_path = os.path.join(os.path.dirname(__file__), "../../models/static_scene/static_scene.xml")
# model_path = os.path.join(os.path.dirname(__file__), "../../models/pendulum/pendulum.xml")
model_path = os.path.join(os.path.expanduser("~"), "github_code/mujoco_menagerie/franka_emika_panda/scene.xml")
model_path = os.path.abspath(model_path)

if __name__ == "__main__":

    # loader = load_model(model_path)
    test_connect(model_path)










