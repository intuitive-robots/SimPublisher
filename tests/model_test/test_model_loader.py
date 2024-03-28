import os

from sim_pub.model_loader import MJCFLoader


def test_load_model_parse(model_path: str) -> MJCFLoader:
    return MJCFLoader(model_path)

def test_loader_result(loader: MJCFLoader):
    asset_library = loader.asset_lib
    for asset in asset_library._assets.values():
        assert os.path.exists(asset.file_path), f"Asset file {asset.file_path} does not exist."

    # for obj in loader.game_object_dict.values():
    #     print(obj.to_dict())
    return

# def test_connect(model_path: str):
#     streamer = MujocoStreamer(model_path=model_path)
#     streamer.start_server_thread(block=True)

    



# model_path = os.path.join(os.path.dirname(__file__), "../../models/static_scene/static_scene.xml")
# model_path = os.path.join(os.path.dirname(__file__), "../../models/pendulum/pendulum.xml")
model_path = os.path.join(os.path.expanduser("~"), "github_repository/mujoco_menagerie/franka_emika_panda/scene.xml")
model_path = os.path.abspath(model_path)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("model_path", help="Path to the model file")
    # args = parser.parse_args()
    # model_path: str = args.model_path
    # loader = load_model(model_path)
    loader = test_load_model_parse(model_path)
    test_loader_result(loader)










