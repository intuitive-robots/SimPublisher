import os

from simpub.model_loader import MJCFLoader


def test_load_model_parse(model_path: str) -> MJCFLoader:
    return MJCFLoader(model_path)

def test_loader_result(loader: MJCFLoader):
    asset_library = loader.asset_lib
    for asset in asset_library._assets.values():
        assert os.path.exists(asset.file_path), f"Asset file {asset.file_path} does not exist."

    for obj in loader.game_object_dict.values():
        print(obj.name, len(obj.visual))
    return


if __name__ == "__main__":

    home_path = os.path.expanduser("~")
    model_path = os.path.join(home_path, "github_repository/mujoco_menagerie/franka_emika_panda/scene.xml")
    model_path = os.path.abspath(model_path)
    loader = test_load_model_parse(model_path)
    test_loader_result(loader)










