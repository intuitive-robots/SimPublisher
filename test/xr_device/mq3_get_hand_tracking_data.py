import mujoco as mj
import mujoco.viewer
import os
from simpub.core import init_xr_node_manager
from simpub.xr_device.meta_quest3 import MetaQuest3
import time
from mujoco import mj_name2id, mjtObj  # type: ignore
import numpy as np

from simpub.xr_device.meta_quest3 import MetaQuest3HandBoneID, MetaQuest3Hand


def update_hand_tracking_data(
    mj_model, hand_data: MetaQuest3Hand, hand_name: str
):
    for bone in MetaQuest3HandBoneID:
        object_id = mj_name2id(
            mj_model, mjtObj.mjOBJ_BODY, f"{hand_name}_{bone.name}"
        )
        mj_model.body_pos[object_id] = np.array(
            hand_data["bones"][bone.value]["pos"]
        )


def run_hand_simulation():
    """
    Load and run the hand simulation with interactive viewer.

    Args:
        xml_file: Path to the MJCF XML file
    """
    try:
        # Load the model
        file_path = os.path.join(
            os.path.dirname(__file__), "./utils/openxr_hand.xml"
        )
        print(f"Loading model from {file_path}...")
        model = mj.MjModel.from_xml_path(file_path)
        data = mj.MjData(model)
        net_manager = init_xr_node_manager("192.168.0.134")
        net_manager.start_discover_node_loop()
        mq3 = MetaQuest3("IRL-MQ3-1")
        mq3.wait_for_connection()
        response = mq3.request("ToggleHandTracking", "")
        print(response)
        print("\nStarting interactive viewer...")
        # Launch the interactive viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                hand_data = mq3.get_hand_tracking_data()
                if hand_data:
                    # update the model data with hand tracking data
                    update_hand_tracking_data(
                        model, hand_data["leftHand"], "Left"
                    )
                    update_hand_tracking_data(
                        model, hand_data["rightHand"], "Right"
                    )
                # Step the simulation
                mj.mj_step(model, data)
                viewer.sync()
                # Control frame rate
                time.sleep(0.01)  # 100 FPS

    except FileNotFoundError:
        print(f"Error: Could not find XML file '{file_path}'")
        print(
            "Make sure the hand_skeleton.xml file is in the current directory"
        )
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    # Run the simulation
    run_hand_simulation()
