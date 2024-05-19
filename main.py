
import math
import numpy as np
from simpub import SimPublisher, SimScene
import mujoco as mj

from simpub.simdata import SimJointType
from simpub.transform import quat2euler
scene = SimScene.from_file("scenes/agility_cassie/scene.xml")

publisher = SimPublisher(scene)

model : mj._structs.MjModel = mj.MjModel.from_xml_string(scene.xml_string, scene.xml_assets)
data : mj._structs.MjData = mj.MjData(model)

for joint in scene.worldbody.get_joints({SimJointType.HINGE}):
    mjjoint = data.joint(joint.name)
    publisher.track_joint(joint.name, (mjjoint,), lambda x: np.degrees([x.qpos[0], x.qvel[0]]))

for joint in scene.worldbody.get_joints({SimJointType.SLIDE}):
    mjjoint = data.joint(joint.name)
    publisher.track_joint(joint.name, (mjjoint,), lambda x: np.concatenate([x.qpos, x.qvel]))

for joint in scene.worldbody.get_joints({SimJointType.BALL}):
    mjjoint = data.joint(joint.name)
    publisher.track_joint(joint.name, (mjjoint,), lambda x: np.concatenate([quat2euler(x.qpos), x.qvel]))




publisher.start()

mj.mj_resetDataKeyframe(model, data, 0)

mj.mj_forward(model,data)
while True:
  mj.mj_step(model, data)
  
publisher.shutdown()
