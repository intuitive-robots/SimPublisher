from simpub.mjcf import MJCFParser
from simpub import SimPublisher
import mujoco
import glfw

xml_path = "model/alr_lab/scene.xml"
parser = MJCFParser(xml_path)

scene = parser.parse()
publisher = SimPublisher(scene)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
# model = load_model_from_xml(scene.xml_string)
# sim = MjSim(model)
glfw.init()
window = glfw.create_window(1200, 900, "MuJoCo Simulation", None, None)
glfw.make_context_current(window)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 仿真和可视化循环
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)

    # 获取窗口尺寸并更新渲染器视图
    viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvScene())
    mujoco.mjr_render(viewport, mujoco.MjvScene(), context)

    # 交换OpenGL缓冲区
    glfw.swap_buffers(window)
    glfw.poll_events()

# 关闭窗口和OpenGL上下文
glfw.terminate()
# REVIEW: I suggest to use the SimPublisher to track them automatically
# and also usrs can specify joints they want or don't want to track
# for joint in scene.worldbody.get_joints({UnityJointType.HINGE}):
#     mjjoint = data.joint(joint.name)
#     publisher.track_joint(joint.name, (mjjoint,), lambda x: np.degrees([x.qpos[0], x.qvel[0]]))

# for joint in scene.worldbody.get_joints({UnityJointType.SLIDE}):
#     mjjoint = data.joint(joint.name)
#     publisher.track_joint(joint.name, (mjjoint,), lambda x: np.concatenate([x.qpos, x.qvel]))

# for joint in scene.worldbody.get_joints({UnityJointType.BALL}):
#     mjjoint = data.joint(joint.name)
    # publisher.track_joint(joint.name, (mjjoint,), lambda x: np.concatenate([quat2euler(x.qpos), x.qvel]))

# publisher.start()

# while True:
#     mujoco.mj_step(model, data)
#     viewer.render()

publisher.shutdown()
