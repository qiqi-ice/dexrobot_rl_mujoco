import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
#model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/scenes/box_piper.xml")
#model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/models/realmanRobot.xml")
model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/models/dexhand021_right_realmanRobot.xml")

data = mujoco.MjData(model)

# 创建单个 Renderer
renderer = mujoco.Renderer(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while(True):
        mujoco.mj_step(model, data)
        viewer.sync()