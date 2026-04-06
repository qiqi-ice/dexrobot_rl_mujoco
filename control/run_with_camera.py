import mujoco
import threading
import time
import cv2
import numpy as np

# 加载模型
#model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/scenes/box_piper.xml")
model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/models/realmanRobot.xml")


data = mujoco.MjData(model)

# 创建单个 Renderer
renderer = mujoco.Renderer(model)

# 获取摄像头 ID
camera_rgb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
camera_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "hand_camera")

# 检查摄像头是否存在
assert camera_rgb_id != -1, "rgb_camera not found"
assert camera_hand_id != -1, "hand_camera not found"

# 模拟线程
def sim_loop():
    while True:
        mujoco.mj_step(model, data)
        time.sleep(0.01)

threading.Thread(target=sim_loop, daemon=True).start()

# 主线程：交替渲染两个摄像头
while True:
    renderer.update_scene(data, camera=camera_rgb_id)
    img_rgb = renderer.render()
    cv2.imshow("RGB Camera", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    renderer.update_scene(data, camera=camera_hand_id)
    img_hand = renderer.render()
    cv2.imshow("Hand Camera", cv2.cvtColor(img_hand, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == 27:
        break
