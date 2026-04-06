#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64, Float32MultiArray
import mujoco
import mujoco.viewer
import cv2
import threading
import time
import numpy as np

# === 加载模型 ===
model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/scenes/box_realmanRobot.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)  # 只用一个 renderer，避免 OpenGL 冲突

# === 摄像头 ID（一个固定，一个跟随）===
camera_rgb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
camera_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "hand_camera")
assert camera_rgb_id != -1 and camera_hand_id != -1, "Camera ID not found!"

# === 初始手势配置 ===
def get_pinch_config():
    return {
        "initial": {
            "joint1": 0, "joint2": 0, "joint3": 0,
            "joint4": 0, "joint5": 0, "joint6": 0,
            "act_r_f_joint1_1": 1.50, "act_r_f_joint1_2": 0.25,
            "act_r_f_joint1_3": 0.35, "act_r_f_joint1_4": 0.25,
            "act_r_f_joint2_2": 0, "act_r_f_joint2_3": 0, "act_r_f_joint2_4": 0,
            "act_r_f_joint3_2": 0, "act_r_f_joint3_3": 0, "act_r_f_joint3_4": 0,
            "act_r_f_joint4_2": 0, "act_r_f_joint4_3": 0, "act_r_f_joint4_4": 0,
            "act_r_f_joint5_2": 0, "act_r_f_joint5_3": 0, "act_r_f_joint5_4": 0,
        },
    }

actuator_names = [
    "joint1", "joint2", "joint3","joint4", "joint5", "joint6",
    "act_r_f_joint1_1", "act_r_f_joint1_2", "act_r_f_joint1_3", "act_r_f_joint1_4",
    "act_r_f_joint2_1", "act_r_f_joint2_2", "act_r_f_joint2_3", "act_r_f_joint2_4",
    "act_r_f_joint3_1", "act_r_f_joint3_2", "act_r_f_joint3_3", "act_r_f_joint3_4",
    "act_r_f_joint4_1", "act_r_f_joint4_2", "act_r_f_joint4_3", "act_r_f_joint4_4",
    "act_r_f_joint5_1", "act_r_f_joint5_2", "act_r_f_joint5_3", "act_r_f_joint5_4"
]

touch_sensor_names = [
    "touch_r_f_link1_pad", "touch_r_f_link2_pad", "touch_r_f_link3_pad",
    "touch_r_f_link4_pad", "touch_r_f_link5_pad"
]

proximity_sensor_names = [
    "rf1", "rf2", "rf3", "rf4", "rf5"
]



# === ROS 初始化 ===
rospy.init_node("ros_control_mujoco_dual_camera", anonymous=True)
ts_force_pub = rospy.Publisher("/ts_forces", Float32MultiArray, queue_size=10)
ts_prox_pub = rospy.Publisher("/ts_proximity", Float32MultiArray, queue_size=10)



# === 控制订阅器回调生成器 ===
def make_callback(act_name, act_index):
    def callback(msg):
        data.ctrl[act_index] = msg.data
    return callback

# === 注册订阅器 ===
for name in actuator_names:
    try:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        rospy.Subscriber(f"/{name}_cmd", Float64, make_callback(name, idx))
    except Exception:
        continue

# === 设置初始值 ===
pinch_cfg = get_pinch_config()["initial"]
for joint_name, value in pinch_cfg.items():
    try:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        data.ctrl[idx] = value
    except:
        continue

mujoco.mj_forward(model, data)

# === 发布触觉数据 ===
def publish_touch_data():
    values = []
    for name in touch_sensor_names:
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            values.append(data.sensordata[sid])
        except:
            values.append(0.0)
    ts_force_pub.publish(Float32MultiArray(data=values))

# === 发布接近觉数据 ===
def publish_prox_data():
    values = []
    for name in proximity_sensor_names:
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            values.append(data.sensordata[sid])
        except:
            values.append(0.0)
    ts_prox_pub.publish(Float32MultiArray(data=values))


# === 模拟推进线程 ===
'''def sim_loop():
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        mujoco.mj_step(model, data)
        #publish_touch_data()
        publish_prox_data()
        rate.sleep()

threading.Thread(target=sim_loop, daemon=True).start()'''

with mujoco.viewer.launch_passive(model, data) as viewer:
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        '''renderer.update_scene(data, camera=camera_rgb_id)
        img_rgb = renderer.render()
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 渲染手部摄像头
        renderer.update_scene(data, camera=camera_hand_id)
        img_hand = renderer.render()
        img_hand = cv2.cvtColor(img_hand, cv2.COLOR_RGB2BGR)

        # 拼接显示
        concat = np.hstack((img_rgb, img_hand))
        cv2.imshow("RGB (Left) | Hand (Right)", concat)

        if cv2.waitKey(1) == 27:
            break'''
        mujoco.mj_step(model, data)
        publish_touch_data()
        viewer.sync()
        rate.sleep()