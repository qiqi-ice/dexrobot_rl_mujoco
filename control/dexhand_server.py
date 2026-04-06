#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64, Float32MultiArray
import mujoco
import mujoco.viewer

# === 加载模型 ===
model = mujoco.MjModel.from_xml_path("/home/star/dzq/paper/dexrobot_mujoco/dexrobot_mujoco/scenes/box_realmanRobot.xml")
#model = mujoco.MjModel.from_xml_path("dexrobot_mujoco/scenes/box_piper.xml")


data = mujoco.MjData(model)

# ✅ 初始手势配置
def get_pinch_config():
    return {
        "initial": {
            "act_ARTx": -0.16,
            "act_ARTy": 0,
            "act_ARTz": -0.3,
            "act_ARRx": 0,
            "act_ARRy": 0,
            "act_ARRz": 0,
            "act_r_f_joint1_2": 0.25,
            "act_r_f_joint1_3": 0.35,
            "act_r_f_joint1_4": 0.25,
            "act_r_f_joint2_2": 0,
            "act_r_f_joint2_3": 0,
            "act_r_f_joint2_4": 0,
            "act_r_f_joint1_1": 1.50,
            "act_r_f_joint3_2": 0,
            "act_r_f_joint3_3": 0,
            "act_r_f_joint3_4": 0,
            "act_r_f_joint4_2": 0,
            "act_r_f_joint4_3": 0,
            "act_r_f_joint4_4": 0,
            "act_r_f_joint5_2": 0,
            "act_r_f_joint5_3": 0,
            "act_r_f_joint5_4": 0,
        },
    }

# === Actuator 名称（与 XML 保持一致）
actuator_names = [
    "act_ARTx", "act_ARTy", "act_ARTz",
    "act_ARRx", "act_ARRy", "act_ARRz",
    "act_r_f_joint1_1", "act_r_f_joint1_2", "act_r_f_joint1_3", "act_r_f_joint1_4",
    "act_r_f_joint2_1", "act_r_f_joint2_2", "act_r_f_joint2_3", "act_r_f_joint2_4",
    "act_r_f_joint3_1", "act_r_f_joint3_2", "act_r_f_joint3_3", "act_r_f_joint3_4",
    "act_r_f_joint4_1", "act_r_f_joint4_2", "act_r_f_joint4_3", "act_r_f_joint4_4",
    "act_r_f_joint5_1", "act_r_f_joint5_2", "act_r_f_joint5_3", "act_r_f_joint5_4"
]

# ✅ 触觉传感器名称（与 XML 保持一致）
touch_sensor_names = [
    "touch_r_f_link1_pad",
    "touch_r_f_link2_pad",
    "touch_r_f_link3_pad",
    "touch_r_f_link4_pad",
    "touch_r_f_link5_pad"
]

# === ROS 节点初始化
rospy.init_node("ros_control_mujoco_explicit", anonymous=True)

# ✅ 创建 ROS 发布器：用于触觉
ts_force_pub = rospy.Publisher("/ts_forces", Float32MultiArray, queue_size=10)

# === 控制订阅器回调生成器
def make_callback(act_name, act_index):
    def callback(msg):
        data.ctrl[act_index] = msg.data
        rospy.loginfo_throttle(1.0, f"[SET] {act_name} = {msg.data:.3f}")
    return callback

# === 注册 actuator 控制订阅器
for name in actuator_names:
    try:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        rospy.Subscriber(f"/{name}_cmd", Float64, make_callback(name, idx))
        rospy.loginfo(f"[SUB] Listening: /{name}_cmd → ctrl[{idx}]")
    except Exception:
        rospy.logwarn(f"[SUB] Actuator {name} not found, skipping...")

# ✅ 应用初始配置
pinch_cfg = get_pinch_config()["initial"]
for joint_name, value in pinch_cfg.items():
    try:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        data.ctrl[idx] = value
        rospy.loginfo(f"[INIT] {joint_name} ← {value}")
    except Exception:
        rospy.logwarn(f"[INIT] {joint_name} not found, skipping...")

mujoco.mj_forward(model, data)

# ✅ 每帧发布触觉数据
def publish_touch_data():
    values = []
    for name in touch_sensor_names:
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            values.append(data.sensordata[sid])
        except:
            values.append(0.0)
    ts_force_pub.publish(Float32MultiArray(data=values))

# === 启动可视化与仿真主循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        mujoco.mj_step(model, data)
        publish_touch_data()
        viewer.sync()
        rate.sleep()
