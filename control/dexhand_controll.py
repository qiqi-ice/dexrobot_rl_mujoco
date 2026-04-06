#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64

# ✅ actuator 名称列表（用于创建 publishers）
joint_names = [
    "act_ARTx", "act_ARTy", "act_ARTz",
    "act_ARRx", "act_ARRy", "act_ARRz",
    "act_r_f_joint1_1", "act_r_f_joint1_2", "act_r_f_joint1_3", "act_r_f_joint1_4",
    "act_r_f_joint2_1", "act_r_f_joint2_2", "act_r_f_joint2_3", "act_r_f_joint2_4",
    "act_r_f_joint3_1", "act_r_f_joint3_2", "act_r_f_joint3_3", "act_r_f_joint3_4",
    "act_r_f_joint4_1", "act_r_f_joint4_2", "act_r_f_joint4_3", "act_r_f_joint4_4",
    "act_r_f_joint5_1", "act_r_f_joint5_2", "act_r_f_joint5_3", "act_r_f_joint5_4"
]

# ✅ 初始位置
initial = {
    "act_ARTx": -0.16,
    "act_ARTy": 0,
    "act_ARTz": -0.3,
    "act_ARRx":0,
    "act_ARRy":0, 
    "act_ARRz":0,
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
}

# ✅ 抓取时的增量变化
pinch_deltas = {
    "act_ARTz": -0.05,
    "act_r_f_joint1_2": 0.1,
    "act_r_f_joint1_3": 0.1,
    "act_r_f_joint1_4": 0.1,
    "act_r_f_joint2_2": 1.15,
    "act_r_f_joint2_3": 0.4,
    "act_r_f_joint2_4": 0.4,
    "act_r_f_joint3_2": 1.15,
    "act_r_f_joint3_3": 0.4,
    "act_r_f_joint3_4": 0.4,
    "act_r_f_joint4_2": 1.15,
    "act_r_f_joint4_3": 0.4,
    "act_r_f_joint4_4": 0.4,
    "act_r_f_joint5_2": 1.15,
    "act_r_f_joint5_3": 0.4,
    "act_r_f_joint5_4": 0.4,
}

# ✅ 抬升末端执行器
pinch_deltas1 = {
    "act_ARTz": 0.1
}



# ✅ 旋转，末端执行器
pinch_deltas2 = {
    "act_ARRy": -1.5,
    "act_r_f_joint2_2": 1.1,
    "act_r_f_joint3_2": 1.1,
    "act_r_f_joint4_2": 1.1,
    "act_r_f_joint5_2": 1.1,
}


if __name__ == "__main__":
    rospy.init_node("init_and_pinch_publisher")
    publishers = {}

    # ✅ 创建所有 publishers
    for joint in joint_names:
        topic_name = f"/{joint}_cmd"
        publishers[joint] = rospy.Publisher(topic_name, Float64, queue_size=10)

    rospy.sleep(5.0)  # 等待发布器初始化

    # ✅ 发布初始值
    rospy.loginfo("📤 Publishing initial joint values...")
    for joint, value in initial.items():
        if joint in publishers:
            publishers[joint].publish(Float64(value))
            rospy.loginfo(f"→ {joint} = {value:.3f}")

    rospy.sleep(5.0)  # 等待就位

    # ✅ 发布抓取动作
    rospy.loginfo("✊ Publishing pinch deltas...")
    for joint, delta in pinch_deltas.items():
        if joint in publishers and joint in initial:
            new_value = initial[joint] + delta
            publishers[joint].publish(Float64(new_value))
            rospy.loginfo(f"→ {joint} = {new_value:.3f} (pinch)")

    rospy.sleep(5.0)

    # ✅ 发布末端抬升动作
    rospy.loginfo("📈 Publishing ARTz delta...")
    for joint, delta in pinch_deltas1.items():
        if joint in publishers and joint in initial:
            new_value = initial[joint] + delta
            publishers[joint].publish(Float64(new_value))
            rospy.loginfo(f"→ {joint} = {new_value:.3f} (lift)")

    rospy.sleep(5.0)

    # ✅ 发布末端抬升动作
    rospy.loginfo("📈 Publishing ARRy delta...")
    for joint, delta in pinch_deltas2.items():
        if joint in publishers and joint in initial:
            new_value = initial[joint] + delta
            publishers[joint].publish(Float64(new_value))
            rospy.loginfo(f"→ {joint} = {new_value:.3f} (lift)")