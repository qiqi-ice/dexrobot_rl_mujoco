#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64

# ✅ actuator 名称列表（用于创建 publishers）
joint_names = [
    "joint1", "joint2", "joint3","joint4", "joint5", "joint6",
    "act_r_f_joint1_1", "act_r_f_joint1_2", "act_r_f_joint1_3", "act_r_f_joint1_4",
    "act_r_f_joint2_1", "act_r_f_joint2_2", "act_r_f_joint2_3", "act_r_f_joint2_4",
    "act_r_f_joint3_1", "act_r_f_joint3_2", "act_r_f_joint3_3", "act_r_f_joint3_4",
    "act_r_f_joint4_1", "act_r_f_joint4_2", "act_r_f_joint4_3", "act_r_f_joint4_4",
    "act_r_f_joint5_1", "act_r_f_joint5_2", "act_r_f_joint5_3", "act_r_f_joint5_4"
]

# ✅ 初始位置
initial = {
    "joint1": 0, "joint2": 0, "joint3": 0,
    "joint4": 0, "joint5": 0, "joint6": 0,
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
pinch_deltas_realmanRobot=[{"joint3":0.5,"joint5":0.55},{ "joint2":0.85,},
{ "joint2":1.05,
    "act_r_f_joint1_2": 0.1,
    "act_r_f_joint1_3": 0.1,
    "act_r_f_joint1_4": 0.1,
    "act_r_f_joint2_2": 0.6,
    "act_r_f_joint2_3": 0.4,
    "act_r_f_joint2_4": 0.4,
    "act_r_f_joint3_2": 0.6,
    "act_r_f_joint3_3": 0.4,
    "act_r_f_joint3_4": 0.4,
    "act_r_f_joint4_2": 0.6,
    "act_r_f_joint4_3": 0.4,
    "act_r_f_joint4_4": 0.4,
    "act_r_f_joint5_2": 0.6,
    "act_r_f_joint5_3": 0.4,
    "act_r_f_joint5_4": 0.4,
},
{
    "joint2":1.1,
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
},{"joint2":0.85}
]


pinch_deltas_piper = [{  "joint3": -1.91,   "joint2": 1.85,}, {"joint2":2.1,},
{
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
},
{
    "joint2": 1.7,
}]

if __name__ == "__main__":
    rospy.init_node("init_and_pinch_publisher")
    publishers = {}

    # ✅ 创建所有 publishers
    for joint in joint_names:
        topic_name = f"/{joint}_cmd"
        publishers[joint] = rospy.Publisher(topic_name, Float64, queue_size=10)

    rospy.sleep(2.0)  

    rospy.loginfo("初始位置设置")
    for joint, value in initial.items():
        if joint in publishers:
            publishers[joint].publish(Float64(value))

    rospy.sleep(2.0)  # 等待就位
    x = len(pinch_deltas_realmanRobot)
    for i in range(x):
        rospy.loginfo(f"进行第{i+1}步抓取")
        for joint, delta in pinch_deltas_realmanRobot[i].items():
            if joint in publishers and joint in initial:
                new_value = initial[joint] + delta
                publishers[joint].publish(Float64(new_value))
        rospy.sleep(4.0)
    rospy.loginfo("抓取完成")