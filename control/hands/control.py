#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import time
from hands.cyberglove import CyberGlove
from hands.Fingermap import DexhandFingerMapper

# 映射关系（CyberGlove映射 → MuJoCo虚拟手部的 actuator）
joint_mapping = {
    0: "act_r_f_joint5_3",  # 小指中间
    1: "act_r_f_joint5_2",  # 小指底部
    
    2: "act_r_f_joint4_3",  # 无名指中间
    3: "act_r_f_joint4_2",  # 无名指底部
    
    4: "act_r_f_joint3_3",  # 中指中间
    5: "act_r_f_joint3_2",  # 中指底部
    
    6: "act_r_f_joint2_3",  # 食指中间
    7: "act_r_f_joint2_2",  # 食指底部
    
    8: "act_r_f_joint1_2",  # 拇指底部弯曲
    9: "act_r_f_joint1_3",  # 拇指弯曲
    
    10: "act_r_f_joint1_1",  # 拇指对掌（旋转）
}

# 初始化 ROS 节点
rospy.init_node("cyberglove_to_mujoco", anonymous=True)

# 创建 ROS publishers
publishers = {}
for _, joint_name in joint_mapping.items():
    topic = f"/{joint_name}_cmd"
    publishers[joint_name] = rospy.Publisher(topic, Float64, queue_size=10)

# 初始化 CyberGlove 和 FingerMapper
glove = CyberGlove()  # 修改为你的串口设备
mapper = DexhandFingerMapper(window_size=3, motion_threshold=10)

# 校准（建议让手张开保持不动几秒）
rospy.loginfo("✋ 请张开手进行校准...")
samples = []
for _ in range(30):
    raw = glove.read()
    samples.append(raw)
    time.sleep(0.05)
mapper.calibrate(samples)
rospy.loginfo("✅ 校准完成")

rate = rospy.Rate(30)  # 控制频率
while not rospy.is_shutdown():
    raw = glove.read()
    mapped = mapper.map_glove_to_dexhand(raw)
    for i, val in enumerate(mapped):
        joint = joint_mapping[i]
        angle = val 
        publishers[joint].publish(Float64(angle))
        rospy.loginfo_throttle(1.0, f"[{joint}] ← {angle:.3f}")
    rate.sleep()
