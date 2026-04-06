#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray

def proximity_callback(msg):
    rospy.loginfo(f"接收到接近觉数据: {msg.data}")

def listener():
    rospy.init_node('proximity_listener', anonymous=True)
    rospy.Subscriber("/ts_proximity", Float32MultiArray, proximity_callback)
    rospy.loginfo("开始监听 /ts_proximity 话题 ...")
    rospy.spin()

if __name__ == '__main__':
    listener()
