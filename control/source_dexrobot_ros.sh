#!/bin/bash


# 1. 设置 ROS 1 Noetic 环境
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
    echo "✅ 已加载 ROS Noetic 环境"
else
    echo "❌ 未找到 /opt/ros/noetic/setup.bash"
    return 1
fi

# 2. 设置项目目录为当前目录（即 dexrobot）
DEXROBOT_DIR=$(pwd)
export PYTHONPATH=$DEXROBOT_DIR:$PYTHONPATH
echo "📂 设置 PYTHONPATH=$PYTHONPATH"

# 3. 提示成功
echo "🎯 当前工作目录: $DEXROBOT_DIR"
echo "💡 你现在可以运行如下命令启动 ROS 节点"


