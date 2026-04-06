# MuJoCo 机械臂-灵巧手一体化抓取强化学习

这个扩展模块面向你的简历项目描述，补上了一个最小可运行的 RL 训练闭环：

- `dexrobot_mujoco/rl/grasp_env.py`：提供 `DexRobotGraspEnv`，复用已有的 RealmanRobot + DexHand 模型。
- `dexrobot_mujoco/rl/scene_builder.py`：每次 reset 时随机生成异形物体场景，支持 `box`、`thin_box`、`cylinder`、`capsule`、`ellipsoid`。
- `dexrobot_mujoco/rl/ppo.py`：提供基于 PyTorch 的 PPO Actor-Critic 实现，默认优先使用 CUDA。
- `scripts/train_grasp_ppo.py`：提供训练入口，支持 `--device auto/cpu/cuda`。

## 设计对应到项目职责

### 1. 联合控制仿真系统

- 手臂控制输出为末端位姿增量：`dx, dy, dz, d_rx, d_ry, d_rz`
- 灵巧手控制输出为 20 个手指关节增量
- 环境内部使用 MuJoCo Jacobian 做阻尼最小二乘 IK，把末端增量映射到 6 个机械臂关节目标
- 灵巧手关节使用位置控制器直接跟踪

### 2. PPO Actor-Critic

- 共享编码器提取观测特征
- Actor 输出联合动作均值
- Critic 估计状态价值
- 训练时使用 PPO clip objective + GAE
- 网络更新默认优先使用 GPU，MuJoCo 仿真仍主要运行在 CPU

### 3. 分阶段奖励函数

环境奖励由以下部分构成：

- `reach reward`：鼓励掌心接近目标物体
- `alignment reward`：鼓励掌心朝向与目标接近方向对齐
- `contact reward`：根据触觉 pad 激活数量鼓励多指稳定接触
- `stable grasp reward`：接触数量足够且物体速度较小时给奖励
- `lift reward`：物体高度超过桌面阈值后给奖励
- `success bonus`：连续稳定抬升若干步后给成功奖励
- `action smoothness penalty`：惩罚动作突变
- `collision penalty`：惩罚机器人与桌面等无效碰撞

## 运行步骤

先确认 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

自动优先 CUDA：

```bash
python scripts/train_grasp_ppo.py \
    --output-dir outputs/grasp_ppo \
    --total-steps 200000 \
    --rollout-steps 1024 \
    --episode-horizon 250 \
    --device auto
```

强制使用 CUDA：

```bash
python scripts/train_grasp_ppo.py --device cuda
```

强制退回 CPU：

```bash
python scripts/train_grasp_ppo.py --device cpu
```

## 后续建议

1. 先只训练立方体，再逐步加入圆柱、薄片和随机姿态。
2. 随机化摩擦、质量、尺寸和初始位姿，提升泛化能力。
3. 把 TaShan 触觉语义力引入奖励，做更细粒度的接触稳定性建模。
4. 增加并行环境，进一步发挥 GPU 上策略网络更新的效率。
