import numpy as np
from collections import deque

class DexhandFingerMapper:
    def __init__(self, window_size=3, motion_threshold=10):
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self.buffers = [deque(maxlen=window_size) for _ in range(11)]
        self.baseline = np.zeros(11)
        self.gain = np.array([10,10,10,10,10,10,10,10,10,10,10])  # 可调
        #self.gain = np.array([12]*11)  # 可调
        self.last_output = np.zeros(11)

    def extract_glove_features(self, raw):
        """提取CyberGlove的10个关键通道"""
        return np.array([
            raw[13],  # 小指中间
            raw[12],  # 小指底部
            raw[10],  # 无名指中间
            raw[9],  # 无名指底部
            raw[7],   # 中指中间
            raw[6],   # 中指底部
            raw[5],   # 食指中间
            raw[4],   # 食指底部
            raw[2],
            raw[1],   # 拇指弯曲
            raw[0]    # 拇指对掌（旋转）
        ])

    def calibrate(self, raw_samples):
        """提供一组张开手的原始数据用于baseline校准"""
        extracted = [self.extract_glove_features(r) for r in raw_samples]
        self.baseline = np.mean(extracted, axis=0)

    def map_glove_to_dexhand(self, raw):
        """映射并滤波"""
        raw_vals = self.extract_glove_features(raw)
        diff = raw_vals - self.baseline
        scaled = diff * self.gain

        # 滤波
        for i in range(11):
            self.buffers[i].append(scaled[i])
        smoothed = np.array([np.mean(buf) for buf in self.buffers])

        # Clip & Normalize to 0~1.5 弧度范围（或其它）
        output = []
        for i in range(11):
            if i==10:
                val = smoothed[i]
                if abs(val - self.last_output[i]) >= self.motion_threshold:
                    angle = np.clip(val / 1000.0 * 2.2, 0, 2.2)
                    self.last_output[i] = val
                else:
                    angle = np.clip(self.last_output[i] / 1000.0 * 2.2, 0, 2.2)
                output.append(angle)
            else:
                val = smoothed[i]
                if abs(val - self.last_output[i]) >= self.motion_threshold:
                    angle = np.clip(val / 1000.0 * 1.3, 0, 1.3)
                    self.last_output[i] = val
                else:
                    angle = np.clip(self.last_output[i] / 1000.0 * 1.3, 0, 1.3)
                output.append(angle)

        return output
