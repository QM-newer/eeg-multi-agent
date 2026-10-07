# data/preprocessor.py
"""
EEG数据预处理流水线
"""

import numpy as np
from utils.signal_utils import bandpass_filter, resample_signal, compute_signal_quality


class EEGPreprocessor:
    """
    EEG预处理流水线

    处理步骤：
    1. 带通滤波
    2. 降采样
    3. 重参考
    4. 伪迹检测（后续完善）
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.low_freq = self.config.get("low_freq", 0.5)
        self.high_freq = self.config.get("high_freq", 45)
        self.target_sfreq = self.config.get("target_sfreq", 250)

    def filter(self, data, sfreq):
        """带通滤波"""
        return bandpass_filter(data, sfreq, self.low_freq, self.high_freq)

    def resample(self, data, original_sfreq):
        """降采样"""
        return resample_signal(data, original_sfreq, self.target_sfreq)

    def rereference(self, data, ref_type="average"):
        """重参考"""
        if ref_type == "average":
            ref = np.mean(data, axis=0, keepdims=True)
            return data - ref
        else:
            return data

    def assess_quality(self, data):
        """信号质量评估"""
        return compute_signal_quality(data)

    def run(self, data, sfreq):
        """
        运行完整预处理流水线

        参数:
            data: 2D array, (n_channels, n_times)
            sfreq: float, 原始采样率

        返回:
            dict: {data, sfreq, quality_score}
        """
        # 1. 带通滤波
        data = self.filter(data, sfreq)

        # 2. 降采样
        if sfreq != self.target_sfreq:
            data = self.resample(data, sfreq)
            current_sfreq = self.target_sfreq
        else:
            current_sfreq = sfreq

        # 3. 重参考（平均参考）
        data = self.rereference(data)

        # 4. 质量评估
        quality_score = self.assess_quality(data)

        return {
            "data": data,
            "sfreq": current_sfreq,
            "quality_score": quality_score,
        }
