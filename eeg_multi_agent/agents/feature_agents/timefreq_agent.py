# agents/feature_agents/timefreq_agent.py

import numpy as np
from agents.base_agent import BaseAgent


class TimeFreqAgent(BaseAgent):
    """时频域特征Agent：多尺度能量特征（简化版，后续替换为CWT）"""

    def _load_model(self):
        pass

    def _preprocess(self, eeg_data, metadata=None):
        # 兼容单通道一维输入
        if isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        return eeg_data

    def _inference(self, processed_data):
        data = processed_data
        features = []
        for ch in range(data.shape[0]):
            sig = data[ch, :]
            # 简化：用不同长度窗口的能量近似时频特征
            for window_len in [50, 100, 200, 500]:
                if len(sig) >= window_len:
                    # 滑动窗口能量
                    n_windows = len(sig) // window_len
                    window_energies = []
                    for i in range(n_windows):
                        segment = sig[i * window_len:(i + 1) * window_len]
                        window_energies.append(np.sum(segment ** 2))
                    features.append(np.mean(window_energies))
                    features.append(np.std(window_energies))
                else:
                    features.extend([0.0, 0.0])
        return np.array(features)

    def _postprocess(self, output):
        return {
            "label": None,
            "prob": None,
            "confidence": None,
            "extra": {
                "features": output,
                "feature_dim": len(output),
                "feature_type": "timefreq"
            }
        }
