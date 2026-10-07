# agents/feature_agents/event_agent.py
"""
事件级特征 Agent：棘波/尖波检测（还原原始信号 + 幅度检测）

输入：归一化后 EEG 信号 (n_channels, n_times)
输出：事件特征向量 (n_channels × 3 + 6) = 63 维 (for 19 channels)

每通道 3 个特征：
  1. spike_count        — 棘波计数
  2. sharp_wave_count   — 尖波计数
  3. max_event_amplitude — 最大事件相对幅度

全局 6 个特征：
  4. total_spike_count    — 总棘波数
  5. total_sharp_wave_count — 总尖波数
  6. max_amplitude_global — 全局最大事件幅度
  7. n_active_channels   — 有事件的通道数（灶性 vs 全导）
  8. lateralization_index — 侧化指数
  9. event_density        — 事件密度（个/秒）
"""

import numpy as np
from agents.base_agent import BaseAgent
from utils.event_detector import compute_event_features


class EventAgent(BaseAgent):
    """事件级特征 Agent（还原原始信号后做幅度检测）"""

    PER_CH_FEATURES = 4
    GLOBAL_FEATURES = 6
    N_CHANNELS = 19

    def __init__(self, name="event", sfreq=250, **kwargs):
        self.sfreq = sfreq
        super().__init__(name, **kwargs)

    def _preprocess(self, eeg_data, metadata=None):
        data = np.asarray(eeg_data, dtype=np.float64)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data[:self.N_CHANNELS, :]

    def _inference(self, processed_data):
        features, per_ch = compute_event_features(
            processed_data, sfreq=self.sfreq
        )
        return features

    def _postprocess(self, output):
        features = np.asarray(output, dtype=np.float64)
        return {
            "label": None,
            "prob": None,
            "confidence": None,
            "extra": {
                "features": features,
                "feature_dim": len(features),
                "feature_type": "event",
            }
        }

    @property
    def n_features(self):
        return self.N_CHANNELS * self.PER_CH_FEATURES + self.GLOBAL_FEATURES
