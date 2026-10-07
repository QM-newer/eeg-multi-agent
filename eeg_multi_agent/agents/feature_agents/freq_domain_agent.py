# agents/feature_agents/freq_domain_agent.py

import numpy as np
from scipy.signal import welch
from agents.base_agent import BaseAgent


class FreqDomainAgent(BaseAgent):
    """频域特征Agent：提取各频段功率谱特征

    每通道特征（共 8 个）：
        delta_relative, theta_relative, alpha_relative, beta_relative, gamma_relative,
        peak_freq, spectral_edge_95, spectral_entropy

    对齐 config.FEATURE_CONFIG["freq_domain"]["features"]：
        psd, band_power, peak_freq, spectral_edge
    """

    def __init__(self, name, sfreq=250, model_path=None, device="cpu"):
        self.sfreq = sfreq
        super().__init__(name, model_path, device)

    def _load_model(self):
        pass

    def _preprocess(self, eeg_data, metadata=None):
        # 兼容单通道一维输入
        if isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        return eeg_data

    @staticmethod
    def _peak_freq(freqs, psd):
        """峰值频率：PSD 最大值对应的频率"""
        if len(psd) == 0 or np.max(psd) <= 0:
            return 0.0
        return float(freqs[np.argmax(psd)])

    @staticmethod
    def _spectral_edge(freqs, psd, pct=0.95):
        """
        频谱边缘频率（Spectral Edge Frequency）：
        累积功率达到总功率 pct% 时的频率。
        常用 SEF95（95%），反映频谱的高频成分占比。
        """
        if len(psd) == 0 or np.sum(psd) <= 0:
            return 0.0
        cum_power = np.cumsum(psd)
        threshold = np.sum(psd) * pct
        idx = np.searchsorted(cum_power, threshold)
        idx = min(idx, len(freqs) - 1)
        return float(freqs[idx])

    @staticmethod
    def _spectral_entropy(psd):
        """频谱熵：PSD 归一化后的信息熵，反映频谱复杂度"""
        psd_sum = np.sum(psd)
        if psd_sum <= 0:
            return 0.0
        psd_norm = psd / psd_sum
        psd_norm = psd_norm[psd_norm > 0]
        return float(-np.sum(psd_norm * np.log(psd_norm)))

    def _inference(self, processed_data):
        data = processed_data
        # 五频段划分
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
        features = []
        for ch in range(data.shape[0]):
            sig = data[ch, :]
            # nperseg 不能超过信号长度
            nperseg = min(256, len(sig))
            freqs, psd = welch(sig, fs=self.sfreq, nperseg=nperseg)
            total_power = np.sum(psd)
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(psd[band_mask])
                # 相对功率
                features.append(band_power / total_power if total_power > 0 else 0)

            # 峰值频率
            features.append(self._peak_freq(freqs, psd))
            # 频谱边缘频率 (SEF95)
            features.append(self._spectral_edge(freqs, psd, pct=0.95))
            # 频谱熵
            features.append(self._spectral_entropy(psd))

        return np.array(features)

    def _postprocess(self, output):
        return {
            "label": None,
            "prob": None,
            "confidence": None,
            "extra": {
                "features": output,
                "feature_dim": len(output),
                "feature_type": "freq_domain"
            }
        }
