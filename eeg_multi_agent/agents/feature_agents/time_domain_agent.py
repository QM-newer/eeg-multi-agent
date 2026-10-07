# agents/feature_agents/time_domain_agent.py

import numpy as np
from scipy import stats
from agents.base_agent import BaseAgent


class TimeDomainAgent(BaseAgent):
    """时域特征Agent：提取各通道时域统计特征

    每通道特征（共 12 个）：
        mean, std, skew, kurtosis, rms, peak_to_peak,
        zc_rate, hjorth_activity, hjorth_mobility, hjorth_complexity,
        max_abs, var

    对齐 config.FEATURE_CONFIG["time_domain"]["features"]：
        mean, std, var, rms, peak_to_peak, skewness, kurtosis,
        zc_rate, hjd(Hjorth 3参数), band_power_ratio(留给频域Agent)
    """

    def _load_model(self):
        pass  # 传统特征提取不需要模型

    def _preprocess(self, eeg_data, metadata=None):
        # 直接用原始数据；兼容单通道一维输入
        if isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        return eeg_data  # shape: (n_channels, n_times)

    @staticmethod
    def _hjorth_params(signal):
        """
        Hjorth 参数（3 个）：
        - activity: 信号方差（= var，与上面 var 重复，但保留作为 Hjorth 体系的一部分）
        - mobility:  一阶差分标准差 / 信号标准差（频率的估计）
        - complexity: 二阶差分 mobility / 一阶差分 mobility（波形复杂度）
        """
        sig = np.asarray(signal, dtype=np.float64)
        d1 = np.diff(sig)
        d2 = np.diff(d1)
        var_sig = np.var(sig)
        var_d1 = np.var(d1) if len(d1) > 0 else 0.0
        var_d2 = np.var(d2) if len(d2) > 0 else 0.0
        activity = var_sig
        mobility = np.sqrt(var_d1 / max(var_sig, 1e-12))
        complexity = np.sqrt(var_d2 / max(var_d1, 1e-12)) / max(mobility, 1e-12)
        return activity, mobility, complexity

    def _inference(self, processed_data):
        data = processed_data
        features = []
        for ch in range(data.shape[0]):
            sig = data[ch, :]
            # Hjorth 参数
            h_act, h_mob, h_comp = self._hjorth_params(sig)
            features.extend([
                np.mean(sig),                   # 均值
                np.std(sig),                    # 标准差
                stats.skew(sig),                # 偏度
                stats.kurtosis(sig),            # 峰度（Fisher 定义，正态分布为 0）
                np.sqrt(np.mean(sig ** 2)),     # RMS（均方根）
                np.max(sig) - np.min(sig),      # peak_to_peak（峰峰值）
                np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * len(sig)),  # 过零率
                h_act,                          # Hjorth activity
                h_mob,                          # Hjorth mobility
                h_comp,                         # Hjorth complexity
                np.max(np.abs(sig)),            # 最大绝对值
                np.var(sig),                    # 方差
            ])
        return np.array(features)

    def _postprocess(self, output):
        return {
            "label": None,
            "prob": None,
            "confidence": None,
            "extra": {
                "features": output,
                "feature_dim": len(output),
                "feature_type": "time_domain"
            }
        }
