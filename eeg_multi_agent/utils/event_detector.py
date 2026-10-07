# utils/event_detector.py
"""
事件级特征提取器 — 聚合形态学指标（v5 优化版）

优化：预计算 SOS 滤波系数，避免每个通道重复调用 butter()

特征维度：per_channel(4) × 19 + global(6) = 82
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

# 预计算 SOS 系数（避免每次调用 butter）
_SOS_CACHE = {}

def _get_sos(sfreq, low, high, order=4):
    key = (sfreq, low, high, order)
    if key not in _SOS_CACHE:
        nyq = sfreq / 2.0
        lo = max(low / nyq, 0.01)
        hi = min(high / nyq, 0.99)
        if hi <= lo:
            _SOS_CACHE[key] = None
        else:
            _SOS_CACHE[key] = butter(order, [lo, hi], btype="band", output="sos")
    return _SOS_CACHE[key]


def compute_event_features(eeg_data, sfreq=250):
    n_channels, n_times = eeg_data.shape
    per_ch_feats = 4
    global_feats = 6
    features = np.zeros(n_channels * per_ch_feats + global_feats, dtype=np.float64)

    left_idx = {0, 2, 4, 6, 8, 10, 12, 14}
    right_idx = {1, 3, 5, 7, 9, 11, 13, 15}

    # 预取 SOS
    sos_wide = _get_sos(sfreq, 1.0, 30.0)
    sos_low = _get_sos(sfreq, 1.0, 12.0)
    sos_high = _get_sos(sfreq, 13.0, 30.0)

    kurt_per_ch = np.zeros(n_channels)
    peak_rms_per_ch = np.zeros(n_channels)
    left_kurt, right_kurt = [], []

    # 向量化：所有通道一起滤波（更快）
    # wide band (1-30 Hz)
    if sos_wide is not None:
        wide = np.vstack([sosfiltfilt(sos_wide, eeg_data[ch]) for ch in range(n_channels)])
    else:
        wide = eeg_data.copy()

    # low band (1-12 Hz)
    if sos_low is not None:
        low_band = np.vstack([sosfiltfilt(sos_low, eeg_data[ch]) for ch in range(n_channels)])
    else:
        low_band = eeg_data.copy()

    # high band (13-30 Hz)
    if sos_high is not None:
        high_band = np.vstack([sosfiltfilt(sos_high, eeg_data[ch]) for ch in range(n_channels)])
    else:
        high_band = np.zeros_like(eeg_data)

    for ch in range(n_channels):
        filt = wide[ch]

        # 1. 峰度
        bp_kurt = kurtosis(filt, fisher=True, bias=False)
        if np.isnan(bp_kurt):
            bp_kurt = 0.0

        # 2. peak-to-RMS
        rms = np.sqrt(np.mean(filt ** 2))
        p2r = np.max(np.abs(filt)) / max(rms, 1e-10)

        # 3. 高/低频功率比
        hf_ratio = np.mean(high_band[ch] ** 2) / max(np.mean(low_band[ch] ** 2), 1e-10)

        # 4. 负峰不对称度
        neg_e = np.mean(np.minimum(filt, 0) ** 2)
        pos_e = np.mean(np.maximum(filt, 0) ** 2)
        neg_asym = (neg_e - pos_e) / max(neg_e + pos_e, 1e-10)

        base = ch * per_ch_feats
        features[base] = bp_kurt
        features[base + 1] = p2r
        features[base + 2] = hf_ratio
        features[base + 3] = neg_asym

        kurt_per_ch[ch] = bp_kurt
        peak_rms_per_ch[ch] = p2r
        if ch in left_idx:
            left_kurt.append(bp_kurt)
        elif ch in right_idx:
            right_kurt.append(bp_kurt)

    g_base = n_channels * per_ch_feats
    features[g_base] = np.sum(kurt_per_ch)
    features[g_base + 1] = np.max(kurt_per_ch)
    features[g_base + 2] = np.std(kurt_per_ch)
    features[g_base + 3] = np.max(peak_rms_per_ch)

    if left_kurt and right_kurt:
        lat = (np.mean(left_kurt) - np.mean(right_kurt)) / (np.mean(left_kurt) + np.mean(right_kurt) + 1e-8)
    else:
        lat = 0.0
    features[g_base + 4] = lat

    if n_channels >= 8:
        sub = eeg_data[:16, :]
        corr_mat = np.corrcoef(sub)
        np.fill_diagonal(corr_mat, 0)
        max_corr = np.max(np.abs(corr_mat))
    else:
        max_corr = 0.0
    features[g_base + 5] = max_corr

    return features, per_ch_feats
