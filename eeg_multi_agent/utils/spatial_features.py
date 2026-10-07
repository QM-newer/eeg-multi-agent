# utils/spatial_features.py
"""
空间域补充特征（供分歧仲裁阶段使用）

对应文档 4.2(2)「仲裁流程第1步：通知特征工程组补充空间域特征
（通道相干性、功能连接）」。

内容：
1. 频带相干矩阵：FFT 单段互谱归一化得到通道间相干性（0-1）
2. 功能连接（AEC 近似）：频带内包络相关 / 相干强度
3. 网络统计摘要：各频带的连接强度统计 + 节点强度（每通道）

说明：
- 本模块属于「仲裁期补充特征」，不是第 10 个 Agent（文档明确 Agent 数为 9）
- 单段 FFT 相干 = 无分段平均的相干性估计，速度远快于逐对 welch，
  适合在仲裁阶段实时计算；若需更平滑估计可加大 n_times 或分段后平均。

用法：
    from utils.spatial_features import extract_spatial_features
    feats, info = extract_spatial_features(eeg_data, sfreq=250)
"""

import numpy as np

# 默认频带（与 config.FEATURE_CONFIG["freq_domain"]["bands"] 保持一致）
DEFAULT_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# 每个频带提取的统计量
BAND_STAT_NAMES = ["mean", "std", "p90", "density"]


def _band_coherence_matrix(data, sfreq, band, window="hann"):
    """
    计算指定频带内通道间相干矩阵（单段 FFT 互谱法）

    参数:
        data: 2D array, (n_channels, n_times)
        sfreq: 采样率
        band: (low, high) 频带边界（Hz）
        window: 窗函数

    返回:
        2D array, (n_channels, n_channels)，取值 0-1，对角线为 1
    """
    n_channels, n_times = data.shape
    freqs = np.fft.rfftfreq(n_times, d=1.0 / float(sfreq))

    if window == "hann":
        win = np.hanning(n_times)
    elif window == "hamming":
        win = np.hamming(n_times)
    else:
        win = np.ones(n_times)

    spectrum = np.fft.rfft(data * win[None, :], axis=1)      # (n_channels, n_freqs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):                                     # 频带内无频点（极短信号）
        mask = np.ones_like(freqs, dtype=bool)

    band_spec = spectrum[:, mask]
    norm = np.linalg.norm(band_spec, axis=1, keepdims=True) + 1e-12
    band_norm = band_spec / norm

    # 归一化互谱的模 = 相干性（0-1）
    coh = np.abs(band_norm @ band_norm.conj().T)
    coh = np.clip(coh, 0.0, 1.0)
    np.fill_diagonal(coh, 1.0)
    return coh


def _matrix_stats(matrix):
    """
    从连接矩阵提取摘要统计量（不含对角线自连接）

    返回:
        list: [mean, std, p90, density]
            density = 连接强度 > 0.5 的比例（强连接密度）
    """
    off = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if off.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(off)),
        float(np.std(off)),
        float(np.percentile(off, 90)),
        float(np.mean(off > 0.5)),
    ]


def extract_spatial_features(data, sfreq=250, bands=None, window="hann"):
    """
    提取空间域补充特征（通道相干性 / 功能连接摘要）

    参数:
        data: 2D array, (n_channels, n_times)
        sfreq: 采样率
        bands: dict，{band_name: (low, high)}，默认五个经典频带
        window: FFT 窗函数

    返回:
        (features, info):
            features: 1D array，空间域特征向量
            info: dict，含各频带相干矩阵、节点强度、特征名列表
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(f"期望输入为 2D (n_channels, n_times)，实际为 {data.shape}")

    bands = bands or DEFAULT_BANDS
    n_channels = data.shape[0]

    matrices, features, names = {}, [], []

    # 1) 各频带相干矩阵 + 4 个统计量
    for band_name, rng in bands.items():
        coh = _band_coherence_matrix(data, sfreq, rng, window=window)
        matrices[band_name] = coh
        for stat_name, value in zip(BAND_STAT_NAMES, _matrix_stats(coh)):
            features.append(value)
            names.append(f"spatial_{band_name}_{stat_name}")

    # 2) 宽带（0.5-45Hz）相干矩阵：统计量 + 节点强度
    broadband = _band_coherence_matrix(data, sfreq, (0.5, 45), window=window)
    matrices["broadband"] = broadband
    for stat_name, value in zip(BAND_STAT_NAMES, _matrix_stats(broadband)):
        features.append(value)
        names.append(f"spatial_broadband_{stat_name}")

    # 节点强度：每个通道与其余通道的平均连接强度（衡量该导联与全脑的同步性）
    node_strength = (broadband.sum(axis=1) - 1.0) / max(n_channels - 1, 1)
    for ch in range(n_channels):
        features.append(float(node_strength[ch]))
        names.append(f"spatial_node_strength_ch{ch}")

    info = {
        "matrices": matrices,
        "node_strength": node_strength,
        "feature_names": names,
        "feature_dim": len(features),
        "bands": bands,
    }
    return np.asarray(features, dtype=np.float64), info


def augment_features(all_features, spatial_features):
    """
    将空间域补充特征拼接到原多域特征之后（仲裁期扩充特征集）
    """
    if all_features is None:
        return np.asarray(spatial_features, dtype=np.float64)
    return np.concatenate([
        np.asarray(all_features, dtype=np.float64).reshape(-1),
        np.asarray(spatial_features, dtype=np.float64).reshape(-1),
    ])
