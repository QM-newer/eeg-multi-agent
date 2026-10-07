# utils/signal_utils.py
"""
信号处理工具函数
供特征Agent和数据预处理模块调用
"""

import numpy as np
from scipy import stats
from scipy.signal import welch


def compute_time_domain_features(signal):
    """
    计算单通道时域特征

    参数:
        signal: 1D array, 单通道信号

    返回:
        dict: 各时域特征
    """
    signal = np.asarray(signal)
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "var": np.var(signal),
        "skew": stats.skew(signal),
        "kurtosis": stats.kurtosis(signal),
        "max": np.max(signal),
        "min": np.min(signal),
        "peak_to_peak": np.max(signal) - np.min(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal)),
    }
    return features


def compute_freq_domain_features(signal, sfreq, nperseg=256):
    """
    计算单通道频域特征（五频段相对/绝对功率 + 频谱熵）

    参数:
        signal: 1D array, 单通道信号
        sfreq: float, 采样率
        nperseg: int, FFT窗口大小

    返回:
        dict: 各频段特征
    """
    signal = np.asarray(signal)
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    total_power = np.sum(psd)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    features = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[mask])
        features[f"{band_name}_relative"] = band_power / total_power if total_power > 0 else 0
        features[f"{band_name}_absolute"] = band_power

    # 频谱熵
    psd_norm = psd / (np.sum(psd) + 1e-10)
    features["spectral_entropy"] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    return features


def compute_signal_quality(signal):
    """
    简易信号质量评估

    参数:
        signal: 2D array, (n_channels, n_times)

    返回:
        float: 0-1的质量评分
    """
    signal = np.asarray(signal)
    n_channels = signal.shape[0]
    scores = []

    for ch in range(n_channels):
        sig = signal[ch, :]

        # 指标1：振幅范围（正常EEG在几十μV量级，太大说明有伪迹）
        amplitude = np.max(np.abs(sig))
        if amplitude < 100:
            amp_score = 1.0
        elif amplitude < 500:
            amp_score = max(0, 1.0 - (amplitude - 100) / 400)
        else:
            amp_score = 0.0

        # 指标2：方差稳定性（方差异常低可能是坏通道）
        var = np.var(sig)
        if var > 1:
            var_score = 1.0
        elif var > 0.1:
            var_score = var / 1.0
        else:
            var_score = 0.1

        scores.append(0.5 * amp_score + 0.5 * var_score)

    return float(np.mean(scores))


def estimate_snr(signal, sfreq=None, smooth_win=None):
    """
    估计EEG信噪比（dB）

    方法：平滑残差法
        信号分量 = 平滑后的低通版本（真实脑电在时间上平滑）
        噪声分量 = 原始信号 - 平滑信号（白噪声近似不被平滑）
        由于平滑会保留 1/w 的噪声能量，需做 (1 - 1/w) 校正

    参数:
        signal: 1D/2D array（2D 时为 (n_channels, n_times)）
        sfreq: 采样率，用于确定平滑窗口（默认按 250Hz 取 5 点）
        smooth_win: 平滑窗口长度（点），None 时自动取 sfreq/50

    返回:
        float: 各通道平均信噪比（dB），已裁剪到 [-10, 40]
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.shape[1] < 4:
        return -10.0

    if smooth_win is None:
        smooth_win = max(3, int(round((sfreq or 250) / 50.0)))
    smooth_win = int(min(smooth_win, signal.shape[1]))

    kernel = np.ones(smooth_win) / smooth_win
    snrs = []
    for ch in range(signal.shape[0]):
        x = signal[ch]
        if np.std(x) < 1e-9:                      # 平坦/掉线通道
            snrs.append(-10.0)
            continue
        smooth = np.convolve(x, kernel, mode="same")
        residual = x - smooth
        power_noise = np.var(residual) / (1.0 - 1.0 / smooth_win)
        power_noise = max(power_noise, 1e-12)
        power_signal = max(np.var(smooth), 1e-12)
        snrs.append(10.0 * np.log10(power_signal / power_noise))

    if not snrs:
        return -10.0
    return float(np.clip(np.mean(snrs), -10.0, 40.0))


def compute_artifact_ratio(signal, sfreq=None, amp_threshold=100.0, flat_std_threshold=0.5):
    """
    估计伪迹污染比例（0-1，越大越脏）

    三个子指标加权：
    - 幅度伪迹：|x| > amp_threshold(µV) 的采样点占比（眨眼/肌电大幅波动）
    - 突变伪迹：一阶差分超过 8 倍中位绝对差分的采样点占比（尖峰/肌电）
    - 平坦通道：标准差过小的通道占比（电极脱落/掉线）

    参数:
        signal: 1D/2D array
        amp_threshold: 幅度阈值（µV）
        flat_std_threshold: 判为坏通道的标准差阈值

    返回:
        float: 伪迹比例，0-1
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    n_channels, n_times = signal.shape
    if n_times < 4:
        return 1.0

    amp_ratios, jump_ratios, flat_flags = [], [], []
    for ch in range(n_channels):
        x = signal[ch]
        amp_ratios.append(float(np.mean(np.abs(x) > amp_threshold)))

        diff = np.abs(np.diff(x))
        mad = np.median(diff) + 1e-9
        jump_ratios.append(float(np.mean(diff > 8.0 * mad)))

        flat_flags.append(1.0 if np.std(x) < flat_std_threshold else 0.0)

    amp_ratio = float(np.mean(amp_ratios))
    jump_ratio = float(np.mean(jump_ratios))
    flat_ratio = float(np.mean(flat_flags))

    artifact = 0.6 * amp_ratio + 0.2 * jump_ratio + 0.2 * flat_ratio
    return float(np.clip(artifact, 0.0, 1.0))


def compute_quality_report(signal, sfreq=None, config=None):
    """
    综合信号质量报告：SNR + 伪迹比例 + 质量分 + 质量等级

    质量分（0-1）用于与旧版 0-1 质量接口兼容：
        quality_score = 0.7 * clip(SNR / 20, 0, 1) + 0.3 * (1 - artifact_ratio)

    质量等级（三档，对应文档 4.2(3)）：
        high   : SNR > snr_high（默认 10 dB）
        medium : snr_low <= SNR <= snr_high（默认 5-10 dB）
        low    : SNR < snr_low（默认 5 dB）或伪迹比例超过阈值

    参数:
        signal: 2D array, (n_channels, n_times)
        sfreq: 采样率
        config: dict，覆盖默认阈值（见 config.QUALITY_CONFIG）

    返回:
        dict: {snr_db, artifact_ratio, quality_score, quality_level}
    """
    cfg = config or {}
    snr_high = cfg.get("snr_high", 10.0)
    snr_low = cfg.get("snr_low", 5.0)
    artifact_threshold = cfg.get("artifact_threshold", 0.30)
    amp_threshold = cfg.get("amp_threshold", 100.0)
    flat_std_threshold = cfg.get("flat_std_threshold", 0.5)

    snr_db = estimate_snr(signal, sfreq=sfreq)
    artifact_ratio = compute_artifact_ratio(
        signal, sfreq=sfreq,
        amp_threshold=amp_threshold,
        flat_std_threshold=flat_std_threshold,
    )

    if snr_db > snr_high and artifact_ratio <= artifact_threshold:
        level = "high"
    elif snr_db < snr_low or artifact_ratio > artifact_threshold:
        level = "low"
    else:
        level = "medium"

    snr_score = float(np.clip(snr_db / 20.0, 0.0, 1.0))
    quality_score = float(np.clip(0.7 * snr_score + 0.3 * (1.0 - artifact_ratio), 0.0, 1.0))

    return {
        "snr_db": float(snr_db),
        "artifact_ratio": float(artifact_ratio),
        "quality_score": quality_score,
        "quality_level": level,
    }


def bandpass_filter(signal, sfreq, low_freq=0.5, high_freq=45):
    """
    简易带通滤波（用scipy，后续可替换为MNE）
    支持一维信号或二维 (n_channels, n_times)
    """
    from scipy.signal import butter, filtfilt
    signal = np.asarray(signal)
    nyq = 0.5 * sfreq
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(4, [low, high], btype="band")
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return filtfilt(b, a, signal, axis=-1)


def resample_signal(signal, original_sfreq, target_sfreq):
    """
    信号降采样
    """
    from scipy.signal import resample
    signal = np.asarray(signal)
    n_samples = int(signal.shape[-1] * target_sfreq / original_sfreq)
    return resample(signal, n_samples, axis=-1)
