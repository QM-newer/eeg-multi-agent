# data/simulator.py
"""
合成EEG数据生成器（真实数据到位前的替代数据源）

用途：在 F 盘真实EEG不可用时，为训练 / 对比实验 / 消融实验提供可控、
可复现、类别可分的三分类数据；真实数据接入后只需替换数据加载器，
下游训练与评估代码无需改动。

类别设计（对应 config.CLASS_NAMES = [Normal, Borderline, Abnormal]）：
    Normal(0)    ：alpha 节律主导，慢波少，无棘波
    Borderline(1)：theta 增多、alpha 减弱，处于界限状态
    Abnormal(2)  ：delta/theta 强、alpha 受抑，伴棘波与全导同步慢波

生成流程：
    1/f 背景 + 各频带窄带振荡（FFT 频域掩码生成）× 空间梯度权重
    + 棘波瞬态（Abnormal）+ 个体偏置 + 白噪声（按目标 SNR 缩放）+ 随机伪迹

用法：
    from data.simulator import generate_dataset
    X, y, subject_ids = generate_dataset(n_subjects=60, epochs_per_subject=4)
"""

import numpy as np

from config import N_CLASSES

# 五个频带的相对幅度（不同类别的差异主要体现于此）
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

CLASS_PROFILES = {
    0: {"delta": 0.4, "theta": 0.5, "alpha": 1.6, "beta": 0.7, "gamma": 0.3, "spike": 0.0},
    1: {"delta": 0.8, "theta": 1.3, "alpha": 0.9, "beta": 0.6, "gamma": 0.3, "spike": 0.0},
    2: {"delta": 1.8, "theta": 1.6, "alpha": 0.5, "beta": 0.4, "gamma": 0.2, "spike": 0.8},
}


def _band_noise(rng, shape, sfreq, band):
    """生成指定频带内的窄带随机振荡（FFT 频域掩码，已标准化到单位标准差）"""
    white = rng.standard_normal(shape)
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(shape[-1], d=1.0 / sfreq)
    mask = ((freqs >= band[0]) & (freqs <= band[1])).astype(float)
    if mask.sum() == 0:
        return np.zeros(shape)
    signal = np.fft.irfft(spec * mask, n=shape[-1], axis=-1)
    return signal / (signal.std() + 1e-9)


def _spatial_weights(n_channels, front_strong=True):
    """
    空间梯度权重：慢波前部强、alpha 后部强（近似 10-20 导联的前后分布）
    """
    grad = np.linspace(0.0, 1.0, n_channels)
    if front_strong:
        return 1.3 - 0.6 * grad      # 前 1.3 → 后 0.7
    return 0.7 + 0.6 * grad          # 前 0.7 → 后 1.3


def _add_spikes(rng, data, sfreq, n_spikes, amplitude):
    """在随机通道/时刻叠加棘波（高斯脉冲，宽度约 20ms）"""
    n_channels, n_times = data.shape
    width = max(1, int(0.02 * sfreq))         # 20ms
    t = np.arange(-3 * width, 3 * width + 1)
    kernel = np.exp(-0.5 * (t / width) ** 2)  # 高斯脉冲
    polarity = rng.choice([-1.0, 1.0])

    for _ in range(n_spikes):
        ch = rng.integers(0, n_channels)
        center = rng.integers(len(kernel), n_times - len(kernel))
        seg = data[ch, center - 3 * width: center + 3 * width + 1]
        if seg.shape[0] == kernel.shape[0]:
            data[ch, center - 3 * width: center + 3 * width + 1] += polarity * amplitude * kernel
    return data


def _add_artifact(rng, data, sfreq, kind, amplitude):
    """叠加伪迹：眼动（低频大漂移，前部通道）/ 肌电（高频爆发，少量通道）"""
    n_channels, n_times = data.shape
    if kind == "eog":
        drift = np.cumsum(rng.standard_normal(n_times)) / (sfreq ** 0.5)
        drift = drift / (np.std(drift) + 1e-9) * amplitude
        for ch in range(max(1, n_channels // 5)):     # 前部若干通道
            data[ch] += drift
    elif kind == "emg":
        burst = rng.standard_normal(n_times) * amplitude
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 50 / (0.5 * sfreq), btype="high")
        burst = filtfilt(b, a, burst)
        ch = rng.integers(0, n_channels)
        start = rng.integers(0, max(1, n_times - sfreq))
        data[ch, start:start + sfreq] += burst[:sfreq]
    return data


def simulate_epoch(rng, label, n_channels=19, n_times=500, sfreq=250,
                   snr_db=12.0, artifact_prob=0.2, scale=25.0):
    """
    合成单个 epoch

    参数:
        rng: np.random.Generator
        label: 0/1/2
        snr_db: 目标信噪比（dB）
        artifact_prob: 叠加伪迹的概率
        scale: 输出幅度缩放（µV 量级）

    返回:
        2D array, (n_channels, n_times)
    """
    profile = CLASS_PROFILES[int(label)]
    data = np.zeros((n_channels, n_times))

    # 1) 1/f 背景活动（低频更强）
    white = rng.standard_normal((n_channels, n_times))
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    pink = 1.0 / np.sqrt(np.maximum(freqs, 0.5))
    pink[freqs > 45] = 0.0
    background = np.fft.irfft(spec * pink, n=n_times, axis=-1)
    background /= (background.std() + 1e-9)
    data += 0.5 * background

    # 2) 各频带节律 × 空间权重
    for band_name, rng_band in BANDS.items():
        osc = _band_noise(rng, (n_channels, n_times), sfreq, rng_band)
        # 慢波前部强、快波/alpha 后部强
        weights = _spatial_weights(n_channels, front_strong=band_name in ("delta", "theta"))
        data += profile[band_name] * osc * weights[:, None]

    # 3) Abnormal：全导同步慢波（提高通道间同步性）
    if label == 2:
        common = _band_noise(rng, (1, n_times), sfreq, BANDS["delta"])
        data += 0.6 * common * _spatial_weights(n_channels, front_strong=True)[:, None]

    data = data / (data.std() + 1e-9)     # 标准化后再控制 SNR

    # 4) 白噪声（按目标 SNR 缩放）
    sigma = 1.0 / (10 ** (snr_db / 20.0))
    data = data + rng.standard_normal((n_channels, n_times)) * sigma

    # 5) 棘波瞬态
    if profile["spike"] > 0 and rng.random() < profile["spike"]:
        data = _add_spikes(rng, data, sfreq,
                           n_spikes=rng.integers(1, 4),
                           amplitude=2.5 * data.std())

    # 6) 随机伪迹
    if rng.random() < artifact_prob:
        kind = "eog" if rng.random() < 0.6 else "emg"
        data = _add_artifact(rng, data, sfreq, kind, amplitude=3.0 * data.std())

    return data * scale


def generate_dataset(n_subjects=60, epochs_per_subject=4, n_channels=19, n_times=500,
                     sfreq=250, snr_range=(6.0, 18.0), artifact_prob=0.2,
                     seed=42, balanced=True):
    """
    生成合成EEG数据集（同一被试的多个 epoch 共享标签与个体偏置）

    参数:
        n_subjects: 被试数
        epochs_per_subject: 每被试的 epoch 数
        n_channels / n_times / sfreq: 信号形状与采样率
        snr_range: 每个被试随机 SNR 范围（dB）
        artifact_prob: 伪迹概率
        seed: 随机种子
        balanced: 是否三类均衡

    返回:
        (X, y, subject_ids)
            X: (n_samples, n_channels, n_times)
            y: (n_samples,)
            subject_ids: (n_samples,) 被试ID（供 split_by_subject 使用）
    """
    rng = np.random.default_rng(seed)

    # 每个被试分配一个标签（均衡）
    if balanced:
        labels = np.tile(np.arange(N_CLASSES), int(np.ceil(n_subjects / N_CLASSES)))[:n_subjects]
        rng.shuffle(labels)
    else:
        labels = rng.integers(0, N_CLASSES, size=n_subjects)

    X_list, y_list, sid_list = [], [], []
    for sid in range(n_subjects):
        label = int(labels[sid])
        # 个体偏置：频带幅度整体缩放 + 该被试的 SNR
        subject_gain = rng.uniform(0.8, 1.25)
        snr_db = rng.uniform(*snr_range)

        for _ in range(epochs_per_subject):
            epoch = simulate_epoch(
                rng, label, n_channels=n_channels, n_times=n_times,
                sfreq=sfreq, snr_db=snr_db, artifact_prob=artifact_prob,
                scale=25.0 * subject_gain,
            )
            X_list.append(epoch.astype(np.float32))
            y_list.append(label)
            sid_list.append(sid)

    return np.stack(X_list), np.array(y_list), np.array(sid_list)
