# config.py
# ============================================================
# 项目全局配置（路径、超参、Agent权重等）
# 说明：config.py 应保持零第三方依赖，仅使用标准库。
# ============================================================

import os
from pathlib import Path

# ============================================================
# 项目根目录（基于 config.py 自动定位）
# 无论从哪里启动脚本，都能正确找到 checkpoints / data / results
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path):
    """将相对路径解析为项目根目录下的绝对路径（Windows / 类Unix通用）。

    若传入的本身就是绝对路径则原样返回。
    """
    p = Path(path)
    return str(p if p.is_absolute() else BASE_DIR / p)


# ============================================================
# 随机性 & 并行
# ============================================================
RANDOM_SEED = 42          # 全局随机种子（模型、数据划分、特征提取共用）
NUM_WORKERS = 0           # DataLoader 并行进程数（Windows 下建议 0，可自行调大）
VERBOSE = True            # 是否输出详细信息

# ============================================================
# 数据配置
# ============================================================
DATA_CONFIG = {
    "data_root": "E:/脑电数据按姓名整理",      # 原始EEG文件根目录（按受试者分文件夹）
    # 病例标签来源：项目上级目录的本地原始备份（优先于 E 盘 organize 阶段导出的索引标签）
    "label_excel": str(BASE_DIR.parent / "脑电+病例_原始备份.xlsx"),
    "subject_master": resolve_path("data/subject_master.csv"),   # 清洗后的样本-标签主表
    # organize 阶段产出的「记录 → 文件」索引（含 folder/record_id/eeg_file/时长/采样率）
    "raw_index_csv": "E:/eeg_organize_tmp/subject_master.csv",
    # 记录级主表：病例标签 + 文件路径 + 划分，训练与缓存构建的唯一入口
    "record_index": resolve_path("data/record_index.csv"),
    # 预处理后的 epoch 缓存目录。
    # 注意：1598 条记录 × 100 epoch × 19 通道 × 500 点 × float32 ≈ 6GB，
    # C 盘空间不足，故放在数据盘 E 盘（1.7TB 可用）
    "epoch_cache_dir": "E:/eeg_asd_cache/epochs",
    "max_epochs_per_record": 100,  # 每条记录最多保留的 epoch 数（均匀抽样）
    # 单条记录读取窗口（原始记录中位数约 4 小时，全量解码不现实，按窗口取样）
    "read_window": {
        "skip_sec": 60,        # 跳过开头（阻抗测试/准备阶段伪迹多）
        "duration_sec": 600,   # 每条记录读取时长（秒）
    },
    "sfreq": 500,            # 原始采样率
    "target_sfreq": 250,     # 目标采样率（降采样后）
    "n_channels": 35,        # 总通道数
    "n_eeg_channels": 19,    # EEG通道数（通道鉴别后确认）
    "eeg_channels": None,    # 可选：EEG通道名列表，如 ["Fp1","Fp2",...]，None 表示取前 n_eeg_channels 个
    # 实际使用的物理通道索引（Ch1~Ch19，即 idx 0~18）。
    # 依据：实测 Ch1~Ch23 幅值在 10µV 量级且均值≈0（EEG 特征），
    # Ch24~Ch27 峰值恰为满 16bit 量程（疑非 EEG 通道），Ch33~Ch35 为恒定 DC 通道。
    # 解剖命名在文件中不存在，待蒙太奇确认后可改此列表。
    "eeg_channel_indices": list(range(19)),
    "epoch_length": 2,       # epoch长度（秒）
    "epoch_overlap": 0.5,    # 重叠比例（0 表示无重叠）
    "train_ratio": 0.7,      # 训练/验证/测试 划分比例
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "normalize": "per_channel",  # 归一化方式: "per_channel" / "global" / None
}

# ============================================================
# 模型权重文件路径（基于项目根目录）
# ============================================================
MODEL_PATHS = {
    "lightgbm": resolve_path("checkpoints/lightgbm/model.pkl"),
    "eegnet": resolve_path("checkpoints/eegnet/best.pth"),
    "tcn": resolve_path("checkpoints/tcn/best.pth"),
    "transformer": resolve_path("checkpoints/transformer/best.pth"),
}

# 特征融合器（mRMR）与训练产物路径
FUSION_PATH = resolve_path("checkpoints/fusion/fusion.joblib")
AGENT_WEIGHTS_PATH = resolve_path("checkpoints/agent_weights.json")
VAL_ACCURACIES_PATH = resolve_path("checkpoints/val_accuracies.json")

# ============================================================
# 特征工程配置（供 3 个特征Agent：时域 / 频域 / 时频）
# ============================================================
FEATURE_CONFIG = {
    # 时域特征
    "time_domain": {
        "enable": True,
        "features": [           # 每通道统计量
            "mean", "std", "var", "rms", "peak_to_peak",
            "skewness", "kurtosis", "zc_rate", "hjd", "band_power_ratio",
        ],
    },
    # 频域特征（FFT 分段）
    "freq_domain": {
        "enable": True,
        "n_fft": 512,                 # FFT 点数（建议按 target_sfreq 调整）
        "window": "hann",             # 窗函数: "hann" / "hamming" / None
        "bands": {                    # 经典 EEG 频带（Hz）
            "delta":    (0.5, 4),
            "theta":    (4, 8),
            "alpha":    (8, 13),
            "beta":     (13, 30),
            "gamma":    (30, 45),
        },
        "features": ["psd", "band_power", "peak_freq", "spectral_edge"],
    },
    # 时频特征（CWT / STFT）
    "timefreq": {
        "enable": True,
        "method": "stft",             # "stft" / "cwt"
        "n_freq_bins": 32,
        "freq_min": 0.5,
        "freq_max": 45,
        "features": ["power_map", "mean_power", "entropy"],
    },
}

# ============================================================
# Agent 初始权重（加权投票的基础）
# ============================================================
AGENT_WEIGHTS = {
    "lightgbm": 0.20,
    "eegnet": 0.30,
    "tcn": 0.25,
    "transformer": 0.25,
}

# 特征/验证类 Agent（不直接投票，仅产出特征或解释，供诊断 Agent 与仲裁使用）
FEATURE_AGENTS = {
    "time_domain": True,
    "freq_domain": True,
    "timefreq": True,
}

# ============================================================
# 协调层配置（Orchestrator / 加权投票 / 动态权重）
# ============================================================
ORCHESTRATOR_CONFIG = {
    "vote_method": "weighted_majority",  # "weighted_majority" / "majority" / "soft_voting"
    "dynamic_weight": {
        "enable": True,                  # 是否启用动态权重
        "update_every": 50,              # 每 N 个样本更新一次
        "min_weight": 0.05,              # 权重下限（防某模型权重收敛到 0）
        "max_weight": 0.60,              # 权重上限
        "decay": 0.95,                   # 历史准确率的衰减系数
    },
    "disagreement_config": {             # 见下方 DISAGREEMENT_CONFIG，此处可覆盖
    },
    "arbitration": {                     # 分歧仲裁（补充空间域特征 → 重推理 → 可信度裁决）
        "enable": True,
        "tta": True,                     # 深度模型二次诊断时用时间翻转做测试时增强
        "gain_range": 0.05,              # 置信增益归一化范围（±0.05 映射到 0-1）
        "credibility_weights": {         # 特征可信度三因子权重（和为1）
            "consensus": 0.40,           # 与共识分布的余弦相似度
            "conf_gain": 0.35,           # 二次诊断后的置信度增益
            "quality_fit": 0.25,         # 与当前信号质量的适配度
        },
        "spatial_bands": {               # 空间域特征使用的频带（与 FEATURE_CONFIG 对齐）
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        },
    },
}

# ============================================================
# 信号质量（SNR）感知配置 —— 对应文档 4.2(3)
# ============================================================
QUALITY_CONFIG = {
    "snr_high": 10.0,                 # dB，SNR > 10 判为高质量
    "snr_low": 5.0,                   # dB，SNR < 5 判为低质量
    "high_boost": 0.10,               # 高质量：深度学习 Agent 权重 +10%
    "low_boost": 0.15,                # 低质量：LightGBM 权重 +15%
    "high_boost_agents": ["eegnet", "transformer"],
    "low_boost_agents": ["lightgbm"],
    "artifact_threshold": 0.30,       # 伪迹比例超过该值强制判为低质量
    "amp_threshold": 100.0,           # 幅度伪迹阈值（µV）
    "flat_std_threshold": 0.5,        # 平坦（掉线）通道标准差阈值
}

# 质量等级 → 各 Agent 适配度（供仲裁阶段计算特征可信度）
QUALITY_FIT_SCORES = {
    "high":   {"lightgbm": 0.80, "eegnet": 1.00, "tcn": 0.90, "transformer": 1.00},
    "medium": {"lightgbm": 1.00, "eegnet": 1.00, "tcn": 1.00, "transformer": 1.00},
    "low":    {"lightgbm": 1.00, "eegnet": 0.80, "tcn": 0.80, "transformer": 0.70},
}

# ============================================================
# 分歧检测配置
# ============================================================
DISAGREEMENT_CONFIG = {
    # 校准依据: scripts/calibrate_disagreement.py（val@20, 4800 epochs, 2026-10-07）
    # 旧配置(0.15/0.6/0.8)触发率 100%——四Agent平均概率接近均匀(中位熵1.0878≈ln3)，
    # 熵规则失效；概率差中位数0.178>0.15 也常年触发。校准后触发率约 40%，
    # 被试精度持平最优(val 0.5833)，epoch 精度最高(0.5417)，推理成本约减半。
    "prob_diff_threshold": 0.30,      # 最大概率差阈值（高于则视为分歧）
    "label_majority_threshold": 0.6,  # 标签一致比例阈值（2-2/2-1-1 分裂时触发，核心规则）
    "entropy_threshold": 1.099,       # 熵值阈值（>ln3≈1.0986 不可能，仅在极端均匀时触发）
    "arbitration": "weighted_vote",   # 仲裁方式: "weighted_vote" / "expert_fallback" / "manual_review"
}

# ============================================================
# 模型结构超参（供 models/ 与 training/ 引用）
# 注：n_channels 等请与 DATA_CONFIG 保持一致
# ============================================================
# ============================================================
# 分类配置（需早于 MODEL_CONFIG 定义）
# ============================================================
CLASS_NAMES = ["Normal", "Borderline", "Abnormal"]
N_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# 二分类配置（Normal vs Abnormal，Borderline 合并到 Abnormal）
BINARY_CLASS_NAMES = ["Normal", "Abnormal"]
N_BINARY_CLASSES = 2
BINARY_LABEL_MAP = {0: 0, 1: 1, 2: 1}   # 三分类 → 二分类映射
BINARY_MODEL_PATHS = {
    "lightgbm": resolve_path("checkpoints/lightgbm_binary/model.pkl"),
    "eegnet": resolve_path("checkpoints/eegnet_binary/best.pth"),
    "tcn": resolve_path("checkpoints/tcn_binary/best.pth"),
    "transformer": resolve_path("checkpoints/transformer_binary/best.pth"),
}
BINARY_FUSION_PATH = resolve_path("checkpoints/fusion_binary/fusion.joblib")

# 事件级特征（+EventAgent，82 维）二分类 LightGBM 变体（A/B 对照用，不覆盖基线）
BINARY_EVENT_MODEL_PATH = resolve_path("checkpoints/lightgbm_binary_event/model.pkl")
BINARY_EVENT_FUSION_PATH = resolve_path("checkpoints/fusion_binary_event/fusion.joblib")

# 置信度 Borderline 判定阈值
BORDERLINE_CONFIDENCE_RANGE = (0.35, 0.65)

# 输入长度：epoch_length(秒) × 目标采样率（如 2s × 250Hz = 500 点）
N_TIMES = int(DATA_CONFIG["epoch_length"] * DATA_CONFIG["target_sfreq"])

MODEL_CONFIG = {
    "eegnet": {
        "n_channels": DATA_CONFIG["n_eeg_channels"],   # 输入通道数
        "n_times": N_TIMES,                            # 输入时间点数
        "sfreq": DATA_CONFIG["target_sfreq"],          # 采样率
        "filters": [8, 16, 16],                        # 三层Conv2D滤波器数（与 models/eegnet.py 对齐）
        "kernel_time": 64,                             # 时间卷积核大小（样本数）
        "pool": 4,                                     # 第一次池化因子
        "pool2": 8,                                    # 第二次池化因子（总降采样 32）
        "dropout": 0.25,
        "activation": "elu",
        "n_classes": N_CLASSES,
    },
    "tcn": {
        "n_channels": DATA_CONFIG["n_eeg_channels"],   # 输入通道数（特征维度）
        "n_layers": 6,
        "hidden_channels": [64, 64, 64, 64, 64, 64],   # 每层通道数（长度须 >= n_layers）
        "kernel_size": 5,
        "dropout": 0.2,
        "dilation_base": 2,                            # 膨胀基数
        "n_classes": N_CLASSES,
    },
    "transformer": {
        "n_channels": DATA_CONFIG["n_eeg_channels"],
        "n_times": N_TIMES,
        "patch_size": 50,                              # 每个patch的时间点数
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "max_len": 512,                                # 序列最大长度
        "n_classes": N_CLASSES,
    },
}

# ============================================================
# 训练配置
# ============================================================
TRAIN_CONFIG = {
    "batch_size": 64,
    "lr": 1e-3,
    "lr_scheduler": "reduce_on_plateau",   # None / "reduce_on_plateau" / "cosine"
    "weight_decay": 1e-4,
    "epochs": 100,
    "early_stop_patience": 15,
    "device": "cpu",            # 有 GPU 改成 "cuda"（或调用 ensure_device()）
    "save_best": True,          # 是否只保留最优权重
    "shuffle": True,
}

def ensure_device(prefer_cuda=True):
    """自动返回可用设备。config.py 不依赖 torch，因此在需要时动态探测。

    prefer_cuda=True 且环境可用时返回 "cuda"，否则返回 "cpu"。
    用法: device = TRAIN_CONFIG["device"] if not prefer_cuda else ensure_device()
    """
    if not prefer_cuda:
        return TRAIN_CONFIG["device"]
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# ============================================================
# 结果 / 日志输出目录（自动创建）
# ============================================================
OUTPUT_DIRS = {
    "results":      resolve_path("results"),
    "baseline":     resolve_path("results/baseline"),
    "multi_agent":  resolve_path("results/multi_agent"),
    "ablation":     resolve_path("results/ablation"),
    "figures":      resolve_path("results/figures"),
    "checkpoints":  resolve_path("checkpoints"),
    "logs":         resolve_path("results/logs"),
}
for _d in OUTPUT_DIRS.values():
    os.makedirs(_d, exist_ok=True)

# ============================================================
# 日志配置
# ============================================================
LOG_CONFIG = {
    "level": "INFO",               # DEBUG / INFO / WARNING / ERROR
    "log_file": resolve_path("results/logs/run.log"),
    "console": True,               # 是否同时输出到控制台
    "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
}
