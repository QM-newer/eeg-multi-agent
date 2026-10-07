# orchestrator/dynamic_weight.py
"""
信号质量（SNR）感知的动态权重调整 —— 对应文档 4.2(3)

三档规则：
- 高信噪比（SNR > 10 dB）  ：Transformer / EEGNet 权重 +10%（深度学习在干净数据上更可靠）
- 中等信噪比（5 - 10 dB）  ：维持默认权重
- 低信噪比 （SNR < 5 dB）  ：LightGBM 权重 +15%（树模型对噪声更鲁棒，手工特征抗干扰）
- 附加：伪迹比例超过阈值时，无论 SNR 一律按低质量处理
"""

import numpy as np

from config import ORCHESTRATOR_CONFIG, QUALITY_CONFIG, QUALITY_FIT_SCORES


def _normalize(weights):
    """归一化到和为 1"""
    total = sum(weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _clamp(weights, dw_config=None):
    """按 min_weight / max_weight 裁剪后重新归一化"""
    dw = dw_config or ORCHESTRATOR_CONFIG.get("dynamic_weight", {})
    min_w = dw.get("min_weight", 0.05)
    max_w = dw.get("max_weight", 0.60)

    weights = {k: float(np.clip(v, min_w, max_w)) for k, v in weights.items()}
    return _normalize(weights)


def classify_quality(snr_db, artifact_ratio=None, config=None):
    """
    根据 SNR 与伪迹比例判定信号质量等级

    返回:
        str: "high" / "medium" / "low"
    """
    cfg = config or QUALITY_CONFIG
    artifact_threshold = cfg.get("artifact_threshold", 0.30)

    if artifact_ratio is not None and artifact_ratio > artifact_threshold:
        return "low"
    if snr_db is None:
        return "medium"
    if snr_db > cfg.get("snr_high", 10.0):
        return "high"
    if snr_db < cfg.get("snr_low", 5.0):
        return "low"
    return "medium"


def adjust_weights_by_snr(base_weights, snr_db, artifact_ratio=None, config=None):
    """
    按文档三档规则调整各诊断Agent权重

    参数:
        base_weights: dict，基础权重
        snr_db: float，信噪比（dB）
        artifact_ratio: float/None，伪迹比例（0-1）
        config: dict/None，覆盖 QUALITY_CONFIG

    返回:
        dict: 调整后的权重（归一化，和为1）
    """
    cfg = config or QUALITY_CONFIG
    weights = dict(base_weights)
    level = classify_quality(snr_db, artifact_ratio, cfg)

    if level == "high":
        boost = 1.0 + cfg.get("high_boost", 0.10)
        for key in cfg.get("high_boost_agents", ["eegnet", "transformer"]):
            if key in weights:
                weights[key] *= boost
    elif level == "low":
        boost = 1.0 + cfg.get("low_boost", 0.15)
        for key in cfg.get("low_boost_agents", ["lightgbm"]):
            if key in weights:
                weights[key] *= boost
    # medium：维持默认权重（仅做归一化与裁剪）

    return _clamp(weights)


def quality_fit_scores(quality_level, agent_names):
    """
    质量等级 → 各 Agent 的适配度分数（供仲裁阶段计算特征可信度）

    参数:
        quality_level: "high" / "medium" / "low"
        agent_names: list[str]

    返回:
        list[float]: 与 agent_names 一一对应的适配度（约 0.7-1.0）
    """
    table = QUALITY_FIT_SCORES.get(quality_level, QUALITY_FIT_SCORES["medium"])
    return [float(table.get(name, 0.9)) for name in agent_names]


def adjust_weights_by_quality(base_weights, quality_score):
    """
    兼容旧接口：按 0-1 质量分调整权重（连续版）

    保留原因：早期脚本/测试按 0-1 质量分调用；
    正式流程请使用 adjust_weights_by_snr（严格对齐文档三档规则）。

    策略：
    - quality >= 0.8：偏向深度学习模型
    - 0.5 < quality < 0.8：维持默认
    - quality <= 0.5：偏向 LightGBM
    """
    weights = dict(base_weights)
    shift = (quality_score - 0.5) * 0.4

    if quality_score >= 0.8:
        for key in ["eegnet", "tcn", "transformer"]:
            if key in weights:
                weights[key] *= (1 + shift * 0.5)
        if "lightgbm" in weights:
            weights["lightgbm"] *= (1 - shift * 0.3)
    elif quality_score <= 0.5:
        if "lightgbm" in weights:
            weights["lightgbm"] *= (1 - shift * 0.5)
        if "transformer" in weights:
            weights["transformer"] *= (1 + shift * 0.3)

    return _clamp(weights)


def adjust_weights_by_disagreement(base_weights, agent_results):
    """
    根据历史表现动态调整权重（占位）
    后续可扩展为：按验证集准确率 / 滑动窗口准确率自动更新 AGENT_WEIGHTS
    """
    return dict(base_weights)
