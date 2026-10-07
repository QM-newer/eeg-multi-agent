# orchestrator/arbitration.py
"""
分歧仲裁模块 —— 对应文档 4.2(2)

仲裁流程：
1. 补充空间域特征（通道相干性 / 功能连接），扩充特征集
2. 诊断组基于扩充特征集重新推理
   - LightGBM（特征驱动）：直接消费扩充后的特征向量
   - 深度学习 Agent（信号驱动）：原始信号 + 时间翻转 TTA 集成推理
3. 计算各 Agent 的「特征可信度」
   - consensus：该 Agent 概率分布与共识分布的余弦相似度
   - conf_gain：二次诊断带来的置信度增益
   - quality_fit：与当前信号质量等级的适配度
4. 协调层基于特征可信度加权裁决（可信度 × 动态权重 → 归一化 → 裁剪）

返回 (new_results, info)，由 Orchestrator 用 info["weights"] 重新投票。
"""

import numpy as np

from config import ORCHESTRATOR_CONFIG, QUALITY_FIT_SCORES
from utils.spatial_features import extract_spatial_features, augment_features


def _tta_predict(agent, eeg_data, metadata=None, enable=True):
    """
    测试时增强推理：原始 + 时间翻转，两次概率取平均（降低方差）
    """
    probs = [np.asarray(agent.predict(eeg_data, metadata)["prob"], dtype=np.float64)]
    if enable:
        flipped = np.ascontiguousarray(np.asarray(eeg_data)[:, ::-1])
        probs.append(np.asarray(agent.predict(flipped, metadata)["prob"], dtype=np.float64))

    prob = np.mean(np.vstack(probs), axis=0)
    prob = np.clip(prob, 1e-8, None)
    return prob / prob.sum()


def _expected_n_features(agent):
    """获取特征驱动模型（LightGBM）训练时期望的输入维度，取不到返回 None"""
    model = getattr(agent, "model", None)
    return getattr(model, "n_features_in_", None)


def _rerun_agent(agent, eeg_data, augmented_features, base_features=None, enable_tta=True):
    """
    二次诊断时重新推理单个 Agent

    参数:
        agent: 诊断Agent实例
        eeg_data: 原始EEG (n_channels, n_times)
        augmented_features: 扩充（含空间域）后的特征向量
        base_features: 第一轮使用的（融合后）特征向量，维度不匹配时回退使用
        enable_tta: 深度模型是否使用时间翻转增强

    返回:
        (result dict, used_augmented bool)
    """
    used_augmented = False

    if agent.name == "lightgbm":
        # 特征驱动：优先用扩充特征集；若与训练维度不符则回退（避免维度错误）
        feat = np.asarray(augmented_features, dtype=np.float64).reshape(-1)
        n_expected = _expected_n_features(agent)
        if n_expected is not None and int(feat.size) != int(n_expected):
            if base_features is not None:
                feat = np.asarray(base_features, dtype=np.float64).reshape(-1)
            used_augmented = False
        else:
            used_augmented = True
        result = dict(agent.predict(eeg_data, {"all_features": feat}))
    else:
        # 信号驱动：原始信号 + TTA 集成
        prob = _tta_predict(agent, eeg_data, None, enable=enable_tta)
        result = dict(agent.predict(eeg_data, None))
        result["prob"] = prob.tolist()
        result["label"] = int(np.argmax(prob))
        result["confidence"] = float(np.max(prob))
        result["extra"] = dict(result.get("extra", {}))
        result["extra"]["tta"] = enable_tta

    result["extra"] = dict(result.get("extra", {}))
    result["extra"]["second_round"] = True
    if agent.name == "lightgbm":
        result["extra"]["used_augmented"] = used_augmented
    return result, used_augmented


def compute_feature_credibility(second_results, first_results, quality_level, config=None):
    """
    计算各 Agent 的「特征可信度」（0-1 越大越可信）

    三因子加权（config.ORCHESTRATOR_CONFIG["arbitration"]["credibility_weights"]）：
    - consensus：与共识概率分布的余弦相似度（越接近群体共识越可信）
    - conf_gain：二次诊断后置信度的提升（增益越大说明补充特征越有帮助）
    - quality_fit：与当前信号质量等级的适配度（低 SNR 时树模型更可信）

    返回:
        (credibility dict, detail dict)
    """
    cfg = (config or {}).get("arbitration", ORCHESTRATOR_CONFIG.get("arbitration", {}))
    w = cfg.get("credibility_weights", {"consensus": 0.40, "conf_gain": 0.35, "quality_fit": 0.25})
    gain_range = cfg.get("gain_range", 0.05)

    names = [r["agent_name"] for r in second_results]
    probs = np.array([np.asarray(r["prob"], dtype=np.float64) for r in second_results])
    conf2 = np.array([r["confidence"] for r in second_results], dtype=np.float64)
    conf1 = np.array([r["confidence"] for r in first_results], dtype=np.float64)

    # 1) 共识一致性：与各 Agent 平均概率分布的余弦相似度
    consensus = probs.mean(axis=0)
    consensus_norm = np.linalg.norm(consensus) + 1e-12
    cos_sim = np.array([
        float(np.dot(p, consensus) / (np.linalg.norm(p) * consensus_norm + 1e-12))
        for p in probs
    ])
    cos_sim = np.clip(cos_sim, 0.0, 1.0)

    # 2) 置信增益：[-gain_range, +gain_range] → [0, 1]
    gain = conf2 - conf1
    gain_norm = np.clip((gain + gain_range) / (2.0 * gain_range), 0.0, 1.0)

    # 3) 质量适配度
    fit_table = QUALITY_FIT_SCORES.get(quality_level, QUALITY_FIT_SCORES["medium"])
    quality_fit = np.array([float(fit_table.get(n, 0.9)) for n in names])

    score = (w.get("consensus", 0.40) * cos_sim
             + w.get("conf_gain", 0.35) * gain_norm
             + w.get("quality_fit", 0.25) * quality_fit)

    credibility = {n: float(s) for n, s in zip(names, score)}
    detail = {
        "consensus": {n: float(v) for n, v in zip(names, cos_sim)},
        "conf_gain": {n: float(v) for n, v in zip(names, gain)},
        "quality_fit": {n: float(v) for n, v in zip(names, quality_fit)},
    }
    return credibility, detail


def _arbitration_weights(base_weights, credibility, names):
    """可信度 × 基础权重 → 归一化 → 裁剪（min/max 来自 ORCHESTRATOR_CONFIG）"""
    dw = ORCHESTRATOR_CONFIG.get("dynamic_weight", {})
    min_w = dw.get("min_weight", 0.05)
    max_w = dw.get("max_weight", 0.60)

    raw = {}
    for n in names:
        raw[n] = float(base_weights.get(n, 1.0 / max(len(names), 1)) * credibility.get(n, 1.0))

    total = sum(raw.values())
    if total <= 0:
        raw = {n: 1.0 / len(names) for n in names}
        total = 1.0
    weights = {n: float(np.clip(v / total, min_w, max_w)) for n, v in raw.items()}

    s = sum(weights.values())
    return {n: v / s for n, v in weights.items()}


def arbitrate(eeg_data, agent_results, all_features, diagnosis_agents=None,
              weights=None, sfreq=250, snr_db=None, quality_level="medium", config=None):
    """
    分歧仲裁主流程：补特征 → 重推理 → 特征可信度裁决

    参数:
        eeg_data: 原始EEG (n_channels, n_times)
        agent_results: 第一轮各Agent结果 list[dict]
        all_features: 第一轮多域特征向量
        diagnosis_agents: 诊断Agent列表（None 时不做重推理，仅返回原结果）
        weights: 当前动态权重 dict
        sfreq: 采样率
        snr_db / quality_level: 信号质量信息（写入 info）
        config: ORCHESTRATOR_CONFIG 覆盖项

    返回:
        (new_results, info)
            new_results: list[dict] 二次诊断结果
            info: {spatial_features, spatial_dim, credibility, credibility_detail,
                   weights, quality_level, snr_db, augmented_dim}
    """
    cfg = config or ORCHESTRATOR_CONFIG
    arb_cfg = cfg.get("arbitration", {})
    bands = arb_cfg.get("spatial_bands", None)

    # 1) 补充空间域特征
    spatial_features, spatial_info = extract_spatial_features(eeg_data, sfreq=sfreq, bands=bands)
    augmented_features = augment_features(all_features, spatial_features)

    # 2) 基于扩充特征集重新推理
    augmented_used = False
    if diagnosis_agents:
        new_results = []
        for agent in diagnosis_agents:
            res, used = _rerun_agent(
                agent, eeg_data, augmented_features,
                base_features=all_features, enable_tta=arb_cfg.get("tta", True),
            )
            augmented_used = augmented_used or used
            new_results.append(res)
    else:
        new_results = list(agent_results)

    # 3) 特征可信度 + 裁决权重
    names = [r["agent_name"] for r in new_results]
    credibility, detail = compute_feature_credibility(
        new_results, agent_results, quality_level, cfg
    )
    base = weights or {n: 1.0 / max(len(names), 1) for n in names}
    final_weights = _arbitration_weights(base, credibility, names)

    info = {
        "spatial_features": spatial_features,
        "spatial_dim": int(spatial_features.size),
        "spatial_info": spatial_info,
        "augmented_features": augmented_features,
        "augmented_dim": int(augmented_features.size),
        "credibility": credibility,
        "credibility_detail": detail,
        "weights": final_weights,
        "quality_level": quality_level,
        "snr_db": snr_db,
        "augmented_used": augmented_used,
        "triggered": bool(diagnosis_agents),
    }
    return new_results, info
