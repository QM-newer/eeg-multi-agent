# orchestrator/voting.py

import numpy as np


def weighted_vote(agent_results, weights):
    """
    加权投票：按各Agent权重融合概率（soft voting）

    参数:
        agent_results: list[dict]，每个Agent的预测结果
        weights: dict，{agent_name: weight}，权重和可以不为1，会自动归一化

    返回:
        dict: {label, prob, confidence}
    """
    if not agent_results:
        raise ValueError("agent_results 为空，无法投票")

    n_classes = len(agent_results[0]["prob"])
    final_prob = np.zeros(n_classes)
    total_weight = 0.0

    for r in agent_results:
        w = weights.get(r["agent_name"], 1.0 / len(agent_results))
        final_prob += w * np.asarray(r["prob"], dtype=np.float64)
        total_weight += w

    # 归一化
    if total_weight > 0:
        final_prob = final_prob / total_weight

    return {
        "label": int(np.argmax(final_prob)),
        "prob": final_prob.tolist(),
        "confidence": float(np.max(final_prob)),
    }
