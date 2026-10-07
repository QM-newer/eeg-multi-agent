# agents/validation_agents/consistency_agent.py

from collections import Counter

import numpy as np
from agents.base_agent import BaseAgent


class ConsistencyAgent(BaseAgent):
    """
    一致性校验Agent：评估多Agent诊断结果的一致性

    计算指标：
    - 标签一致率（多数派比例）
    - 概率方差（各Agent预测概率的离散程度）
    - 熵值（平均概率的不确定性）
    - 综合一致性评分（0-1分，越高越一致）
    """

    def _load_model(self):
        pass

    def _preprocess(self, eeg_data, metadata=None):
        """从metadata获取各Agent诊断结果"""
        agent_results = metadata.get("agent_results", []) if metadata else []
        return agent_results

    def _inference(self, processed_data):
        """计算一致性指标"""
        results = processed_data
        if not results:
            return np.array([0.0, 0.0, 0.0, 0.0])

        probs = np.array([r["prob"] for r in results])
        labels = [r["label"] for r in results]

        # 指标1：标签一致率
        majority_count = Counter(labels).most_common(1)[0][1]
        label_agreement = majority_count / len(labels)

        # 指标2：概率标准差（平均每个类别的概率离散度）
        prob_std = np.mean(np.std(probs, axis=0))

        # 指标3：平均概率的熵值
        avg_prob = np.mean(probs, axis=0)
        entropy = -np.sum(avg_prob * np.log(avg_prob + 1e-10))
        max_entropy = np.log(len(avg_prob))  # 均匀分布时的最大熵
        normalized_entropy = entropy / max_entropy  # 归一化到0-1

        # 指标4：综合一致性评分（越高越一致）
        # = 标签一致率 * 0.5 + (1 - 归一化熵) * 0.3 + (1 - 概率标准差) * 0.2
        consistency_score = (
            label_agreement * 0.5
            + (1 - normalized_entropy) * 0.3
            + max(0, 1 - prob_std * 5) * 0.2
        )
        consistency_score = max(0.0, min(1.0, consistency_score))

        return np.array([
            label_agreement,      # 标签一致率
            prob_std,             # 概率标准差
            normalized_entropy,   # 归一化熵
            consistency_score,    # 综合一致性评分
        ])

    def _postprocess(self, output):
        label_agreement, prob_std, norm_entropy, consistency_score = output

        # 判断是否需要人工复核
        need_review = consistency_score < 0.6  # 一致性低于0.6建议人工复核

        # 一致性等级
        if consistency_score >= 0.8:
            level = "高"
        elif consistency_score >= 0.6:
            level = "中"
        else:
            level = "低"

        return {
            "label": None,
            "prob": None,
            "confidence": None,
            "extra": {
                "label_agreement": float(label_agreement),
                "prob_std": float(prob_std),
                "normalized_entropy": float(norm_entropy),
                "consistency_score": float(consistency_score),
                "consistency_level": level,
                "need_human_review": need_review,
            }
        }
