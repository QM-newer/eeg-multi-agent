# agents/validation_agents/explainability_agent.py

import numpy as np
from agents.base_agent import BaseAgent


class ExplainabilityAgent(BaseAgent):
    """
    可解释性Agent：分析诊断结果的特征贡献度
    当前为骨架实现，后续可接入SHAP/LIME

    功能：
    - 计算各特征对最终诊断的贡献度
    - 生成特征重要性排序
    - 输出自然语言解释文本
    """

    def _load_model(self):
        pass  # 可解释性Agent不需要预训练模型

    def _preprocess(self, eeg_data, metadata=None):
        """从metadata中获取诊断结果和特征"""
        return {
            "eeg_data": eeg_data,
            "agent_results": metadata.get("agent_results", []) if metadata else [],
            "all_features": metadata.get("all_features", None) if metadata else None,
            "final_label": metadata.get("final_label", None) if metadata else None,
        }

    def _inference(self, processed_data):
        """
        特征重要性分析（简化版）
        后续替换为SHAP/LIME等正式方法
        """
        all_features = processed_data["all_features"]

        if all_features is None:
            return np.array([0.0])

        # 简化：用特征绝对值大小近似重要性（仅用于占位）
        feature_importance = np.abs(np.asarray(all_features, dtype=np.float64))
        feature_importance = feature_importance / (feature_importance.sum() + 1e-10)

        return feature_importance

    def _postprocess(self, output):
        output = np.asarray(output, dtype=np.float64).reshape(-1)

        # 取Top-10重要特征
        top_indices = np.argsort(output)[::-1][:10]
        top_importance = output[top_indices]

        # 生成解释文本（模板）：逐个列举Top-3特征
        parts = [
            f"特征{int(top_indices[i])}（贡献度{top_importance[i]:.3f}）"
            for i in range(min(3, len(top_indices)))
        ]
        explanation = "诊断主要依据以下特征：" + "、".join(parts) + "等。"

        return {
            "label": None,
            "prob": None,
            "confidence": None,
            "extra": {
                "feature_importance": output.tolist(),
                "top_features": top_indices.tolist(),
                "top_importance": top_importance.tolist(),
                "explanation": explanation,
            }
        }
