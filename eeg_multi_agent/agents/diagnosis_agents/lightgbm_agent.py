# agents/diagnosis_agents/lightgbm_agent.py

import os

import numpy as np
from agents.base_agent import BaseAgent


class LightGbmAgent(BaseAgent):
    """
    LightGBM诊断Agent
    输入：EEG epoch (n_channels, n_times) 或已融合的特征向量
    输出：分类概率

    推理链路必须与训练一致：
        原始epoch → 逐通道归一化(训练集统计量) → 多域特征
        → 融合器标准化+mRMR选择 → LightGBM
    若上游（orchestrator）已给出"选择后"维数的特征，则直接采用，避免重复计算。

    支持三分类和二分类模式：自动从模型和融合器路径推断。
    """

    def __init__(self, name, model_path=None, device="cpu"):
        # 必须在 super().__init__() 之前初始化：基类构造里就会调用 _load_model()
        self.fusion = None
        self.norm_mean = self.norm_std = None
        super().__init__(name, model_path, device)
        # 基类只在传入 model_path 时加载模型，这里保证 _load_model 至少执行一次
        if self.model is None:
            self._load_model()

    def _load_model(self):
        """加载模型，没有模型文件时返回None（用随机概率占位）"""
        if self.model_path:
            try:
                import joblib
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None
        else:
            self.model = None

        # 特征融合器：优先加载与模型匹配的版本
        # 二分类模型路径含 "_binary"，对应 fusion_binary；
        # lightgbm_binary_event 变体对应 fusion_binary_event（含事件特征）
        try:
            from config import resolve_path, BINARY_FUSION_PATH, BINARY_EVENT_FUSION_PATH
            from features.fusion import FeatureFusion
            mp = str(self.model_path) if self.model_path else ""
            if "binary_event" in mp:
                p = BINARY_EVENT_FUSION_PATH
            elif "_binary" in mp:
                p = BINARY_FUSION_PATH
            else:
                p = resolve_path("checkpoints/fusion/fusion.joblib")
            if os.path.exists(p):
                self.fusion = FeatureFusion.load(p)
        except Exception:
            self.fusion = None

        # 逐通道归一化统计量（与训练时同一变换）
        try:
            from data.epoch_cache import load_norm_stats
            self.norm_mean, self.norm_std = load_norm_stats()
        except Exception:
            self.norm_mean = self.norm_std = None

    def _preprocess(self, eeg_data, metadata=None):
        """得到 (1, n_selected) 的输入特征"""
        # 1) 上游已融合过的特征：维数等于"选择后"维数时直接采用
        if metadata and "all_features" in metadata:
            f = np.asarray(metadata["all_features"], dtype=np.float64)
            if f.ndim == 1:
                f = f.reshape(1, -1)
            if self.fusion is None or f.shape[1] == self.fusion.n_output:
                return f

        # 2) 从原始 epoch 现场提取（保证与训练同尺度）
        from training.common import extract_multi_domain_features

        x = np.asarray(eeg_data, dtype=np.float64)
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        if self.norm_mean is not None and self.norm_mean.shape[1] == x.shape[1]:
            x = (x - self.norm_mean) / self.norm_std
        # 融合器训练时含事件特征 → 推理同样提取（旧融合器无该属性，默认关闭）
        include_event = bool(getattr(self.fusion, "include_event", False)) if self.fusion else False
        feats, _ = extract_multi_domain_features(x.astype(np.float32), n_jobs=1,
                                                 include_event=include_event)
        if self.fusion is not None:
            feats = self.fusion.transform(feats)
        return feats

    def _inference(self, processed_data):
        """模型推理"""
        if self.model is not None:
            prob = self.model.predict_proba(processed_data)
            return np.asarray(prob, dtype=np.float64).reshape(-1)
        # 无模型时返回随机概率（占位）
        prob = np.random.dirichlet(np.ones(3))  # 随机生成和为1的概率
        return prob

    def _postprocess(self, output):
        output = np.asarray(output, dtype=np.float64).reshape(-1)
        return {
            "label": int(np.argmax(output)),
            "prob": output.tolist(),
            "confidence": float(np.max(output)),
            "extra": {"model_type": "lightgbm"}
        }
