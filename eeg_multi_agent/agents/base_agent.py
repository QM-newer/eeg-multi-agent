# agents/base_agent.py
# ============================================================
# 所有 Agent 的抽象基类。统一接口：predict(eeg_data, metadata) -> dict
# ============================================================

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseAgent(ABC):
    """
    所有Agent的抽象基类

    统一输出格式：
    {
        "agent_name": str,      # Agent名称
        "label": int,           # 预测类别 0/1/2（诊断Agent用，特征Agent为None）
        "prob": list,           # 三类概率 [p0, p1, p2]（诊断Agent用）
        "confidence": float,    # 置信度（最大概率）
        "extra": dict           # 额外输出（特征向量、解释信息等）
    }
    """

    # 子类后处理返回字典必须包含的字段（防御性校验用）
    REQUIRED_OUTPUT_KEYS = {"agent_name", "label", "prob", "confidence", "extra"}

    def __init__(self, name: str, model_path: Optional[str] = None, device: str = "cpu"):
        self.name = name
        self.model_path = model_path
        self.device = device
        self.model = None
        if model_path:
            self._load_model()

    def _load_model(self) -> None:
        """加载模型，默认空实现，子类按需重写"""
        pass

    # ============================================================
    # 子类必须实现的三个步骤
    # ============================================================

    @abstractmethod
    def _preprocess(self, eeg_data: np.ndarray, metadata: Optional[dict] = None) -> Any:
        """
        数据预处理
        输入：原始EEG数据 shape=(n_channels, n_times)
        输出：处理后的模型输入
        """
        raise NotImplementedError

    @abstractmethod
    def _inference(self, processed_data: Any) -> np.ndarray:
        """
        模型推理
        输入：预处理后的数据
        输出：概率数组 shape=(n_classes,) 或 特征向量 shape=(n_features,)
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, output: np.ndarray) -> dict:
        """
        结果后处理
        输入：推理输出
        输出：统一格式的结果字典（label/prob/confidence/extra 四个键）
        """
        raise NotImplementedError

    # ============================================================
    # 对外统一入口
    # ============================================================

    def predict(self, eeg_data: np.ndarray, metadata: Optional[dict] = None) -> dict:
        """
        对外统一接口：预处理 → 推理 → 后处理
        所有Agent调用方式完全一样
        """
        try:
            processed = self._preprocess(eeg_data, metadata)
            output = self._inference(processed)
            result = self._postprocess(output)
        except NotImplementedError:
            raise
        except Exception as exc:
            raise RuntimeError(f"[{self.name}] predict 执行失败: {exc}") from exc

        result["agent_name"] = self.name
        self._validate_output(result)
        return result

    def _validate_output(self, result: dict) -> None:
        """校验后处理返回的字典是否满足统一格式，尽早暴露子类实现遗漏。"""
        missing = self.REQUIRED_OUTPUT_KEYS - set(result.keys())
        if missing:
            raise ValueError(f"[{self.name}] 输出字典缺少字段: {sorted(missing)}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r}, device={self.device!r})>"
