# agents/diagnosis_agents/eegnet_agent.py

import numpy as np
import torch
from agents.base_agent import BaseAgent
from config import DATA_CONFIG, MODEL_CONFIG, N_TIMES
from models.eegnet import EEGNet


class EegNetAgent(BaseAgent):
    """
    EEGNet诊断Agent
    输入：原始EEG信号 (n_channels, n_times)
    输出：三分类概率

    模型结构超参统一来自 config.MODEL_CONFIG["eegnet"]，确保训练与推理一致
    """

    N_CHANNELS = DATA_CONFIG["n_eeg_channels"]   # 19
    N_TIMES = N_TIMES                            # 500

    def __init__(self, name, model_path=None, device="cpu", n_classes=None):
        self._n_classes_override = n_classes   # None=自动从权重推断
        super().__init__(name, model_path, device)
        from data.epoch_cache import load_norm_stats
        self.norm_mean, self.norm_std = load_norm_stats()
        if self.model is None:
            self._load_model()

    def _load_model(self):
        """加载模型权重，没有则用随机初始化。自动从权重文件推断 n_classes。"""
        cfg = MODEL_CONFIG["eegnet"]
        # 先尝试从权重文件推断类别数
        n_classes = self._n_classes_override or cfg["n_classes"]
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location="cpu")
                for key in ["classifier.weight", "fc.weight", "head.weight"]:
                    if key in state_dict and state_dict[key].ndim == 2:
                        n_classes = state_dict[key].shape[0]
                        break
            except Exception:
                pass

        self.model = EEGNet(
            n_classes=n_classes,
            n_channels=self.N_CHANNELS,
            n_times=self.N_TIMES,
            dropout_rate=cfg["dropout"],
            filters=cfg["filters"],
            kernel_time=cfg["kernel_time"],
            pool=(cfg["pool"], cfg["pool2"]),
        )
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                pass  # 加载失败就用随机初始化
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, eeg_data, metadata=None):
        """预处理：取前19通道 + 裁剪/填充到500点 + 标准化"""
        data = np.asarray(eeg_data, dtype=np.float64)
        data = data[:self.N_CHANNELS, :]  # 取前19个EEG通道

        # 统一长度到500
        n_times = data.shape[1]
        if n_times > self.N_TIMES:
            data = data[:, :self.N_TIMES]
        elif n_times < self.N_TIMES:
            padding = np.zeros((data.shape[0], self.N_TIMES - n_times))
            data = np.concatenate([data, padding], axis=1)

        # 标准化：优先用训练集的逐通道统计量（与训练一致，且保留幅值差异）；
        # 没有统计量文件（如尚未训练）时退回逐样本全局标准化
        if self.norm_mean is not None and self.norm_std is not None \
                and self.norm_mean.shape[1] == data.shape[0]:
            data = (data - self.norm_mean[0]) / self.norm_std[0]
        else:
            data = (data - data.mean()) / (data.std() + 1e-8)

        # 增加batch维度，转tensor
        data = data[np.newaxis, :, :]  # (1, 19, 500)
        return torch.FloatTensor(data).to(self.device)

    def _inference(self, processed_data):
        """前向推理"""
        with torch.no_grad():
            logits = self.model(processed_data)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        return prob

    def _postprocess(self, output):
        output = np.asarray(output, dtype=np.float64).reshape(-1)
        return {
            "label": int(np.argmax(output)),
            "prob": output.tolist(),
            "confidence": float(np.max(output)),
            "extra": {"model_type": "eegnet"}
        }
