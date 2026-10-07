# agents/diagnosis_agents/transformer_agent.py

import numpy as np
import torch
from agents.base_agent import BaseAgent
from config import DATA_CONFIG, MODEL_CONFIG, N_TIMES
from models.eeg_transformer import EEGTransformer


class TransformerAgent(BaseAgent):
    """
    Transformer诊断Agent
    输入：原始EEG时序 (n_channels, n_times)
    输出：三分类概率

    模型结构超参统一来自 config.MODEL_CONFIG["transformer"]
    """

    N_CHANNELS = DATA_CONFIG["n_eeg_channels"]   # 19
    N_TIMES = N_TIMES                            # 500
    PATCH_SIZE = MODEL_CONFIG["transformer"]["patch_size"]

    def __init__(self, name, model_path=None, device="cpu", n_classes=None):
        self._n_classes_override = n_classes
        super().__init__(name, model_path, device)
        from data.epoch_cache import load_norm_stats
        self.norm_mean, self.norm_std = load_norm_stats()
        if self.model is None:
            self._load_model()

    def _load_model(self):
        cfg = MODEL_CONFIG["transformer"]
        n_classes = self._n_classes_override or cfg["n_classes"]
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location="cpu")
                # 查找分类层：2D权重矩阵才是Linear，1D是LayerNorm
                for key in ["classifier.1.weight", "classifier.weight",
                            "fc.weight", "head.weight"]:
                    if key in state_dict and state_dict[key].ndim == 2:
                        n_classes = state_dict[key].shape[0]
                        break
            except Exception:
                pass

        self.model = EEGTransformer(
            n_classes=n_classes,
            n_channels=self.N_CHANNELS,
            n_times=self.N_TIMES,
            patch_size=self.PATCH_SIZE,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        )
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                pass
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, eeg_data, metadata=None):
        """预处理：取前19通道 + 裁剪/填充到500点 + 标准化"""
        data = np.asarray(eeg_data, dtype=np.float64)
        data = data[:self.N_CHANNELS, :]

        # 统一长度到500（patch_size的整数倍）
        n_times = data.shape[1]
        if n_times > self.N_TIMES:
            data = data[:, :self.N_TIMES]
        elif n_times < self.N_TIMES:
            padding = np.zeros((data.shape[0], self.N_TIMES - n_times))
            data = np.concatenate([data, padding], axis=1)

        # 优先用训练集的逐通道统计量（与训练一致），否则退回逐样本全局标准化
        if self.norm_mean is not None and self.norm_std is not None \
                and self.norm_mean.shape[1] == data.shape[0]:
            data = (data - self.norm_mean[0]) / self.norm_std[0]
        else:
            data = (data - data.mean()) / (data.std() + 1e-8)
        data = data[np.newaxis, :, :]  # (1, 19, 500)
        return torch.FloatTensor(data).to(self.device)

    def _inference(self, processed_data):
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
            "extra": {"model_type": "transformer"}
        }
