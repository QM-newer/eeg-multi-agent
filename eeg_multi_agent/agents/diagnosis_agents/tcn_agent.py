# agents/diagnosis_agents/tcn_agent.py

import numpy as np
import torch
from agents.base_agent import BaseAgent
from config import DATA_CONFIG, MODEL_CONFIG
from models.tcn import TCN


class TcnAgent(BaseAgent):
    """
    TCN诊断Agent
    输入：原始EEG时序 (n_channels, n_times)
    输出：三分类概率

    模型结构超参统一来自 config.MODEL_CONFIG["tcn"]
    """

    N_CHANNELS = DATA_CONFIG["n_eeg_channels"]   # 19

    def __init__(self, name, model_path=None, device="cpu", n_classes=None):
        self._n_classes_override = n_classes
        super().__init__(name, model_path, device)
        from data.epoch_cache import load_norm_stats
        self.norm_mean, self.norm_std = load_norm_stats()
        if self.model is None:
            self._load_model()

    def _load_model(self):
        cfg = MODEL_CONFIG["tcn"]
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

        self.model = TCN(
            n_classes=n_classes,
            n_channels=self.N_CHANNELS,
            num_layers=cfg["n_layers"],
            hidden_channels=cfg["hidden_channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
            dilation_base=cfg["dilation_base"],
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
        """预处理：取前19通道 + 标准化"""
        data = np.asarray(eeg_data, dtype=np.float64)
        data = data[:self.N_CHANNELS, :]
        # 优先用训练集的逐通道统计量（与训练一致），否则退回逐样本全局标准化
        if self.norm_mean is not None and self.norm_std is not None \
                and self.norm_mean.shape[1] == data.shape[0]:
            data = (data - self.norm_mean[0]) / self.norm_std[0]
        else:
            data = (data - data.mean()) / (data.std() + 1e-8)
        data = data[np.newaxis, :, :]  # (1, 19, T)
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
            "extra": {"model_type": "tcn"}
        }
