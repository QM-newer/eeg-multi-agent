# orchestrator/orchestrator.py

import os

import numpy as np
from config import (
    AGENT_WEIGHTS, AGENT_WEIGHTS_PATH, DISAGREEMENT_CONFIG, FUSION_PATH,
    MODEL_PATHS, ORCHESTRATOR_CONFIG, QUALITY_CONFIG, DATA_CONFIG,
)
from utils.signal_utils import compute_quality_report

from agents.feature_agents.time_domain_agent import TimeDomainAgent
from agents.feature_agents.freq_domain_agent import FreqDomainAgent
from agents.feature_agents.timefreq_agent import TimeFreqAgent

from agents.diagnosis_agents.lightgbm_agent import LightGbmAgent
from agents.diagnosis_agents.eegnet_agent import EegNetAgent
from agents.diagnosis_agents.tcn_agent import TcnAgent
from agents.diagnosis_agents.transformer_agent import TransformerAgent

from agents.validation_agents.explainability_agent import ExplainabilityAgent
from agents.validation_agents.consistency_agent import ConsistencyAgent

from orchestrator.voting import weighted_vote
from orchestrator.disagreement import check_disagreement, second_opinion
from orchestrator.dynamic_weight import adjust_weights_by_snr
from orchestrator.arbitration import arbitrate


class Orchestrator:
    """
    多Agent诊断系统协调器

    工作流程（对应文档 4.1 / 4.2）：
    1. 信号质量评估（SNR + 伪迹比例 → 三档质量等级）
    2. SNR 感知的动态权重调整
    3. 特征提取（三个特征Agent）
    4. 第一轮诊断（四个诊断Agent）
    5. 加权投票
    6. 分歧检测
    7. 若有分歧：补充空间域特征 → 重推理 → 特征可信度裁决 → 重新投票
    8. 一致性校验 + 可解释性分析
    9. 返回最终诊断结果
    """

    def __init__(self, model_paths=None, device="cpu", sfreq=None,
                 fusion=None, auto_load=True):
        self.device = device
        self.disagreement_config = DISAGREEMENT_CONFIG
        self.sfreq = sfreq or DATA_CONFIG.get("target_sfreq", 250)

        # 未显式指定时，自动加载 checkpoints 下已训练好的权重
        if model_paths is None and auto_load:
            model_paths = {k: v for k, v in MODEL_PATHS.items() if os.path.exists(v)}
        self.model_paths = model_paths or {}

        # 投票权重：优先用训练阶段按验证集准确率归一化得到的权重
        self.weights = self._load_weights()
        # 输入EEG通道数（超过时截取前 n_eeg_channels 个，保证与训练一致）
        self.n_eeg_channels = DATA_CONFIG.get("n_eeg_channels", 19)
        self._fusion_warned = False
        # 特征融合器（mRMR）：训练阶段拟合后保存，推理时复用同一特征子集
        self.fusion = fusion if fusion is not None else self._load_fusion()
        # 逐通道归一化统计量：与训练/各诊断Agent共用同一变换
        from data.epoch_cache import load_norm_stats
        self.norm_mean, self.norm_std = load_norm_stats()

        self._init_feature_agents()
        self._init_diagnosis_agents()
        self._init_validation_agents()

    def _load_weights(self):
        """加载训练阶段回写的权重，否则用 config.AGENT_WEIGHTS"""
        if os.path.exists(AGENT_WEIGHTS_PATH):
            try:
                from utils.io_utils import load_json
                data = load_json(AGENT_WEIGHTS_PATH) or {}
                weights = data.get("weights")
                if weights:
                    return {**AGENT_WEIGHTS, **weights}
            except Exception:
                pass
        return AGENT_WEIGHTS.copy()

    def _load_fusion(self):
        """加载训练阶段拟合的 mRMR 融合器（不存在则不做特征选择）"""
        if os.path.exists(FUSION_PATH):
            try:
                from features.fusion import FeatureFusion
                return FeatureFusion.load(FUSION_PATH)
            except Exception:
                return None
        return None

    def _init_feature_agents(self):
        """初始化三个特征Agent"""
        self.feature_agents = [
            TimeDomainAgent("time_domain"),
            FreqDomainAgent("freq_domain", sfreq=250),
            TimeFreqAgent("timefreq"),
        ]

    def _init_diagnosis_agents(self):
        """初始化四个诊断Agent"""
        self.diagnosis_agents = [
            LightGbmAgent(
                "lightgbm",
                model_path=self.model_paths.get("lightgbm"),
                device=self.device
            ),
            EegNetAgent(
                "eegnet",
                model_path=self.model_paths.get("eegnet"),
                device=self.device
            ),
            TcnAgent(
                "tcn",
                model_path=self.model_paths.get("tcn"),
                device=self.device
            ),
            TransformerAgent(
                "transformer",
                model_path=self.model_paths.get("transformer"),
                device=self.device
            ),
        ]

    def _init_validation_agents(self):
        """初始化两个验证Agent"""
        self.validation_agents = {
            "explainability": ExplainabilityAgent("explainability"),
            "consistency": ConsistencyAgent("consistency"),
        }

    def _extract_features(self, eeg_data):
        """
        第一步：三个特征Agent提取特征 → 拼接 → mRMR 融合选择（若已训练融合器）

        注意：训练时特征提取是在"逐通道归一化"之后做的（见 training/common.prepare_dataset），
        这里必须施加同一个归一化，否则特征尺度与训练不一致（带通能量类特征尤其敏感）。
        """
        x = np.asarray(eeg_data, dtype=np.float64)
        if self.norm_mean is not None and self.norm_mean.shape[1] == x.shape[0]:
            x = (x - self.norm_mean[0]) / self.norm_std[0]

        all_features = []
        for agent in self.feature_agents:
            result = agent.predict(x)
            all_features.append(result["extra"]["features"])
        features = np.concatenate(all_features).reshape(-1)

        if self.fusion is not None:
            if features.size == self.fusion.n_input_:
                features = self.fusion.transform(features.reshape(1, -1)).reshape(-1)
            elif not self._fusion_warned:
                print(f"  [警告] 特征维度 {features.size} 与融合器期望的 "
                      f"{self.fusion.n_input_} 不一致，本次跳过 mRMR 特征选择")
                self._fusion_warned = True
        return features

    def _run_diagnosis(self, eeg_data, all_features):
        """第二步：四个诊断Agent推理"""
        results = []
        for agent in self.diagnosis_agents:
            # LightGBM从metadata传特征，其他模型吃原始信号
            metadata = {"all_features": all_features} if agent.name == "lightgbm" else None
            result = agent.predict(eeg_data, metadata)
            results.append(result)
        return results

    def _assess_quality(self, eeg_data):
        """
        信号质量评估：SNR（dB）+ 伪迹比例 → 质量分与三档等级
        见 utils.signal_utils.compute_quality_report
        """
        return compute_quality_report(eeg_data, sfreq=self.sfreq, config=QUALITY_CONFIG)

    def _adjust_weights(self, quality_report):
        """
        SNR 感知的动态权重调整（文档 4.2(3) 三档规则）
        """
        return adjust_weights_by_snr(
            self.weights,
            snr_db=quality_report["snr_db"],
            artifact_ratio=quality_report["artifact_ratio"],
        )

    def diagnose(self, eeg_data):
        """
        完整诊断流程

        参数:
            eeg_data: numpy array, shape=(n_channels, n_times)

        返回:
            dict: 最终诊断结果
        """
        # 0. 统一输入：截取前 n_eeg_channels 个EEG通道，保证与训练时一致
        eeg_data = np.asarray(eeg_data, dtype=np.float64)
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        if eeg_data.shape[0] > self.n_eeg_channels:
            eeg_data = eeg_data[:self.n_eeg_channels, :]

        # 1. 信号质量评估（SNR + 伪迹比例）
        quality_report = self._assess_quality(eeg_data)
        quality_score = quality_report["quality_score"]

        # 2. SNR 感知的动态权重调整
        current_weights = self._adjust_weights(quality_report)

        # 3. 特征提取
        all_features = self._extract_features(eeg_data)

        # 4. 第一轮诊断
        results = self._run_diagnosis(eeg_data, all_features)

        # 5. 加权投票
        vote_result = weighted_vote(results, current_weights)

        # 6. 分歧检测
        has_disagreement = check_disagreement(results, self.disagreement_config)

        # 7. 分歧仲裁：补空间域特征 → 重推理 → 特征可信度裁决 → 重新投票
        arbitration = None
        final_weights = current_weights
        used_features = all_features
        if has_disagreement and ORCHESTRATOR_CONFIG["arbitration"]["enable"]:
            results, arbitration = arbitrate(
                eeg_data, results, all_features,
                diagnosis_agents=self.diagnosis_agents,
                weights=current_weights,
                sfreq=self.sfreq,
                snr_db=quality_report["snr_db"],
                quality_level=quality_report["quality_level"],
            )
            final_weights = arbitration["weights"]
            used_features = arbitration["augmented_features"]
            vote_result = weighted_vote(results, final_weights)

        # 8. 一致性校验
        consistency_result = self.validation_agents["consistency"].predict(
            eeg_data, {"agent_results": results}
        )

        # 9. 可解释性分析（仲裁时使用扩充特征集）
        explain_result = self.validation_agents["explainability"].predict(
            eeg_data, {
                "agent_results": results,
                "all_features": used_features,
                "final_label": vote_result["label"],
            }
        )

        return {
            "final_label": vote_result["label"],
            "final_prob": vote_result["prob"],
            "confidence": vote_result["confidence"],
            "quality_score": quality_score,
            "quality_report": quality_report,
            "weights": final_weights,
            "disagreement": has_disagreement,
            "second_opinion_triggered": arbitration is not None,
            "arbitration": arbitration,
            "consistency": consistency_result["extra"],
            "explanation": explain_result["extra"]["explanation"],
            "top_features": explain_result["extra"]["top_features"],
            "agent_results": results,
        }
