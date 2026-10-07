# scripts/evaluate_binary.py
"""
二分类评估 + 置信度 Borderline 判定

核心逻辑：
    1. 加载二分类模型（Normal vs Abnormal）
    2. 被试级聚合概率
    3. 置信度判可疑：
       - P(Abnormal) ∈ [lo, hi] → Borderline/可疑
       - P(Abnormal) < lo → Normal
       - P(Abnormal) > hi → Abnormal
    4. 同时出 2-class 和 3-class-with-confidence 指标

用法：
    python scripts/evaluate_binary.py --split test
    python scripts/evaluate_binary.py --split test --borderline-range 0.35 0.65
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
)

from config import (
    BINARY_MODEL_PATHS, BINARY_CLASS_NAMES, BINARY_LABEL_MAP,
    BORDERLINE_CONFIDENCE_RANGE, CLASS_NAMES, RANDOM_SEED, resolve_path,
)
from training.common import prepare_dataset, remap_labels_binary
from utils.io_utils import ensure_dir, save_json

from agents.diagnosis_agents.lightgbm_agent import LightGbmAgent
from agents.diagnosis_agents.eegnet_agent import EegNetAgent
from agents.diagnosis_agents.tcn_agent import TcnAgent
from agents.diagnosis_agents.transformer_agent import TransformerAgent


BINARY_AGENT_FACTORY = {
    "lightgbm": (LightGbmAgent, "checkpoints/lightgbm_binary/model.pkl"),
    "eegnet": (EegNetAgent, "checkpoints/eegnet_binary/best.pth"),
    "tcn": (TcnAgent, "checkpoints/tcn_binary/best.pth"),
    "transformer": (TransformerAgent, "checkpoints/transformer_binary/best.pth"),
}


def trained_binary_agents():
    """返回已有二分类权重的 Agent 列表"""
    out = []
    for name, (_, rel) in BINARY_AGENT_FACTORY.items():
        if os.path.exists(resolve_path(rel)):
            out.append(name)
    return out


def aggregate_subject(probs, sids, uniq, method="mean"):
    """按被试聚合二分类概率"""
    if method == "mean":
        return np.vstack([probs[sids == u].mean(axis=0) for u in uniq])
    elif method == "max":
        return np.vstack([probs[sids == u].max(axis=0) for u in uniq])
    elif method == "p90":
        return np.vstack([np.percentile(probs[sids == u], 90, axis=0) for u in uniq])
    else:
        raise ValueError(f"未知聚合方式: {method}")


def apply_borderline_threshold(p_abnormal, lo, hi):
    """
    将二分类概率映射为三分类标签（含置信度可疑区间）

    P(Abnormal) < lo  → 0 (Normal)
    P(Abnormal) > hi  → 2 (Abnormal)
    else              → 1 (Borderline/可疑)
    """
    labels = np.ones(len(p_abnormal), dtype=np.int64)   # 默认 Borderline
    labels[p_abnormal < lo] = 0   # Normal
    labels[p_abnormal > hi] = 2   # Abnormal
    return labels


def main(split="test", max_epochs_per_record=20, agents=None,
         agg_method="mean", borderline_range=None):
    if borderline_range is None:
        borderline_range = BORDERLINE_CONFIDENCE_RANGE
    lo, hi = borderline_range

    print("=" * 74)
    print(f"二分类评估 + 置信度 Borderline（{split} 集）")
    print(f"  置信度阈值: P(Abnormal) ∈ [{lo}, {hi}] → Borderline/可疑")
    print(f"  聚合方式: {agg_method}")
    print("=" * 74)

    # 加载数据（三分类标签，后面做映射）
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize=None, verbose=False, label_mode="3class")
    idx = data["test_idx"] if split == "test" else data["val_idx"]
    X = data["X"][idx]
    y_3class = data["y"][idx]       # 原始三分类标签（用于评估）
    y_binary = remap_labels_binary(y_3class)  # 二分类标签
    sids = np.asarray(data["subject_ids"])[idx]

    names = agents.split(",") if agents else trained_binary_agents()
    if not names:
        print("  未找到已训练的二分类模型，请先运行: python scripts/train_binary_agents.py")
        return

    print(f"参与评估的Agent: {names}")
    print(f"样本: {X.shape[0]} 个 epoch，{len(set(sids))} 个被试")

    uniq = sorted(set(sids))
    sub_y_3class = np.array([y_3class[sids == u][0] for u in uniq])
    sub_y_binary = np.array([y_binary[sids == u][0] for u in uniq])

    # 基线
    base_binary = max(np.bincount(sub_y_binary).tolist()) / len(sub_y_binary)
    base_3class = max(np.bincount(sub_y_3class).tolist()) / len(sub_y_3class)
    print(f"  二分类基线: {base_binary:.4f}  三分类基线: {base_3class:.4f}")

    report = {
        "split": split, "n_epochs": int(X.shape[0]),
        "n_subjects": len(uniq), "borderline_range": borderline_range,
        "agg_method": agg_method, "agents": {},
    }

    for name in names:
        cls, rel = BINARY_AGENT_FACTORY[name]
        model_path = resolve_path(rel)
        agent = cls(name, model_path=model_path)

        # 二分类推理
        probs = np.zeros((X.shape[0], 2), dtype=np.float64)
        t0 = time.time()
        for i in range(X.shape[0]):
            r = agent.predict(X[i])
            p = np.asarray(r["prob"], dtype=np.float64)
            # 兼容：如果模型输出 3 类概率（加载了旧的3分类模型），取后两类的和作为 Abnormal
            if len(p) == 3:
                probs[i] = np.array([p[0], p[1] + p[2]])
            elif len(p) == 2:
                probs[i] = p
            else:
                probs[i] = np.array([1 - p[0], p[0]]) if len(p) == 1 else p
        elapsed = time.time() - t0

        # 被试级聚合
        P = aggregate_subject(probs, sids, uniq, method=agg_method)
        p_abnormal = P[:, 1]   # P(Abnormal)

        # ---- 二分类评估 ----
        sub_pred_binary = (p_abnormal >= 0.5).astype(int)
        bin_acc = accuracy_score(sub_y_binary, sub_pred_binary)
        bin_f1 = f1_score(sub_y_binary, sub_pred_binary, average="macro")
        bin_auc = roc_auc_score(sub_y_binary, p_abnormal)

        # ---- 置信度 Borderline 评估 ----
        sub_pred_3class = apply_borderline_threshold(p_abnormal, lo, hi)
        # 只有原始三分类中 Normal(0) 和 Abnormal(2) 的被试才能算"正确"
        # Borderline 原始标签(1) → 无论模型怎么判都不算错（临床模糊地带）
        mask_decisive = sub_y_3class != 1   # 去掉 Borderline 标签的被试
        acc_3class = accuracy_score(sub_y_3class[mask_decisive], sub_pred_3class[mask_decisive])
        f1_3class = f1_score(sub_y_3class[mask_decisive], sub_pred_3class[mask_decisive],
                             average="macro", labels=[0, 2])

        # 全量三分类指标（Borderline 也要评估）
        f1_3class_all = f1_score(sub_y_3class, sub_pred_3class, average="macro")
        n_borderline_pred = int((sub_pred_3class == 1).sum())

        agent_report = {
            "seconds": round(elapsed, 1),
            "binary": {
                "accuracy": round(float(bin_acc), 4),
                "macro_f1": round(float(bin_f1), 4),
                "auc": round(float(bin_auc), 4),
            },
            "confidence_borderline": {
                "accuracy_non_borderline": round(float(acc_3class), 4),
                "macro_f1_non_borderline": round(float(f1_3class), 4),
                "macro_f1_all": round(float(f1_3class_all), 4),
                "n_predicted_borderline": n_borderline_pred,
            },
        }
        report["agents"][name] = agent_report

        print(f"\n  [{name}]  ({elapsed:.0f}s)")
        print(f"    二分类: acc={bin_acc:.4f}  F1={bin_f1:.4f}  AUC={bin_auc:.4f}  "
              f"(基线 {base_binary:.4f})")
        print(f"    置信度3分类: acc(非Borderline)={acc_3class:.4f}  F1(非BL)={f1_3class:.4f}  "
              f"F1(全量)={f1_3class_all:.4f}  判可疑={n_borderline_pred}人")

    # 汇总
    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    for name in names:
        r = report["agents"][name]
        print(f"  {name}: binary AUC={r['binary']['auc']:.4f}  "
              f"binary acc={r['binary']['accuracy']:.4f}  "
              f"conf3class F1={r['confidence_borderline']['macro_f1_all']:.4f}")

    # 最优 Agent
    best_name = max(names, key=lambda a: report["agents"][a]["binary"]["auc"])
    best_auc = report["agents"][best_name]["binary"]["auc"]
    print(f"\n  最优 Agent: {best_name} (AUC={best_auc:.4f})")

    out_path = resolve_path("outputs/binary_eval.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(report, out_path)
    print(f"\n结果已保存: {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-epochs-per-record", type=int, default=20)
    ap.add_argument("--agents", default=None, help="逗号分隔；默认自动挑选已训练的")
    ap.add_argument("--agg-method", default="mean", help="聚合方式: mean/max/p90")
    ap.add_argument("--borderline-range", nargs=2, type=float, default=None,
                    help="置信度 Borderline 判定区间，如 0.35 0.65")
    a = ap.parse_args()

    brange = tuple(a.borderline_range) if a.borderline_range else None
    main(split=a.split, max_epochs_per_record=a.max_epochs_per_record,
         agents=a.agents, agg_method=a.agg_method, borderline_range=brange)
