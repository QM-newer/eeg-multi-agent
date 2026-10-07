# scripts/train_stacking.py
"""
Stacking 元模型融合

在 val 集上：
  1. 4 个二分类 Agent 各自推理 → 每被试 4 个 P(Abnormal) 特征
  2. 训练 LogisticRegression / LightGBM 元模型
  3. 在 test 集上评估

用法：
    python scripts/train_stacking.py --split test
    python scripts/train_stacking.py --meta-model lightgbm
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

from config import (
    BINARY_MODEL_PATHS, BINARY_CLASS_NAMES, BINARY_LABEL_MAP,
    CLASS_NAMES, RANDOM_SEED, resolve_path,
)
from training.common import prepare_dataset, remap_labels_binary
from data.epoch_cache import compute_class_weights
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
    out = []
    for name, (_, rel) in BINARY_AGENT_FACTORY.items():
        if os.path.exists(resolve_path(rel)):
            out.append(name)
    return out


def predict_agents(X, agent_names, agent_factory):
    """对所有 epoch 跑全部 Agent，返回 (n_epochs, n_agents, 2) 概率"""
    n = X.shape[0]
    n_agents = len(agent_names)
    probs = np.zeros((n, n_agents, 2), dtype=np.float64)

    for k, name in enumerate(agent_names):
        cls, rel = agent_factory[name]
        agent = cls(name, model_path=resolve_path(rel))
        t0 = time.time()
        for i in range(n):
            r = agent.predict(X[i])
            p = np.asarray(r["prob"], dtype=np.float64)
            if len(p) == 3:
                p = np.array([p[0], p[1] + p[2]])
            elif len(p) == 1:
                p = np.array([1 - p[0], p[0]])
            probs[i, k, :] = p[:2]
        print(f"    [{name}] {time.time()-t0:.0f}s")
    return probs


def aggregate_subject_features(probs, sids, uniq, method="mean"):
    """按被试聚合各 Agent 的 P(Abnormal)，返回 (n_subjects, n_agents) 特征矩阵"""
    # probs: (n_epochs, n_agents, 2)
    # 取 P(Abnormal) = probs[:, :, 1]
    p_abn = probs[:, :, 1]   # (n_epochs, n_agents)
    if method == "mean":
        return np.vstack([p_abn[sids == u].mean(axis=0) for u in uniq])
    elif method == "max":
        return np.vstack([p_abn[sids == u].max(axis=0) for u in uniq])
    elif method == "p90":
        return np.vstack([np.percentile(p_abn[sids == u], 90, axis=0) for u in uniq])
    else:
        raise ValueError(f"未知聚合: {method}")


def main(split="test", max_epochs_per_record=20, meta_model="lr",
         agg_method="mean", agents=None):
    print("=" * 74)
    print(f"Stacking 元模型融合（元模型={meta_model}，聚合={agg_method}）")
    print("=" * 74)

    # 加载三分类数据
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize=None, verbose=False, label_mode="3class")

    agent_names = agents.split(",") if agents else trained_binary_agents()
    if not agent_names:
        print("  未找到已训练的二分类模型")
        return

    print(f"  Agent: {agent_names}")

    # ---- Step 1: 在 val 集上收集元特征 ----
    print("\n[Step 1] val 集：收集 Agent 预测作为元特征...")
    val_idx = data["val_idx"]
    X_val = data["X"][val_idx]
    y_val_3class = data["y"][val_idx]
    y_val_binary = remap_labels_binary(y_val_3class)
    sids_val = np.asarray(data["subject_ids"])[val_idx]

    val_probs = predict_agents(X_val, agent_names, BINARY_AGENT_FACTORY)

    uniq_val = sorted(set(sids_val))
    meta_X_val = aggregate_subject_features(val_probs, sids_val, uniq_val, method=agg_method)
    meta_y_val = np.array([y_val_binary[sids_val == u][0] for u in uniq_val])

    print(f"  val: {len(uniq_val)} 被试, 元特征维度: {meta_X_val.shape[1]}")
    print(f"  元特征范围: P(Abnormal) ∈ [{meta_X_val.min():.3f}, {meta_X_val.max():.3f}]")
    print(f"  元特征均值 per Agent: {np.round(meta_X_val.mean(axis=0), 3)}")

    # ---- Step 2: 训练元模型 ----
    print(f"\n[Step 2] 训练元模型 ({meta_model})...")
    if meta_model == "lr":
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                  class_weight="balanced")
    elif meta_model == "lightgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=50, learning_rate=0.1, num_leaves=8,
            objective="binary", random_state=42, verbose=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"未知元模型: {meta_model}")

    clf.fit(meta_X_val, meta_y_val)

    # val 上元模型表现
    val_pred = clf.predict(meta_X_val)
    val_prob = clf.predict_proba(meta_X_val)[:, 1] if hasattr(clf, "predict_proba") else val_pred
    val_acc_meta = accuracy_score(meta_y_val, val_pred)
    val_auc_meta = roc_auc_score(meta_y_val, val_prob)
    print(f"  元模型 val: acc={val_acc_meta:.4f}  AUC={val_auc_meta:.4f}")

    # ---- Step 3: 在 test 集上评估 ----
    print(f"\n[Step 3] test 集：评估 Stacking 融合...")
    test_idx = data["test_idx"]
    X_test = data["X"][test_idx]
    y_test_3class = data["y"][test_idx]
    y_test_binary = remap_labels_binary(y_test_3class)
    sids_test = np.asarray(data["subject_ids"])[test_idx]

    test_probs = predict_agents(X_test, agent_names, BINARY_AGENT_FACTORY)

    uniq_test = sorted(set(sids_test))
    meta_X_test = aggregate_subject_features(test_probs, sids_test, uniq_test, method=agg_method)
    meta_y_test = np.array([y_test_binary[sids_test == u][0] for u in uniq_test])
    meta_y_test_3class = np.array([y_test_3class[sids_test == u][0] for u in uniq_test])

    # 元模型预测
    test_pred = clf.predict(meta_X_test)
    test_prob = clf.predict_proba(meta_X_test)[:, 1] if hasattr(clf, "predict_proba") else test_pred.astype(float)

    # 二分类指标
    stack_acc = accuracy_score(meta_y_test, test_pred)
    stack_f1 = f1_score(meta_y_test, test_pred, average="macro")
    stack_auc = roc_auc_score(meta_y_test, test_prob)

    # 各单 Agent 对比（被试级）
    base_binary = max(np.bincount(meta_y_test).tolist()) / len(meta_y_test)

    # 各单 Agent 的被试级结果
    p_abn_test = test_probs[:, :, 1]
    agent_results = {}
    for k, name in enumerate(agent_names):
        if agg_method == "mean":
            agent_p = np.array([p_abn_test[sids_test == u, k].mean() for u in uniq_test])
        elif agg_method == "max":
            agent_p = np.array([p_abn_test[sids_test == u, k].max() for u in uniq_test])
        elif agg_method == "p90":
            agent_p = np.array([np.percentile(p_abn_test[sids_test == u, k], 90) for u in uniq_test])

        agent_pred = (agent_p >= 0.5).astype(int)
        agent_acc = accuracy_score(meta_y_test, agent_pred)
        agent_auc = roc_auc_score(meta_y_test, agent_p)
        agent_results[name] = {"acc": round(float(agent_acc), 4),
                                "auc": round(float(agent_auc), 4)}

    # 软投票融合（平均概率后投票）
    if agg_method == "mean":
        soft_p = np.array([p_abn_test[sids_test == u].mean(axis=0) for u in uniq_test]).mean(axis=1)
    elif agg_method == "max":
        soft_p = np.array([p_abn_test[sids_test == u].max(axis=0) for u in uniq_test]).mean(axis=1)
    elif agg_method == "p90":
        soft_p = np.array([np.percentile(p_abn_test[sids_test == u], 90, axis=0) for u in uniq_test]).mean(axis=1)

    soft_pred = (soft_p >= 0.5).astype(int)
    soft_acc = accuracy_score(meta_y_test, soft_pred)
    soft_auc = roc_auc_score(meta_y_test, soft_p)

    # ---- 输出 ----
    print("\n" + "=" * 74)
    print("结果对比（test 集，被试级二分类）")
    print("=" * 74)
    print(f"  {'方法':<25s} {'Acc':>6s} {'AUC':>6s}")
    print(f"  {'─'*25} {'─'*6} {'─'*6}")
    print(f"  {'基线(多数类)':<25s} {base_binary:.4f} {'─':>6s}")
    for name in agent_names:
        r = agent_results[name]
        print(f"  {name+' (单Agent)':<25s} {r['acc']:.4f} {r['auc']:.4f}")
    print(f"  {'软投票(平均概率)':<25s} {soft_acc:.4f} {soft_auc:.4f}")
    print(f"  {'Stacking('+meta_model+')':<25s} {stack_acc:.4f} {stack_auc:.4f}")

    # 保存结果
    report = {
        "meta_model": meta_model, "agg_method": agg_method,
        "split": split, "n_subjects": len(uniq_test),
        "stacking": {"acc": round(float(stack_acc), 4), "auc": round(float(stack_auc), 4),
                     "f1": round(float(stack_f1), 4)},
        "soft_voting": {"acc": round(float(soft_acc), 4), "auc": round(float(soft_auc), 4)},
        "single_agents": agent_results,
        "baseline": round(float(base_binary), 4),
    }

    out_path = resolve_path("outputs/stacking_eval.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(report, out_path)
    print(f"\n结果已保存: {out_path}")

    # 保存元模型
    import joblib
    meta_path = resolve_path("checkpoints/stacking/meta_model.joblib")
    ensure_dir(os.path.dirname(meta_path))
    joblib.dump(clf, meta_path)
    print(f"元模型已保存: {meta_path}")

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-epochs-per-record", type=int, default=20)
    ap.add_argument("--meta-model", default="lr", help="lr 或 lightgbm")
    ap.add_argument("--agg-method", default="mean", help="mean/max/p90")
    ap.add_argument("--agents", default=None, help="逗号分隔；默认全部")
    a = ap.parse_args()
    main(split=a.split, max_epochs_per_record=a.max_epochs_per_record,
         meta_model=a.meta_model, agg_method=a.agg_method, agents=a.agents)
