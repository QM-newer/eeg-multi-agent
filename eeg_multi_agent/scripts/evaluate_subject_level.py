# scripts/evaluate_subject_level.py
"""
被试级评估：把每个被试所有 epoch 的预测概率平均后投票，再算指标

为什么必须做这一步：
    实测 epoch 级最佳单特征 AUC 仅 0.59，按被试平均后升到 0.65 —— 判别信息主要存在于
    "被试整体"而非单个 2 秒片段。临床场景也是"给一段记录下一个结论"，故被试级才是主指标。

用法：
    python scripts/evaluate_subject_level.py --max-epochs-per-record 20 --split test
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from config import CLASS_NAMES, RANDOM_SEED
from training.common import prepare_dataset
from utils.io_utils import ensure_dir, save_json
from config import resolve_path

from agents.diagnosis_agents.lightgbm_agent import LightGbmAgent
from agents.diagnosis_agents.eegnet_agent import EegNetAgent
from agents.diagnosis_agents.tcn_agent import TcnAgent
from agents.diagnosis_agents.transformer_agent import TransformerAgent

AGENT_FACTORY = {
    "lightgbm": (LightGbmAgent, "checkpoints/lightgbm/model.pkl"),
    "eegnet": (EegNetAgent, "checkpoints/eegnet/best.pth"),
    "tcn": (TcnAgent, "checkpoints/tcn/best.pth"),
    "transformer": (TransformerAgent, "checkpoints/transformer/best.pth"),
}


def trained_agents(since_ts=None):
    """
    返回已在真实数据上训练完成的 Agent 名称列表
    判据：权重文件的修改时间晚于 epoch 缓存清单（缓存构建完成时刻）
    """
    out = []
    for name, (_, rel) in AGENT_FACTORY.items():
        path = resolve_path(rel)
        if not os.path.exists(path):
            continue
        if since_ts is not None and os.path.getmtime(path) < since_ts:
            continue
        out.append(name)
    return out


def aggregate_subject(probs, sids, uniq, method="mean"):
    """
    按被试聚合概率，支持三种方式：
    
    - mean: 各 epoch 概率取平均（当前默认，适合连续性异常）
    - max:  取各 epoch 最大概率（一个异常 epoch 即判异常，适合一过性事件）
    - p90:  取 90 分位数（折中：对极端值敏感但不过度）
    
    返回:
        (n_subjects, n_classes) 概率矩阵
    """
    if method == "mean":
        return np.vstack([probs[sids == u].mean(axis=0) for u in uniq])
    elif method == "max":
        return np.vstack([probs[sids == u].max(axis=0) for u in uniq])
    elif method == "p90":
        return np.vstack([np.percentile(probs[sids == u], 90, axis=0) for u in uniq])
    else:
        raise ValueError(f"未知聚合方式: {method}，可选: mean / max / p90")


def main(split="test", max_epochs_per_record=20, agents=None, n_limit=None,
         since_ts=None, agg_methods=None):
    if agg_methods is None:
        agg_methods = ["mean"]
    print("=" * 74)
    print(f"被试级评估（{split} 集，每记录最多 {max_epochs_per_record} 个 epoch）")
    print(f"聚合方式: {agg_methods}")
    print("=" * 74)

    # 关键：这里取 RAW(µV) 数据。训练时 prepare_dataset(normalize="per_channel") 的输出
    # 等价于"Agent 内部归一化之后"的结果；Agent 推理时自己会再按训练集统计量归一化，
    # 若这里再传归一化后的数据就会二次归一化，导致输入尺度错误（预测退化为单一类）。
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize=None, verbose=False)
    idx = data["test_idx"] if split == "test" else data["val_idx"]
    X = data["X"][idx]
    y = data["y"][idx]
    sids = np.asarray(data["subject_ids"])[idx]

    if n_limit and X.shape[0] > n_limit:
        rng = np.random.default_rng(RANDOM_SEED)
        sel = np.sort(rng.choice(X.shape[0], n_limit, replace=False))
        X, y, sids = X[sel], y[sel], sids[sel]

    names = agents.split(",") if agents else trained_agents(since_ts)
    print(f"参与评估的Agent: {names}")
    print(f"样本: {X.shape[0]} 个 epoch，{len(set(sids))} 个被试")

    uniq = sorted(set(sids))
    sub_y = np.array([y[sids == u][0] for u in uniq])
    report = {"split": split, "n_epochs": int(X.shape[0]),
              "n_subjects": len(uniq), "agg_methods": agg_methods, "agents": {}}

    for name in names:
        cls, rel = AGENT_FACTORY[name]
        agent = cls(name, model_path=resolve_path(rel))
        probs = np.zeros((X.shape[0], 3), dtype=np.float64)
        t0 = time.time()
        for i in range(X.shape[0]):
            r = agent.predict(X[i])
            probs[i] = r["prob"]
        ep_pred = probs.argmax(axis=1)
        ep_acc = accuracy_score(y, ep_pred)

        agent_report = {
            "epoch_accuracy": round(float(ep_acc), 4),
            "seconds": round(time.time() - t0, 1),
            "aggregation": {},
        }

        print(f"\n  [{name}]  epoch级 acc={ep_acc:.4f}  ({agent_report['seconds']}s)")

        # 遍历所有聚合方式
        for method in agg_methods:
            P = aggregate_subject(probs, sids, uniq, method=method)
            sub_pred = P.argmax(axis=1)
            sub_acc = accuracy_score(sub_y, sub_pred)
            sub_f1 = f1_score(sub_y, sub_pred, average="macro")
            cm = confusion_matrix(sub_y, sub_pred, labels=[0, 1, 2]).tolist()

            agent_report["aggregation"][method] = {
                "subject_accuracy": round(float(sub_acc), 4),
                "subject_macro_f1": round(float(sub_f1), 4),
                "subject_confusion_matrix": cm,
            }
            print(f"    {method:>4s}: 被试级 acc={sub_acc:.4f}  macro-F1={sub_f1:.4f}")

        report["agents"][name] = agent_report

    # 多数类基线（对照）
    base = max(np.bincount(sub_y).tolist()) / len(sub_y)
    report["majority_baseline_subject"] = round(float(base), 4)
    print(f"\n  多数类基线（被试级）: {base:.4f}")

    # 聚合方式对比摘要
    if len(agg_methods) > 1:
        print(f"\n  聚合方式对比摘要（跨 Agent 平均）：")
        for method in agg_methods:
            accs = [report["agents"][a]["aggregation"][method]["subject_accuracy"]
                    for a in names]
            print(f"    {method:>4s}: avg acc = {np.mean(accs):.4f}")
        # 最佳单 Agent 对比
        for method in agg_methods:
            best = max(names, key=lambda a: report["agents"][a]["aggregation"][method]["subject_accuracy"])
            best_acc = report["agents"][best]["aggregation"][method]["subject_accuracy"]
            print(f"    {method:>4s}: best = {best} ({best_acc:.4f})")

    out = resolve_path("outputs/subject_level_eval.json")
    ensure_dir(os.path.dirname(out))
    save_json(report, out)
    print(f"\n结果已保存: {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-epochs-per-record", type=int, default=20)
    ap.add_argument("--agents", default=None, help="逗号分隔；默认自动挑选已训练完成的")
    ap.add_argument("--n-limit", type=int, default=None)
    ap.add_argument("--since-ts", type=float, default=None,
                    help="只统计权重文件晚于该时间戳的Agent（默认用缓存清单时间）")
    ap.add_argument("--agg-methods", default="mean,max,p90",
                    help="聚合方式（逗号分隔），可选: mean / max / p90。默认全部对比")
    a = ap.parse_args()

    since = a.since_ts
    if since is None:
        from config import DATA_CONFIG
        mpath = os.path.join(DATA_CONFIG["epoch_cache_dir"], "manifest.csv")
        since = os.path.getmtime(mpath) if os.path.exists(mpath) else None
    
    agg_methods = [m.strip() for m in a.agg_methods.split(",") if m.strip()]
    main(split=a.split, max_epochs_per_record=a.max_epochs_per_record,
         agents=a.agents, n_limit=a.n_limit, since_ts=since,
         agg_methods=agg_methods)
