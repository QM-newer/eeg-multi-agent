# scripts/compare_label_sources.py
"""
标签源对比实验：project（临床病例表） vs raw_index（原始文件编码）

目的：验证 54% 来自 raw_index 的标签是否与 project 标签同样可靠。
若两子集模型表现差异 >0.05，说明 raw_index 标签可能是噪声，需要剔除或降权。

用法：
    python scripts/compare_label_sources.py --split test
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import csv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import (
    DATA_CONFIG, CLASS_NAMES, RANDOM_SEED, resolve_path, MODEL_PATHS
)
from training.common import prepare_dataset
from utils.io_utils import ensure_dir, save_json


def load_record_index():
    """加载 record_index.csv，返回 record_id → label_source 映射"""
    ri_path = resolve_path(DATA_CONFIG["record_index"])
    source_map = {}
    with open(ri_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rec_id = row.get("record_id", "")
            ls = row.get("label_source", "unknown")
            if rec_id:
                source_map[rec_id] = ls
    return source_map


def load_manifest_label_sources():
    """从 epoch 缓存 manifest 读取 label_source（与缓存一一对应）"""
    cache_dir = DATA_CONFIG["epoch_cache_dir"]
    manifest = os.path.join(cache_dir, "manifest.csv")
    if not os.path.exists(manifest):
        return {}
    source_map = {}
    with open(manifest, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rec_id = row.get("record_id", "")
            ls = row.get("label_source", "unknown")
            if rec_id:
                source_map[rec_id] = ls
    return source_map


def get_subject_source_map(subject_ids, epoch_cache=None):
    """
    为每个 subject_id 找到其 label_source。
    
    策略：
    1. 尝试从 manifest.csv 读取（最直接）
    2. 回退到 record_index.csv
    """
    # 先试 manifest
    manifest_map = load_manifest_label_sources()
    if manifest_map:
        return manifest_map
    
    # 回退到 record_index
    return load_record_index()


def main(split="test", max_epochs_per_record=20):
    print("=" * 74)
    print(f"标签源对比实验（{split} 集）")
    print("=" * 74)

    # 加载数据（RAW µV，归一化由 Agent 内部做）
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize=None, verbose=False)
    idx = data["test_idx"] if split == "test" else data["val_idx"]
    X = data["X"][idx]
    y = data["y"][idx]
    sids = np.asarray(data["subject_ids"])[idx]

    # 读取 label_source 映射
    source_map = load_manifest_label_sources()
    if not source_map:
        source_map = load_record_index()
    
    if not source_map:
        print("  [错误] 无法获取 label_source 信息，实验终止")
        return

    # 从缓存 manifest 构建 subject_id → label_source 映射
    cache_dir = DATA_CONFIG["epoch_cache_dir"]
    manifest = os.path.join(cache_dir, "manifest.csv")
    subject_source = {}
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = row.get("subject_id", "")
                ls = row.get("label_source", "unknown")
                if sid:
                    subject_source[sid] = ls

    if not subject_source:
        print("  [错误] manifest.csv 中没有 label_source 列")
        print("  尝试从 record_index.csv 构建映射...")
        ri_path = resolve_path(DATA_CONFIG["record_index"])
        with open(ri_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = row.get("subject_id", "")
                ls = row.get("label_source", "unknown")
                if sid:
                    subject_source[sid] = ls
    
    if not subject_source:
        print("  [错误] 无法建立 subject_id → label_source 映射")
        return

    # 为每个被试标注来源
    uniq = sorted(set(sids))
    sub_source = {}
    for u in uniq:
        sub_source[u] = subject_source.get(u, "unknown")

    # 统计
    from collections import Counter
    source_counts = Counter(sub_source.values())
    print(f"\n  被试总数: {len(uniq)}")
    for src, cnt in sorted(source_counts.items()):
        print(f"    label_source={src}: {cnt} 被试")

    # 按来源分组
    groups = {}
    for u in uniq:
        src = sub_source[u]
        groups.setdefault(src, []).append(u)

    # 基线：各子组的多数类基线
    sub_y = {u: y[sids == u][0] for u in uniq}
    print(f"\n  各子组标签分布:")
    for src, members in sorted(groups.items()):
        labels = [sub_y[u] for u in members]
        dist = Counter(labels)
        base = max(dist.values()) / len(labels) if labels else 0
        print(f"    {src}: N={len(members)}  "
              f"Normal={dist.get(0,0)} Borderline={dist.get(1,0)} Abnormal={dist.get(2,0)}  "
              f"基线={base:.4f}")

    # 加载已训练的 Agent
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

    trained = []
    for name, (_, rel) in AGENT_FACTORY.items():
        if os.path.exists(resolve_path(rel)):
            trained.append(name)
    print(f"\n  已训练Agent: {trained}")

    report = {"split": split, "groups": {}}

    for agent_name in trained:
        cls, rel = AGENT_FACTORY[agent_name]
        agent = cls(agent_name, model_path=resolve_path(rel))

        print(f"\n  [{agent_name}] 推理中...")
        t0 = time.time()

        # 全部 epoch 推理
        probs = np.zeros((X.shape[0], 3), dtype=np.float64)
        for i in range(X.shape[0]):
            r = agent.predict(X[i])
            probs[i] = r["prob"]

        print(f"    推理完成 ({time.time()-t0:.1f}s)")

        # 被试级：概率平均后投票
        sub_pred = {}
        for u in uniq:
            P_u = probs[sids == u].mean(axis=0)
            sub_pred[u] = int(P_u.argmax())

        # 按来源分组评估
        report["groups"][agent_name] = {}
        print(f"    被试级准确率:")
        for src, members in sorted(groups.items()):
            y_true = np.array([sub_y[u] for u in members])
            y_pred = np.array([sub_pred[u] for u in members])
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            base = max(np.bincount(y_true).tolist()) / len(y_true) if len(y_true) > 0 else 0
            margin = acc - base
            report["groups"][agent_name][src] = {
                "n_subjects": len(members),
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
                "baseline": round(float(base), 4),
                "margin_over_baseline": round(float(margin), 4),
            }
            print(f"      {src}: acc={acc:.4f}  baseline={base:.4f}  "
                  f"margin={margin:+.4f}  macro-F1={f1:.4f}  (N={len(members)})")

    # 汇总：跨 Agent 的来源差异
    print(f"\n  汇总（跨 Agent 平均）：")
    for src in sorted(source_counts.keys()):
        margins = []
        for aname in trained:
            entry = report["groups"].get(aname, {}).get(src, {})
            if entry:
                margins.append(entry.get("margin_over_baseline", 0))
        avg_margin = np.mean(margins) if margins else 0
        print(f"    {src}: 平均 margin = {avg_margin:+.4f}")

    # 判断
    if "project" in source_counts and "raw_index" in source_counts:
        proj_margins = [report["groups"][a]["project"]["margin_over_baseline"]
                        for a in trained if "project" in report["groups"].get(a, {})]
        raw_margins = [report["groups"][a]["raw_index"]["margin_over_baseline"]
                       for a in trained if "raw_index" in report["groups"].get(a, {})]
        if proj_margins and raw_margins:
            diff = np.mean(proj_margins) - np.mean(raw_margins)
            print(f"\n  project vs raw_index margin 差: {diff:+.4f}")
            if diff > 0.05:
                print("  ⚠️  差异 > 0.05：raw_index 标签可能不可靠，建议只用 project 子集训练")
            elif diff > 0.02:
                print("  ⚡ 差异 0.02~0.05：raw_index 标签质量偏低，可考虑降权")
            else:
                print("  ✅ 差异 < 0.02：两种标签源基本一致，可安全合并使用")

    out = resolve_path("outputs/label_source_comparison.json")
    ensure_dir(os.path.dirname(out))
    save_json(report, out)
    print(f"\n结果已保存: {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--max-epochs-per-record", type=int, default=20)
    a = ap.parse_args()
    main(split=a.split, max_epochs_per_record=a.max_epochs_per_record)
