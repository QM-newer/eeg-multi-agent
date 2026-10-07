# scripts/check_label_signal.py
"""
标签可分性诊断：epoch 级 vs 被试级，信号到底有多强？

做法：抽取若干 Normal / Abnormal 记录，逐 epoch 提取多域特征，
分别看 (a) 单特征在 epoch 级的 AUC，(b) 按被试平均后在被试级的 AUC。
若被试级明显更高，说明应以"被试级聚合"为主（也更符合临床场景）。

用法：
    python scripts/check_label_signal.py --n-per-class 40 --epochs-per-record 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import DATA_CONFIG
from data.epoch_cache import EpochCache
from training.common import extract_multi_domain_features


def auc(y, score):
    """二分类 AUC（Normal=0 vs Abnormal=2）"""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # 秩和法（含并列修正）
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(len(all_s), dtype=np.float64)
    sorted_s = all_s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main(n_per_class=40, epochs_per_record=8, n_jobs=None):
    cache = EpochCache()
    rng = np.random.default_rng(0)

    picks = {}
    for cls, name in ((0, "Normal"), (2, "Abnormal")):
        idx = [i for i, r in enumerate(cache.rows)
               if int(r["label"]) == cls and r["split"] == "train"]
        sel = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        picks[cls] = sorted(sel.tolist())
        print(f"{name}: 抽取 {len(picks[cls])} 条记录")

    feats, labels, rec_ids = [], [], []
    for cls in picks:
        for i in picks[cls]:
            arr = cache._load(i)
            n = min(epochs_per_record, arr.shape[0])
            sel_ep = np.linspace(0, arr.shape[0] - 1, n).astype(int)
            X = np.stack([np.asarray(arr[k], dtype=np.float32) for k in sel_ep])
            f, dims = extract_multi_domain_features(X, n_jobs=n_jobs)
            feats.append(f)
            labels.append(np.full(n, 1 if cls == 2 else 0))
            rec_ids.append(np.full(n, len(rec_ids)))

    F = np.vstack(feats)
    y = np.concatenate(labels)
    rid = np.concatenate(rec_ids)
    print(f"\nepoch 级样本: {F.shape[0]}  特征: {F.shape[1]} 维  "
          f"(Normal={int((y==0).sum())}, Abnormal={int((y==1).sum())})")

    # --- epoch 级：每个特征的 AUC，取偏离 0.5 最大的几个 ---
    aucs = np.array([auc(y, F[:, j]) for j in range(F.shape[1])])
    dev = np.abs(aucs - 0.5)
    top = np.argsort(dev)[::-1][:10]
    print("\n[epoch 级] AUC 偏离 0.5 最大的 10 个特征:")
    for j in top:
        print(f"    特征#{j:4d}  AUC={aucs[j]:.3f}")
    print(f"    全部特征 |AUC-0.5| 中位 = {np.median(dev):.4f}  最大 = {dev.max():.4f}")

    # --- 被试级：同一被试的 epoch 特征取平均 ---
    uniq = np.unique(rid)
    Fs = np.stack([F[rid == u].mean(axis=0) for u in uniq])
    ys = np.array([y[rid == u][0] for u in uniq])
    aucs_s = np.array([auc(ys, Fs[:, j]) for j in range(Fs.shape[1])])
    dev_s = np.abs(aucs_s - 0.5)
    top_s = np.argsort(dev_s)[::-1][:10]
    print(f"\n[被试级] 每条记录 {epochs_per_record} 个 epoch 取平均后，"
          f"共 {len(uniq)} 个被试 (Normal={int((ys==0).sum())}, Abnormal={int((ys==1).sum())})")
    for j in top_s:
        print(f"    特征#{j:4d}  AUC={aucs_s[j]:.3f}")
    print(f"    全部特征 |AUC-0.5| 中位 = {np.median(dev_s):.4f}  最大 = {dev_s.max():.4f}")

    print("\n结论提示：被试级最大 |AUC-0.5| 明显高于 epoch 级时，"
          "应采用「epoch 特征 → 按被试聚合」的建模方式。")
    return {"epoch_auc_max": float(aucs[top[0]]), "subject_auc_max": float(aucs_s[top_s[0]])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--epochs-per-record", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    main(n_per_class=a.n_per_class, epochs_per_record=a.epochs_per_record, n_jobs=a.n_jobs)
