# scripts/extract_full_features.py
"""
一次性提取 614 维多域特征（时域228 + 频域152 + 时频152 + 事件82）并存盘。

供事件特征重训与消融实验（A1~A3、event on/off）共用，避免每个变体重复
40 分钟的提取。列块布局（与 _make_feature_agents 拼接顺序一致）：
    [0:228)   time_domain
    [228:380) freq_domain
    [380:532) timefreq
    [532:614) event

口径（与既有基线一致）：
    train/val: max 100 epochs/record（lightgbm_binary 训练口径）
    test:      max 20  epochs/record（evaluate_binary 评估口径）
    归一化:    train@100 fit 的逐通道统计量，并回写 checkpoints/norm_stats.json

输出（E:/eeg_asd_cache/features/）:
    {split}_feat.npy        (n, 614) float32
    {split}_meta.npz        y(3分类), subject_ids
    manifest.json           维度布局与口径说明

用法:
    python scripts/extract_full_features.py
    python scripts/extract_full_features.py --n-jobs 8
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import DATA_CONFIG
from data.epoch_cache import EpochCache, save_norm_stats
from training.common import extract_multi_domain_features
from utils.io_utils import save_json

OUT_DIR = os.path.join(os.path.dirname(DATA_CONFIG["epoch_cache_dir"]), "features")

SPLIT_PROTOCOL = [("train", 100), ("val", 100), ("test", 20)]


def main(n_jobs=None):
    print("=" * 74)
    print("全量多域特征提取（含事件级 614 维）")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 74)

    os.makedirs(OUT_DIR, exist_ok=True)

    cache = EpochCache()
    print(cache.summary())

    # 归一化统计量：train@100（与 lightgbm_binary 训练时一致），并回写 norm_stats.json
    print("\n[1/2] 拟合归一化统计量（train @100 epochs/record）...")
    mean, std = cache.fit_normalizer("train", max_epochs_per_record=100)
    save_norm_stats(mean, std)
    print("  已回写 checkpoints/norm_stats.json")

    print("\n[2/2] 逐 split 提取特征...")
    manifest = {
        "dims": {"time_domain": 228, "freq_domain": 152, "timefreq": 152, "event": 82},
        "layout": {"time_domain": [0, 228], "freq_domain": [228, 380],
                   "timefreq": [380, 532], "event": [532, 614]},
        "normalize": "per_channel (train@100 stats, 同 norm_stats.json)",
        "splits": {},
    }

    for split, max_ep in SPLIT_PROTOCOL:
        t0 = time.time()
        X, y, sids, _ = cache.load_split(
            split, max_epochs_per_record=max_ep,
            normalize="per_channel", norm_stats=(mean, std), verbose=False)
        print(f"\n  [{split}] {X.shape[0]} epochs "
              f"({len(set(sids))} 被试, max {max_ep}/record)，提取中...")

        feats, dims = extract_multi_domain_features(
            X, n_jobs=n_jobs, include_event=True, verbose=True)

        np.save(os.path.join(OUT_DIR, f"{split}_feat.npy"), feats.astype(np.float32))
        np.savez(os.path.join(OUT_DIR, f"{split}_meta.npz"), y=y, subject_ids=sids)

        manifest["splits"][split] = {
            "n_epochs": int(X.shape[0]),
            "n_subjects": int(len(set(sids))),
            "max_epochs_per_record": max_ep,
            "feature_shape": list(feats.shape),
            "dims": dims,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  [{split}] 完成: {feats.shape}  耗时 {time.time()-t0:.0f}s")
        del X, feats

    save_json(manifest, os.path.join(OUT_DIR, "manifest.json"))
    print("\n全部完成。manifest 已保存。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    main(n_jobs=a.n_jobs)
