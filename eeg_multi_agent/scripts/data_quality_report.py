# scripts/data_quality_report.py
"""
阶段 2 数据质量报告：epoch 缓存的规模、类别分布、信号质量统计

输出：
    - 控制台表格
    - outputs/data_quality.json（供论文"数据"章节与后续报告引用）

用法：
    python scripts/data_quality_report.py [--sample-records 200]
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import DATA_CONFIG, CLASS_NAMES, resolve_path
from data.epoch_cache import EpochCache
from utils.io_utils import ensure_dir, save_json


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "mean": round(float(values.mean()), 3),
        "median": round(float(np.median(values)), 3),
        "p10": round(float(np.percentile(values, 10)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def main(sample_records=200):
    cache = EpochCache()
    print("=" * 70)
    print("epoch 缓存数据质量报告")
    print("=" * 70)
    print(cache.summary())

    rows = cache.rows
    print(f"\n清单记录数: {len(rows)}")

    # ---- 规模与类别分布 ----
    report = {"cache_dir": cache.cache_dir, "n_records": len(rows),
              "n_channels": cache.n_channels, "n_times": cache.n_times,
              "sfreq": cache.sfreq}

    for sp in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == sp]
        cls_ep, cls_rec = {}, {}
        for r in sub:
            n = int(r["n_epochs"])
            cls_ep[r["label_name"]] = cls_ep.get(r["label_name"], 0) + n
            cls_rec[r["label_name"]] = cls_rec.get(r["label_name"], 0) + 1
        report[sp] = {
            "n_records": len(sub),
            "n_subjects": len({r["subject_id"] for r in sub}),
            "n_epochs": sum(cls_ep.values()),
            "epochs_by_class": cls_ep,
            "records_by_class": cls_rec,
        }
        print(f"\n[{sp}] 记录 {len(sub)}  受试者 {report[sp]['n_subjects']}  "
              f"epoch {sum(cls_ep.values())}")
        for name in CLASS_NAMES:
            if name in cls_rec:
                print(f"    {name:12s} 记录 {cls_rec[name]:5d}   epoch {cls_ep[name]:7d}")

    # ---- 预处理质量指标 ----
    for key in ("snr_db", "artifact_ratio", "keep_rate", "n_epochs"):
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        report[key] = _stats(vals)
        print(f"\n{key}: " + "  ".join(f"{k}={v}" for k, v in report[key].items()))

    # ---- 抽样检查幅值/频谱 ----
    rng = np.random.default_rng(42)
    n_sample = min(sample_records, len(rows))
    sel = rng.choice(len(rows), n_sample, replace=False)
    stds, ptps, flat = [], [], []
    band_names = ["delta0.5-4", "theta4-8", "alpha8-13", "beta13-30", "gamma30-45"]
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    band_acc = np.zeros(len(bands))
    n_ep = 0

    for i in sel:
        r = rows[i]
        x = np.load(os.path.join(cache.cache_dir, r["epochs_file"]), mmap_mode="r")
        sub = np.asarray(x[:5], dtype=np.float32)          # 每条记录看 5 个 epoch
        stds.append(sub.std(axis=2).mean())
        ptps.append(np.ptp(sub, axis=2).max())
        flat.append(int((sub.std(axis=2) < 1.0).any()))     # 是否有掉线通道
        f = np.abs(np.fft.rfft(sub - sub.mean(axis=2, keepdims=True), axis=2)).mean(axis=(0, 1))
        fr = np.fft.rfftfreq(sub.shape[2], 1.0 / cache.sfreq)
        denom = f[(fr >= 0.5) & (fr <= 45)].sum()
        if denom > 0:
            band_acc += np.array([f[(fr >= lo) & (fr < hi)].sum() for lo, hi in bands]) / denom
            n_ep += 1

    report["amplitude"] = {
        "epoch_std_uV": _stats(stds),
        "epoch_ptp_uV": _stats(ptps),
        "records_with_flat_channel": int(sum(flat)),
        "sampled_records": n_sample,
    }
    print(f"\n幅值（抽样 {n_sample} 条记录）:")
    print(f"  epoch std 中位 {report['amplitude']['epoch_std_uV']['median']} µV  "
          f"[p10 {report['amplitude']['epoch_std_uV']['p10']}, "
          f"p90 {report['amplitude']['epoch_std_uV']['p90']}]")
    print(f"  epoch 峰峰值中位 {report['amplitude']['epoch_ptp_uV']['median']} µV")
    print(f"  含掉线通道的记录: {sum(flat)}/{n_sample}")

    if n_ep:
        band_acc = band_acc / n_ep * 100
        report["band_power_percent"] = {n: round(float(v), 2)
                                        for n, v in zip(band_names, band_acc)}
        print("\n频带能量占比（抽样平均，%）:")
        for n, v in zip(band_names, band_acc):
            print(f"  {n:14s} {v:6.2f}")

    out = resolve_path("outputs/data_quality.json")
    ensure_dir(os.path.dirname(out))
    save_json(report, out)
    print(f"\n报告已保存: {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-records", type=int, default=200)
    a = ap.parse_args()
    main(sample_records=a.sample_records)
