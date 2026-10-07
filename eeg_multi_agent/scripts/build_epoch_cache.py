# scripts/build_epoch_cache.py
"""
epoch 缓存构建（阶段 2：把真实 .erd 记录变成可直接训练的 epoch）

对 record_index.csv 中每条有标签记录：
    读取窗口 → 0.5-45Hz 带通 → 降采样 250Hz → 平均参考 → 切 epoch
    → 质量过滤 → 均匀抽样到 max_epochs → 存 .npy + manifest.csv

之所以要缓存：原始记录中位数约 4 小时（663GB 全库），解码约 30× 实时，
训练时反复读取不可行，故一次性预处理成小体积 epoch。

用法：
    python scripts/build_epoch_cache.py --limit 8 --workers 4       # 先跑试点
    python scripts/build_epoch_cache.py --workers 8                 # 全量构建
    python scripts/build_epoch_cache.py --all-records               # 含同一受试者的重复记录
    python scripts/build_epoch_cache.py --overwrite                 # 重算已存在的记录

产物：
    <epoch_cache_dir>/00000.npy ...       每条记录一个 float32 数组 (n_epochs, C, T)
    <epoch_cache_dir>/manifest.csv        记录 ↔ 缓存文件 ↔ 标签 ↔ 划分 的清单
"""

import os
import sys
import csv
import time
import argparse
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_CONFIG                                    # noqa: E402
from data.xltek_reader import XltekRecord                         # noqa: E402
from utils.signal_utils import bandpass_filter, compute_quality_report  # noqa: E402

# 质量过滤阈值
# 实测：本数据集带通后 epoch 峰峰值中位数约 500µV（儿童慢波睡眠脑电幅值偏大），
# 且分布重尾（90 分位约 2200µV），固定阈值不适用，故按每条记录自身中位数做自适应。
PTP_MEDIAN_MULT = 3.0     # 峰峰值阈值 = 本记录中位数 × 该倍数
PTP_MIN = 400.0           # 阈值下限（µV）
PTP_MAX = 5000.0          # 阈值上限（µV）
STD_MIN = 0.5             # 最小通道标准差（µV），过小 = 掉线/平坦
NOTCH_FREQ = 50.0         # 工频陷波（国内 50Hz）


def parse_channels(spec):
    """'0-18' / '0,1,2' → [0,1,...,18]"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_epochs(x, win, hop):
    """(C, T) → (n, C, win)"""
    n = (x.shape[1] - win) // hop + 1
    if n <= 0:
        return np.zeros((0, x.shape[0], win), dtype=np.float64)
    idx = np.arange(n) * hop
    return np.stack([x[:, i:i + win] for i in idx], axis=0)


def process_record(task):
    """单条记录：解码 → 预处理 → epoch 过滤 → 存盘。返回 manifest 行（dict）"""
    idx, rec, params = task
    row = {
        "record_id": rec["record_id"], "subject_id": rec["subject_id"],
        "folder": rec["folder"], "label": rec["label"], "label_name": rec["label_name"],
        "label_source": rec["label_source"], "split": rec["split"],
        "n_epochs": 0, "epochs_file": "", "sfreq": params["target_sfreq"],
        "n_channels": len(params["channels"]),
        "channels": "-".join(str(c) for c in params["channels"]),
        "skip_sec": params["skip_sec"], "duration_sec": params["duration_sec"],
        "snr_db": "", "artifact_ratio": "", "keep_rate": "", "ptp_thr": "",
        "status": "ok", "error": "",
    }
    path = os.path.join(params["out_dir"], "%05d.npy" % idx)
    row["epochs_file"] = os.path.basename(path)

    if os.path.exists(path) and not params["overwrite"]:
        row["status"] = "skipped_exists"
        try:
            row["n_epochs"] = int(np.load(path, mmap_mode="r").shape[0])
        except Exception:                                   # noqa: BLE001
            pass
        return row

    try:
        from scipy.signal import resample_poly

        rec_obj = XltekRecord(eeg_path=rec["eeg_path"])
        data, sfreq, _ = rec_obj.read_window(
            skip_sec=params["skip_sec"], duration_sec=params["duration_sec"],
            channels=params["channels"])
        if data.shape[1] < int(params["epoch_length"] * sfreq) + 10:
            row["status"] = "error"
            row["error"] = "too_short:%d samples" % data.shape[1]
            return row

        x = np.asarray(data, dtype=np.float64)

        # 1) 工频陷波（本数据集 50Hz 干扰可占 60% 能量，
        #    45Hz 四阶 Butterworth 对 50Hz 仅衰减约 5dB，必须先陷波）
        if params["notch_freq"] > 0:
            from scipy.signal import iirnotch, filtfilt
            b, a = iirnotch(params["notch_freq"], 30.0, sfreq)
            x = filtfilt(b, a, x, axis=-1)

        # 2) 带通 0.5-45Hz
        x = bandpass_filter(x, sfreq, params["low_freq"], params["high_freq"])

        # 2) 降采样到目标采样率
        down = int(round(sfreq / params["target_sfreq"]))
        if down > 1:
            x = resample_poly(x, up=1, down=down, axis=-1)

        # 3) 平均参考
        x = x - x.mean(axis=0, keepdims=True)

        # 4) 切 epoch
        win = int(round(params["epoch_length"] * params["target_sfreq"]))
        hop = max(1, int(round(win * (1.0 - params["overlap"]))))
        ep = make_epochs(x, win, hop)
        if ep.shape[0] == 0:
            row["status"] = "error"
            row["error"] = "no_epoch"
            return row

        # 5) 质量过滤（自适应阈值：本记录峰峰值中位数 × 3）
        ptp = np.ptp(ep, axis=2).max(axis=1)          # 每 epoch 最大通道峰峰值
        std = ep.std(axis=2)
        thr = float(np.clip(PTP_MEDIAN_MULT * np.median(ptp), PTP_MIN, PTP_MAX))
        keep = (ptp <= thr) & (std.min(axis=1) >= STD_MIN) & np.isfinite(ep).all(axis=(1, 2))
        n_keep = int(keep.sum())
        row["keep_rate"] = round(float(n_keep) / ep.shape[0], 3)
        row["ptp_thr"] = round(thr, 1)
        if n_keep == 0:
            row["status"] = "error"
            row["error"] = "all_rejected"
            return row

        ep = ep[keep]

        # 6) 均匀抽样到 max_epochs（保留整条记录的时间覆盖度）
        if params["max_epochs"] > 0 and ep.shape[0] > params["max_epochs"]:
            sel = np.linspace(0, ep.shape[0] - 1, params["max_epochs"]).astype(int)
            ep = ep[sel]

        # 7) 质量统计（抽样若干 epoch 估 SNR / 伪迹比例）
        probe = ep[::max(1, ep.shape[0] // 5)][:5]
        q = [compute_quality_report(e, sfreq=params["target_sfreq"]) for e in probe]
        row["snr_db"] = round(float(np.mean([v["snr_db"] for v in q])), 2)
        row["artifact_ratio"] = round(float(np.mean([v["artifact_ratio"] for v in q])), 3)

        np.save(path, ep.astype(np.float32))
        row["n_epochs"] = int(ep.shape[0])
        return row

    except Exception as e:                                  # noqa: BLE001
        row["status"] = "error"
        row["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        row["_tb"] = traceback.format_exc()[-800:]
        return row


def load_records(first_only):
    path = DATA_CONFIG["record_index"]
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    sel = [r for r in rows if r["label"] != "" and r["erd_paths"]]
    if first_only:
        sel = [r for r in sel if r["first_record"] == "1"]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip", type=float, default=DATA_CONFIG["read_window"]["skip_sec"])
    ap.add_argument("--duration", type=float,
                    default=DATA_CONFIG["read_window"]["duration_sec"])
    ap.add_argument("--epoch-length", type=float, default=DATA_CONFIG["epoch_length"])
    ap.add_argument("--overlap", type=float, default=DATA_CONFIG["epoch_overlap"])
    ap.add_argument("--max-epochs", type=int, default=DATA_CONFIG["max_epochs_per_record"])
    ap.add_argument("--channels", default=",".join(
        str(c) for c in DATA_CONFIG["eeg_channel_indices"]))
    ap.add_argument("--low-freq", type=float, default=0.5)
    ap.add_argument("--high-freq", type=float, default=45.0)
    ap.add_argument("--notch-freq", type=float, default=NOTCH_FREQ, help="工频陷波，0=关闭")
    ap.add_argument("--out", default=DATA_CONFIG["epoch_cache_dir"])
    ap.add_argument("--all-records", action="store_true", help="含同一受试者的重复记录")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    channels = parse_channels(args.channels)

    params = {
        "out_dir": out_dir, "channels": channels, "skip_sec": args.skip,
        "duration_sec": args.duration, "epoch_length": args.epoch_length,
        "overlap": args.overlap, "max_epochs": args.max_epochs,
        "target_sfreq": DATA_CONFIG["target_sfreq"],
        "low_freq": args.low_freq, "high_freq": args.high_freq,
        "notch_freq": args.notch_freq, "overwrite": args.overwrite,
    }

    recs = load_records(first_only=not args.all_records)
    if args.limit > 0:
        recs = recs[:args.limit]

    print("=" * 70)
    print("epoch 缓存构建")
    print("=" * 70)
    print(f"记录数      : {len(recs)}")
    print(f"读取窗口    : 跳过 {args.skip}s，读取 {args.duration}s")
    print(f"通道        : {channels}（共 {len(channels)} 个）")
    print(f"预处理      : {args.low_freq}-{args.high_freq}Hz → {params['target_sfreq']}Hz "
          f"→ 平均参考")
    print(f"epoch       : {args.epoch_length}s，重叠 {args.overlap}，"
          f"每条最多 {args.max_epochs} 个")
    print(f"输出目录    : {out_dir}")
    print(f"并行进程    : {args.workers}")
    print("=" * 70)

    tasks = [(i, r, params) for i, r in enumerate(recs)]
    t0 = time.time()
    results = []
    if args.workers <= 1:
        for t in tasks:
            results.append(process_record(t))
    else:
        with Pool(processes=args.workers) as pool:
            for k, row in enumerate(pool.imap_unordered(process_record, tasks), 1):
                results.append(row)
                if k % 10 == 0 or k == len(tasks):
                    done = time.time() - t0
                    eta = done / k * (len(tasks) - k)
                    print(f"  进度 {k}/{len(tasks)}  用时 {done/60:.1f}min  "
                          f"预计剩余 {eta/60:.1f}min", flush=True)

    # 写 manifest
    man_path = os.path.join(out_dir, "manifest.csv")
    fields = ["record_id", "subject_id", "folder", "label", "label_name",
              "label_source", "split", "n_epochs", "epochs_file", "sfreq",
              "n_channels", "channels", "skip_sec", "duration_sec", "snr_db",
              "artifact_ratio", "keep_rate", "ptp_thr", "status", "error"]
    with open(man_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # 汇总
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    n_ep = sum(r["n_epochs"] for r in ok)
    size = sum(os.path.getsize(os.path.join(out_dir, r["epochs_file"]))
               for r in ok if r["epochs_file"] and
               os.path.exists(os.path.join(out_dir, r["epochs_file"])))

    print("\n" + "=" * 70)
    print(f"成功 {len(ok)} 条，失败 {len(err)} 条，总 epoch {n_ep:,}")
    print(f"缓存体积    : {size/1e9:.2f} GB")
    print(f"总耗时      : {(time.time()-t0)/60:.1f} min")
    if ok:
        by_cls = {}
        for r in ok:
            by_cls.setdefault(r["label_name"], [0, 0])
            by_cls[r["label_name"]][0] += 1
            by_cls[r["label_name"]][1] += r["n_epochs"]
        print("类别分布（记录数 / epoch 数）:")
        for k, (nrec, nep) in sorted(by_cls.items(), key=lambda x: -x[1][0]):
            print(f"  {k:<12s} {nrec:>5d} 条  {nep:>8,} epoch")
        keep = [r["keep_rate"] for r in ok if r["keep_rate"] != ""]
        snr = [r["snr_db"] for r in ok if r["snr_db"] != ""]
        if keep:
            print(f"epoch 保留率: 中位 {np.median(keep):.2f}  "
                  f"（最低 {min(keep):.2f}）")
        if snr:
            print(f"SNR(dB)    : 中位 {np.median(snr):.1f}")
    if err:
        print("\n失败前 10 条:")
        for r in err[:10]:
            print(f"  {r['record_id'][:40]}  {r['error']}")
        tb = [r for r in err if r.get("_tb")]
        if tb:
            print("\n首个失败的堆栈:")
            print(tb[0]["_tb"])
    print(f"\nmanifest -> {man_path}")


if __name__ == "__main__":
    main()
