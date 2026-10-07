# scripts/check_window_signal.py
"""
取样窗口诊断：判别信息在记录的哪个时间段最强？

临床判读看的是整段记录（尤其入睡后的慢波/异常放电），而 epoch 缓存目前只取开头 10 分钟
（跳过 60s）。若信号在记录中后段更强，就需要改成"多窗口分散取样"或加大读取时长。

做法：对同一批 Normal / Abnormal 记录，分别在 skip = 60s / 15min / 30min / 60min 处
各读 2 分钟，提取多域特征并按被试平均，比较各窗口下特征的 AUC。

用法：
    python scripts/check_window_signal.py --n-per-class 40 --workers 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.signal import iirnotch, filtfilt, resample_poly

from config import DATA_CONFIG
from data.epoch_cache import EpochCache
from data.xltek_reader import XltekRecord
from utils.signal_utils import bandpass_filter
from scripts.check_label_signal import auc

WINDOWS = [60, 900, 1800, 3600]     # 从记录开头跳过的秒数
DURATION = 120                      # 每个窗口读取 120 秒
N_EPOCHS = 10                       # 每个窗口取多少个 2 秒 epoch


def _preprocess(x, sfreq):
    b, a = iirnotch(50.0, 30.0, sfreq)
    x = filtfilt(b, a, x, axis=-1)
    x = bandpass_filter(x, sfreq, 0.5, 45.0)
    down = int(round(sfreq / DATA_CONFIG["target_sfreq"]))
    if down > 1:
        x = resample_poly(x, up=1, down=down, axis=-1)
    x = x - x.mean(axis=0, keepdims=True)
    return x


def _one_record(args):
    """worker：一条记录 × 多个窗口 → 各窗口的被试级特征向量"""
    eeg_path, label = args
    from training.common import extract_multi_domain_features

    out = {"label": label, "windows": {}}
    try:
        rec = XltekRecord(eeg_path=eeg_path)
        for skip in WINDOWS:
            data, sfreq, _ = rec.read_window(
                skip_sec=skip, duration_sec=DURATION,
                channels=DATA_CONFIG["eeg_channel_indices"])
            if data.shape[1] < 600:
                continue
            x = _preprocess(np.asarray(data, dtype=np.float64), sfreq)
            sf = DATA_CONFIG["target_sfreq"]
            step = int(DATA_CONFIG["epoch_length"] * sf * (1 - DATA_CONFIG["epoch_overlap"]))
            n_ep = (x.shape[1] - 500) // step
            if n_ep < N_EPOCHS:
                continue
            sel = np.linspace(0, n_ep - 1, N_EPOCHS).astype(int)
            ep = np.stack([x[:, k * step:k * step + 500] for k in sel])
            f, _ = extract_multi_domain_features(ep.astype(np.float32), n_jobs=1)
            out["windows"][skip] = f.mean(axis=0)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main(n_per_class=40, workers=10):
    cache = EpochCache()
    rng = np.random.default_rng(0)
    tasks = []
    for cls in (0, 2):
        idx = [i for i, r in enumerate(cache.rows)
               if int(r["label"]) == cls and r["split"] == "train"]
        for i in rng.choice(idx, min(n_per_class, len(idx)), replace=False):
            r = cache.rows[i]
            folder = r["folder"]
            if not os.path.isabs(folder):
                folder = os.path.join(DATA_CONFIG["data_root"], folder)
            eeg_path = os.path.join(folder, r["record_id"] + ".eeg")
            tasks.append((os.path.normpath(eeg_path), 1 if cls == 2 else 0))
    print(f"共 {len(tasks)} 条记录（Normal {n_per_class} / Abnormal {n_per_class}）")
    print(f"窗口: {[f'{w//60}min' if w >= 60 else f'{w}s' for w in WINDOWS]}")

    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_one_record, tasks)

    ok = [r for r in results if r.get("windows")]
    err = [r for r in results if r.get("error")]
    print(f"成功 {len(ok)} 条，失败 {len(err)} 条")
    for r in err[:3]:
        print("   失败示例:", r["error"])

    print("\n各窗口下「被试级平均特征」的可分性（Normal vs Abnormal）:")
    print(f"{'窗口':>8s}  {'被试数':>6s}  {'最佳AUC':>8s}  {'|AUC-0.5|中位':>12s}  {'|AUC-0.5|最大':>12s}")
    for skip in WINDOWS:
        sub = [r for r in ok if skip in r["windows"]]
        if len(sub) < 10:
            print(f"{skip:>8d}  样本不足（{len(sub)}）")
            continue
        F = np.stack([r["windows"][skip] for r in sub])
        y = np.array([r["label"] for r in sub])
        aucs = np.array([auc(y, F[:, j]) for j in range(F.shape[1])])
        dev = np.abs(aucs - 0.5)
        best_j = int(np.argmax(dev))
        print(f"{skip:>8d}  {len(sub):>6d}  {aucs[best_j]:>8.3f}  "
              f"{np.median(dev):>12.4f}  {dev.max():>12.4f}")

    print("\n结论提示：若中后段窗口的最佳 AUC 明显高于开头 60s 处，"
          "说明判别信息集中在记录中后段（多为入睡后），应改取样策略。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(n_per_class=a.n_per_class, workers=a.workers)
