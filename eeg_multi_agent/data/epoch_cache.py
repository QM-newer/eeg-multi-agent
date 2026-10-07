# data/epoch_cache.py
"""
epoch 缓存加载器（阶段 2 产物 → 训练输入）

build_epoch_cache.py 产出的目录结构：
    <cache_dir>/00000.npy ...     每条记录一个 float32 数组 (n_epochs, C, T)
    <cache_dir>/manifest.csv      记录 ↔ 文件 ↔ 标签 ↔ 划分

本模块负责：
    - 按 split（已在 record_index 里按受试者划分好，不会泄露）取样本
    - 支持两种取数方式：一次性载入内存（LightGBM 特征工程）或懒加载（深度模型）
    - 逐通道 z-score 归一化（统计量只用训练集估计）

用法：
    from data.epoch_cache import EpochCache
    cache = EpochCache()
    print(cache.summary())
    X, y, sids = cache.load_split("train", max_epochs_per_record=40)
"""

import os
import csv
import numpy as np

from config import DATA_CONFIG


class EpochCache(object):
    """epoch 缓存：manifest + 若干 .npy"""

    def __init__(self, cache_dir=None, seed=None):
        self.cache_dir = cache_dir or DATA_CONFIG["epoch_cache_dir"]
        self.seed = seed or 42
        self.manifest_path = os.path.join(self.cache_dir, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"未找到缓存清单 {self.manifest_path}，请先运行：\n"
                f"  python scripts/build_epoch_cache.py --workers 8")

        with open(self.manifest_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows
                     if r.get("status") in ("ok", "skipped_exists")
                     and int(r.get("n_epochs") or 0) > 0]
        if not self.rows:
            raise RuntimeError(f"缓存清单里没有可用记录: {self.manifest_path}")

        self.n_channels = int(self.rows[0]["n_channels"])
        self.sfreq = float(self.rows[0]["sfreq"])
        self._n_times = None
        self._cache = {}          # record_idx -> mmap 数组

    # ---------- 基本信息 ----------
    @property
    def n_times(self):
        if self._n_times is None:
            arr = self._load(0)
            self._n_times = int(arr.shape[2])
        return self._n_times

    @property
    def n_records(self):
        return len(self.rows)

    @property
    def total_epochs(self):
        return sum(int(r["n_epochs"]) for r in self.rows)

    def summary(self):
        lines = [
            "epoch 缓存: %s" % self.cache_dir,
            "  记录数 %d  总 epoch %d  形状 (%d, %d)" %
            (self.n_records, self.total_epochs, self.n_channels, self.n_times),
        ]
        for sp in ("train", "val", "test"):
            sub = [r for r in self.rows if r["split"] == sp]
            if sub:
                ep = sum(int(r["n_epochs"]) for r in sub)
                cls = {}
                for r in sub:
                    cls[r["label_name"]] = cls.get(r["label_name"], 0) + int(r["n_epochs"])
                lines.append("  %-5s %4d 条记录 %7d epoch   %s" %
                             (sp, len(sub), ep,
                              "  ".join(f"{k}={v}" for k, v in sorted(cls.items()))))
        return "\n".join(lines)

    # ---------- 底层读取 ----------
    def _path(self, rec_idx):
        return os.path.join(self.cache_dir, self.rows[rec_idx]["epochs_file"])

    def _load(self, rec_idx):
        arr = self._cache.get(rec_idx)
        if arr is None:
            arr = np.load(self._path(rec_idx), mmap_mode="r")
            self._cache[rec_idx] = arr
        return arr

    def get_epoch(self, rec_idx, ep_idx):
        """取单个 epoch，返回 (C, T) float32"""
        return np.asarray(self._load(rec_idx)[ep_idx], dtype=np.float32)

    # ---------- 索引构建 ----------
    def build_index(self, split=None, max_epochs_per_record=None, seed=None):
        """
        构建样本级索引

        返回:
            dict: {rec_ids, ep_ids, y, subject_ids, label_names}
            rec_ids/ep_ids: 第 i 个样本来自 records[rec_ids[i]] 的第 ep_ids[i] 个 epoch
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        rec_ids, ep_ids, ys, sids = [], [], [], []
        for i, r in enumerate(self.rows):
            if split is not None and r["split"] != split:
                continue
            n = int(r["n_epochs"])
            if max_epochs_per_record and n > max_epochs_per_record:
                sel = np.sort(rng.choice(n, max_epochs_per_record, replace=False))
            else:
                sel = np.arange(n)
            rec_ids.append(np.full(len(sel), i, dtype=np.int64))
            ep_ids.append(sel.astype(np.int64))
            ys.append(np.full(len(sel), int(r["label"]), dtype=np.int64))
            sids.extend([r["subject_id"]] * len(sel))
        return {
            "rec_ids": np.concatenate(rec_ids) if rec_ids else np.zeros(0, np.int64),
            "ep_ids": np.concatenate(ep_ids) if ep_ids else np.zeros(0, np.int64),
            "y": np.concatenate(ys) if ys else np.zeros(0, np.int64),
            "subject_ids": np.array(sids),
        }

    # ---------- 全量载入 ----------
    def load_split(self, split, max_epochs_per_record=None, seed=None,
                   normalize=None, norm_stats=None, verbose=False):
        """
        把某个 split 的全部 epoch 载入内存

        参数:
            split: "train" / "val" / "test"
            max_epochs_per_record: 每条记录最多取多少 epoch（控制内存/训练量）
            normalize: "per_channel" 时按通道 z-score
            norm_stats: (mean, std)，None 时用本 split 自身统计量

        返回:
            (X, y, subject_ids)，X: (n, C, T) float32
        """
        idx = self.build_index(split, max_epochs_per_record=max_epochs_per_record, seed=seed)
        n = len(idx["y"])
        X = np.empty((n, self.n_channels, self.n_times), dtype=np.float32)
        for k in range(n):
            X[k] = self.get_epoch(idx["rec_ids"][k], idx["ep_ids"][k])
            if verbose and (k + 1) % 20000 == 0:
                print(f"    载入 {k+1}/{n}")

        if normalize == "per_channel":
            if norm_stats is None:
                mean = X.mean(axis=(0, 2), keepdims=True)
                std = X.std(axis=(0, 2), keepdims=True)
            else:
                mean, std = norm_stats
            X = (X - mean) / np.maximum(std, 1e-8)
        else:
            mean = std = None

        return X, idx["y"], idx["subject_ids"], (mean, std)

    def fit_normalizer(self, split="train", max_epochs_per_record=None, sample=20000):
        """只用训练集估计逐通道 mean/std（避免划分泄露）"""
        idx = self.build_index(split, max_epochs_per_record=max_epochs_per_record)
        n = len(idx["y"])
        sel = np.linspace(0, max(n - 1, 0), min(sample, n)).astype(int) if n else np.zeros(0, int)
        acc = []
        for k in sel:
            acc.append(self.get_epoch(idx["rec_ids"][k], idx["ep_ids"][k]))
        if not acc:
            return np.zeros((1, self.n_channels, 1)), np.ones((1, self.n_channels, 1))
        stack = np.stack(acc)
        return (stack.mean(axis=(0, 2), keepdims=True).astype(np.float32),
                stack.std(axis=(0, 2), keepdims=True).astype(np.float32))


class CachedEpochDataset(object):
    """
    懒加载版数据集（配合 torch DataLoader 用；避免一次性占用数 GB 内存）

    __getitem__ 返回 (C, T) float32 归一化后的样本
    """

    def __init__(self, cache, index, norm_stats=None):
        self.cache = cache
        self.index = index
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.index["y"])

    def __getitem__(self, i):
        x = self.cache.get_epoch(self.index["rec_ids"][i], self.index["ep_ids"][i])
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            x = (x - mean) / np.maximum(std, 1e-8)
        return x, int(self.index["y"][i])


NORM_PATH = "checkpoints/norm_stats.json"


def save_norm_stats(mean, std, path=None):
    """
    保存逐通道归一化统计量（训练集估计）

    训练与推理必须看到同一变换：诊断Agent（agents/diagnosis_agents/*）在推理时
    会读取这里的 mean/std 做同样的标准化，避免训练/推理分布不一致导致准确率下降。
    """
    from config import resolve_path
    from utils.io_utils import ensure_dir, save_json
    import os

    path = path or resolve_path(NORM_PATH)
    ensure_dir(os.path.dirname(path))
    save_json({"mean": np.asarray(mean).reshape(-1).tolist(),
               "std": np.asarray(std).reshape(-1).tolist()}, path)
    return path


def load_norm_stats(path=None):
    """
    读取归一化统计量，返回 (mean, std)，形状均为 (1, C, 1)；不存在则返回 (None, None)
    """
    import os
    from config import resolve_path
    from utils.io_utils import load_json

    path = path or resolve_path(NORM_PATH)
    if not os.path.exists(path):
        return None, None
    try:
        d = load_json(path) or {}
        mean = np.asarray(d["mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.asarray(d["std"], dtype=np.float32).reshape(1, -1, 1)
        return mean, np.maximum(std, 1e-8)
    except Exception:
        return None, None


def compute_class_weights(y, n_classes=3):
    """类别加权（Borderline 只有约 10%，必须处理不平衡）

    返回: 长度为 n_classes 的 float32 数组，按 1/频数 归一化
    """
    y = np.asarray(y)
    counts = np.array([max(int((y == c).sum()), 1) for c in range(n_classes)], dtype=np.float64)
    w = counts.sum() / (len(counts) * counts)
    return (w / w.mean()).astype(np.float32)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    c = EpochCache()
    print(c.summary())
    idx = c.build_index("train")
    print("train 样本数:", len(idx["y"]), " 类别分布:", np.bincount(idx["y"]))
    print("类别权重:", compute_class_weights(idx["y"]))
