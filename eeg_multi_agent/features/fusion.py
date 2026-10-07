# features/fusion.py
"""
多域特征融合 + mRMR 特征选择（对应文档 4.1 第3步 / 研究内容三）

流程：
    1. 拼接：时域(114) + 频域(95) + 时频域(152) = 361 维（19通道）
    2. 标准化：训练集拟合均值/标准差（per-feature z-score）
    3. mRMR 选择：最小冗余最大相关
        MID 准则：score(f) = I(f; y) - (1/|S|) Σ_{s∈S} I(f; s)
        MIQ 准则：score(f) = I(f; y) / ((1/|S|) Σ_{s∈S} I(f; s))
       互信息在分箱（等频分箱）后的离散特征上计算，保证特征间/特征-标签口径一致

用法：
    fusion = FeatureFusion(n_select=128)
    X_sel = fusion.fit_transform(X_train, y_train)
    X_test_sel = fusion.transform(X_test)
    fusion.save("checkpoints/fusion/fusion.joblib")
"""

import numpy as np


def _equal_freq_binning(X, n_bins=10):
    """
    逐列等频分箱，返回整数编码矩阵 (n_samples, n_features)，取值 0..n_bins-1
    等频分箱对量纲与异常值不敏感，适合互信息估计
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    bins = np.floor(np.linspace(0, n_samples, n_bins + 1)).astype(int)
    codes = np.zeros(X.shape, dtype=np.int64)

    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        ranks = np.empty(n_samples, dtype=np.int64)
        ranks[order] = np.arange(n_samples)
        # 按分位数切分
        code = np.searchsorted(bins[1:-1], ranks, side="right")
        codes[:, j] = np.clip(code, 0, n_bins - 1)

    return codes


def _joint_counts(a, b, n_bins_a, n_bins_b):
    """联合分布计数矩阵"""
    idx = a.astype(np.int64) * n_bins_b + b.astype(np.int64)
    joint = np.bincount(idx, minlength=n_bins_a * n_bins_b)
    return joint.reshape(n_bins_a, n_bins_b).astype(np.float64)


def _mutual_info(a, b, n_bins_a, n_bins_b):
    """离散互信息 I(a; b)（自然对数单位）"""
    joint = _joint_counts(a, b, n_bins_a, n_bins_b)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    outer = np.outer(pa, pb)

    mask = joint > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / np.maximum(outer[mask], 1e-12))))


def compute_mi_matrix(X, n_bins=10, max_samples=None, verbose=False):
    """
    计算特征间互信息矩阵 (n_features, n_features)

    参数:
        X: 2D array, (n_samples, n_features)
        n_bins: 分箱数
        max_samples: 超过该样本数时随机子采样（加速）

    返回:
        2D array，对称矩阵
    """
    X = np.asarray(X, dtype=np.float64)
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    codes = _equal_freq_binning(X, n_bins=n_bins)
    n_features = codes.shape[1]
    mi = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            v = _mutual_info(codes[:, i], codes[:, j], n_bins, n_bins)
            mi[i, j] = mi[j, i] = v

    if verbose:
        print(f"  互信息矩阵计算完成: {n_features}×{n_features}")
    return mi


def compute_mi_with_y(X, y, n_bins=10):
    """
    计算每个特征与标签的互信息 (n_features,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.int64).reshape(-1)
    n_classes = int(y.max()) + 1

    codes = _equal_freq_binning(X, n_bins=n_bins)
    n_features = codes.shape[1]
    return np.array([
        _mutual_info(codes[:, j], y, n_bins, n_classes) for j in range(n_features)
    ])


def mrmr_select(X, y, n_select=128, n_bins=10, criterion="MID",
                max_samples=None, verbose=False):
    """
    mRMR 贪心特征选择

    参数:
        X: (n_samples, n_features)
        y: (n_samples,)
        n_select: 选择特征数（超过总维数时自动取全部）
        criterion: "MID"（差式，默认）或 "MIQ"（商式）

    返回:
        (selected_indices list, info dict)
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]
    n_select = int(min(max(1, n_select), n_features))

    mi_y = compute_mi_with_y(X, y, n_bins=n_bins)
    mi_matrix = compute_mi_matrix(X, n_bins=n_bins, max_samples=max_samples)

    selected = []
    remaining = list(range(n_features))

    # 第一步：选择与标签互信息最大的特征
    first = int(np.argmax(mi_y))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_select and remaining:
        redundancy = mi_matrix[np.ix_(remaining, selected)].mean(axis=1)

        if criterion.upper() == "MIQ":
            score = mi_y[remaining] / (redundancy + 1e-6)
        else:  # MID
            score = mi_y[remaining] - redundancy

        best_pos = int(np.argmax(score))
        best = remaining[best_pos]
        selected.append(best)
        remaining.remove(best)

        if verbose and len(selected) % 20 == 0:
            print(f"    已选 {len(selected)}/{n_select} 个特征")

    info = {
        "n_input": int(n_features),
        "n_select": int(len(selected)),
        "criterion": criterion,
        "mi_with_y": mi_y.tolist(),
        "mean_mi_selected": float(np.mean(mi_y[selected])) if selected else 0.0,
    }
    return sorted(selected), info


class FeatureFusion:
    """
    多域特征融合器：标准化 + mRMR 选择

    训练阶段 fit（在训练集上拟合标准化参数与特征子集），
    推理/评估阶段 transform（复用同一子集，保证维度一致）。
    """

    def __init__(self, n_select=128, n_bins=10, criterion="MID",
                 normalize=True, max_samples=None, feature_names=None,
                 include_event=False):
        self.n_select = n_select
        self.n_bins = n_bins
        self.criterion = criterion
        self.normalize = normalize
        self.max_samples = max_samples
        self.feature_names = feature_names
        # 训练时是否包含事件级特征Agent（推理侧据此决定特征提取口径；
        # 旧版融合器无该属性，读取处用 getattr(..., False) 兜底）
        self.include_event = include_event

        self.mean_ = None
        self.std_ = None
        self.selected_ = None
        self.mrmr_info_ = None
        self.n_input_ = None

    def fit(self, X, y):
        """在训练集上拟合：标准化参数 + mRMR 特征子集"""
        X = np.asarray(X, dtype=np.float64)
        self.n_input_ = X.shape[1]

        if self.normalize:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ < 1e-8] = 1.0
            Xn = (X - self.mean_) / self.std_
        else:
            Xn = X

        self.selected_, self.mrmr_info_ = mrmr_select(
            Xn, y, n_select=self.n_select, n_bins=self.n_bins,
            criterion=self.criterion, max_samples=self.max_samples,
        )
        return self

    def transform(self, X):
        """
        按已拟合的标准化参数与特征子集变换

        输入维度必须与 fit 时一致（否则说明通道数/特征提取配置发生了变化）
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.n_input_ is not None and X.shape[1] != self.n_input_:
            raise ValueError(
                f"特征维度不匹配: 输入 {X.shape[1]} 维，融合器期望 {self.n_input_} 维。"
                f"请确认输入通道数与训练时一致（config.DATA_CONFIG['n_eeg_channels']）。"
            )

        if self.normalize and self.mean_ is not None:
            X = (X - self.mean_) / self.std_
        if self.selected_ is not None:
            X = X[:, self.selected_]
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_output(self):
        if self.selected_ is not None:
            return len(self.selected_)
        return self.n_input_

    def selected_names(self):
        """返回被选中特征的名称（若构造时提供了 feature_names）"""
        if self.feature_names is None or self.selected_ is None:
            return None
        return [self.feature_names[i] for i in self.selected_]

    def save(self, path):
        import joblib
        from utils.io_utils import ensure_dir
        import os
        ensure_dir(os.path.dirname(path))
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)

    def __repr__(self):
        return (f"<FeatureFusion(n_input={self.n_input_}, n_output={self.n_output}, "
                f"criterion={self.criterion})>")
