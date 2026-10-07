# data/dataset.py
"""
EEG数据集构建
"""

import numpy as np
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    EEG数据集（PyTorch格式）
    用于深度学习模型训练
    """

    def __init__(self, X, y, transform=None):
        """
        参数:
            X: numpy array, (n_samples, n_channels, n_times)
            y: numpy array, (n_samples,) 标签
            transform: 数据增强变换
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x.astype(np.float32), y


def make_epochs(data, sfreq, epoch_length=2, overlap=0.5):
    """
    将连续EEG切成epoch

    参数:
        data: 2D array, (n_channels, n_times)
        sfreq: 采样率
        epoch_length: epoch长度（秒）
        overlap: 重叠比例

    返回:
        epochs: 3D array, (n_epochs, n_channels, n_samples_per_epoch)
    """
    n_channels, n_times = data.shape
    samples_per_epoch = int(epoch_length * sfreq)
    step = max(1, int(samples_per_epoch * (1 - overlap)))

    epochs = []
    start = 0
    while start + samples_per_epoch <= n_times:
        epoch = data[:, start:start + samples_per_epoch]
        epochs.append(epoch)
        start += step

    return np.array(epochs)


def split_by_subject(subject_ids, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    按受试者划分数据集（避免数据泄露）

    参数:
        subject_ids: list/array, 每个样本对应的受试者ID
        labels: list/array, 每个样本的标签
        train_ratio/val_ratio/test_ratio: 划分比例
        seed: 随机种子

    返回:
        train_idx, val_idx, test_idx: 各集合的样本索引
    """
    rng = np.random.default_rng(seed)

    # 先按受试者分组
    unique_subjects = list(set(subject_ids))
    rng.shuffle(unique_subjects)

    n_train = int(len(unique_subjects) * train_ratio)
    n_val = int(len(unique_subjects) * val_ratio)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val:])

    # 按受试者ID分配样本
    train_idx = [i for i, sid in enumerate(subject_ids) if sid in train_subjects]
    val_idx = [i for i, sid in enumerate(subject_ids) if sid in val_subjects]
    test_idx = [i for i, sid in enumerate(subject_ids) if sid in test_subjects]

    return train_idx, val_idx, test_idx
