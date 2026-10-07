# models/eegnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet - 轻量级脑电卷积网络
    参考：EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces
    简化实现版，实际项目建议用 braindecode.models.EEGNetv4

    输入: (batch_size, n_channels, n_times)
    输出: (batch_size, n_classes)
    """

    def __init__(self, n_classes=3, n_channels=19, n_times=500, dropout_rate=0.25,
                 filters=(8, 16, 16), kernel_time=64, pool=(4, 8)):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        f1, f2, f3 = int(filters[0]), int(filters[1]), int(filters[2])
        pool1, pool2 = int(pool[0]), int(pool[1])
        downsample = pool1 * pool2

        # 第一阶段：时间卷积（捕捉时域特征）
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_time),
                      padding=(0, kernel_time // 2), bias=False),
            nn.BatchNorm2d(f1),
        )

        # 第二阶段：深度可分离卷积（空间+深度）
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout_rate),
        )

        # 第三阶段：可分离卷积
        self.separable_conv = nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f3, f3, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout_rate),
        )

        # 分类头：时间维度经过两次池化（//pool1 再 //pool2）
        self.classifier = nn.Linear(f3 * (n_times // downsample), n_classes)

    def forward(self, x):
        # x shape: (batch_size, n_channels, n_times)
        x = x.unsqueeze(1)  # -> (B, 1, C, T)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)

        # 防御：n_times 非 32 的整数倍时维度可能与分类头不符，给出明确报错
        if x.size(1) != self.classifier.in_features:
            raise ValueError(
                f"EEGNet 特征维度不匹配: 实际 {x.size(1)} vs 期望 {self.classifier.in_features}。"
                f"请确认输入 n_times={self.n_times} 能被总降采样因子整除。"
            )

        x = self.classifier(x)
        return x
