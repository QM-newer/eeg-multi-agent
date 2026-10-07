# models/tcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """时序卷积块：因果卷积 + 残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # 1x1卷积用于匹配维度（残差连接）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        # 裁剪掉右边的padding（因果卷积）
        out = out[:, :, :x.size(2)]

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCN(nn.Module):
    """
    时序卷积网络 TCN
    特点：因果卷积 + 空洞卷积 + 残差连接，捕捉长程时序依赖

    输入: (batch_size, n_channels, n_times)
    输出: (batch_size, n_classes)
    """

    def __init__(self, n_classes=3, n_channels=19, num_layers=6,
                 hidden_channels=64, kernel_size=5, dropout=0.2, dilation_base=2):
        super().__init__()

        # hidden_channels 支持 int（各层相同）或序列（逐层指定）
        if isinstance(hidden_channels, (list, tuple)):
            hidden_list = [int(h) for h in hidden_channels]
        else:
            hidden_list = [int(hidden_channels)] * num_layers

        layers = []
        in_ch = n_channels
        for i in range(num_layers):
            dilation = dilation_base ** i  # 膨胀系数指数增长
            out_ch = hidden_list[min(i, len(hidden_list) - 1)]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        # x shape: (batch_size, n_channels, n_times)
        x = self.tcn_layers(x)       # -> (B, hidden, T)
        x = x.mean(dim=-1)           # 全局平均池化 -> (B, hidden)
        x = self.classifier(x)       # -> (B, n_classes)
        return x
