# models/eeg_transformer.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGTransformer(nn.Module):
    """
    EEG Transformer - 基于patch的时序Transformer
    将EEG时序切成固定长度的patch，用Transformer编码

    输入: (batch_size, n_channels, n_times)
    输出: (batch_size, n_classes)
    """

    def __init__(self, n_classes=3, n_channels=19, n_times=500,
                 patch_size=50, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4

        self.patch_size = patch_size
        self.n_patches = n_times // patch_size
        patch_dim = n_channels * patch_size

        # Patch嵌入
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, n_channels, n_times)
        B, C, T = x.shape

        # 防御：输入过长时截断，保证 patch 数与位置编码一致
        max_len = self.n_patches * self.patch_size
        if T > max_len:
            x = x[:, :, :max_len]
        elif T < max_len:
            raise ValueError(
                f"输入时间步 {T} 不足，至少需要 {max_len} "
                f"(n_patches={self.n_patches} x patch_size={self.patch_size})"
            )

        # 切成patch: (B, C, n_patches, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size)
        # 展平每个patch: (B, n_patches, C * patch_size)
        x = x.permute(0, 2, 1, 3).flatten(2)

        # Patch嵌入 + 位置编码
        x = self.patch_embed(x)          # (B, n_patches, d_model)
        x = x + self.pos_encoding        # 加位置编码

        # Transformer编码
        x = self.transformer(x)          # (B, n_patches, d_model)

        # 全局平均池化 + 分类
        x = x.mean(dim=1)                # (B, d_model)
        x = self.classifier(x)           # (B, n_classes)
        return x
