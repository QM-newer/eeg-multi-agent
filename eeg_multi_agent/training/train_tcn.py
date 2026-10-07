# training/train_tcn.py
"""
TCN 诊断Agent 训练脚本（吃原始时序，因果空洞卷积 + 残差）

用法：
    python training/train_tcn.py [--epochs 30] [--n-subjects 90]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_CONFIG
from training.common import train_deep_agent


def main(epochs=30, n_subjects=90, epochs_per_subject=5, batch_size=None,
         device=None, seed=None, max_epochs_per_record=None):
    return train_deep_agent(
        "tcn", n_subjects=n_subjects, epochs_per_subject=epochs_per_subject,
        epochs=epochs, batch_size=batch_size, device=device, seed=seed,
        max_epochs_per_record=max_epochs_per_record,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-subjects", type=int, default=90)
    parser.add_argument("--epochs-per-subject", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=TRAIN_CONFIG["device"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs-per-record", type=int, default=None)
    args = parser.parse_args()
    main(epochs=args.epochs, n_subjects=args.n_subjects,
         epochs_per_subject=args.epochs_per_subject, batch_size=args.batch_size,
         device=args.device, seed=args.seed,
         max_epochs_per_record=args.max_epochs_per_record)
