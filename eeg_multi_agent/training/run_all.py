# training/run_all.py
"""
一键训练全部四个诊断Agent，并按验证集准确率回写动态投票权重

对应文档 4.2(1)：w_i 由各Agent在验证集上的准确率 Acc_i 归一化得到

用法：
    python training/run_all.py --epochs 30
    python training/run_all.py --only lightgbm eegnet     # 只训练部分
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AGENT_WEIGHTS, resolve_path
from training.common import (
    accuracies_to_weights, load_val_accuracies, save_val_accuracies,
    train_deep_agent,
)
from utils.io_utils import save_json

WEIGHTS_PATH = resolve_path("checkpoints/agent_weights.json")

AGENT_ORDER = ["lightgbm", "eegnet", "tcn", "transformer"]


def main(epochs=30, n_subjects=90, epochs_per_subject=5, n_select=128,
         device=None, only=None, weights_only=False, seed=None,
         max_epochs_per_record=None):
    print("=" * 60)
    print("EEG-MAS 全部诊断Agent训练")
    print("=" * 60)

    targets = [] if weights_only else (only or AGENT_ORDER)
    accuracies = load_val_accuracies()
    if weights_only:
        print("  仅回写权重模式：读取已记录的验证集准确率，跳过训练")

    for name in targets:
        if name == "lightgbm":
            from training.train_lightgbm import main as train_lgbm
            acc = train_lgbm(n_subjects=n_subjects,
                             epochs_per_subject=epochs_per_subject,
                             n_select=n_select, seed=seed,
                             max_epochs_per_record=max_epochs_per_record)
        else:
            acc = train_deep_agent(name, n_subjects=n_subjects,
                                   epochs_per_subject=epochs_per_subject,
                                   epochs=epochs, device=device, seed=seed,
                                   max_epochs_per_record=max_epochs_per_record)
        accuracies[name] = acc
        print(f"  -> {name} 验证准确率 = {acc:.4f}\n")

    save_val_accuracies(accuracies)

    print("=" * 60)
    print("各Agent验证集准确率")
    print("=" * 60)
    for name in AGENT_ORDER:
        if name in accuracies:
            print(f"  {name:15s}: {accuracies[name]:.4f}")

    weights = accuracies_to_weights(accuracies)
    print("\n按验证准确率归一化后的投票权重:")
    for name, w in weights.items():
        print(f"  {name:15s}: {w:.4f}")
    print(f"  （旧权重: {AGENT_WEIGHTS}）")

    save_json({"accuracies": accuracies, "weights": weights}, WEIGHTS_PATH)
    print(f"\n已保存: {WEIGHTS_PATH}")
    print("提示：将 weights 同步到 config.AGENT_WEIGHTS 即可在多Agent系统中生效；")
    print("      Orchestrator 会自动读取 checkpoints/agent_weights.json（若存在）。")
    return accuracies, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-subjects", type=int, default=90)
    parser.add_argument("--epochs-per-subject", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--weights-only", action="store_true",
                        help="跳过训练，仅按已记录的验证集准确率回写权重")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs-per-record", type=int, default=None,
                        help="每条记录最多取多少 epoch（控制内存与 CPU 训练时长）")
    args = parser.parse_args()
    main(epochs=args.epochs, n_subjects=args.n_subjects,
         epochs_per_subject=args.epochs_per_subject, n_select=args.n_select,
         device=args.device, only=args.only, weights_only=args.weights_only,
         seed=args.seed, max_epochs_per_record=args.max_epochs_per_record)
