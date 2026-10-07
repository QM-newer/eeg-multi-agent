# scripts/train_binary_agents.py
"""
二分类训练：Normal vs Abnormal（Borderline 合并到 Abnormal）

训练全部 4 个诊断 Agent 并保存到 checkpoints/<agent>_binary/ 目录。
LightGBM 会重建专用的 mRMR 融合器。

用法：
    python scripts/train_binary_agents.py
    python scripts/train_binary_agents.py --agents lightgbm,eegnet
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import (
    BINARY_MODEL_PATHS, BINARY_FUSION_PATH, N_BINARY_CLASSES,
    DATA_CONFIG, TRAIN_CONFIG, resolve_path,
)
from training.common import (
    prepare_dataset, extract_multi_domain_features, set_seed,
    build_model, train_torch_model, make_loaders, remap_labels_binary,
    save_val_accuracies,
)
from data.epoch_cache import compute_class_weights
from features.fusion import FeatureFusion
from utils.io_utils import ensure_dir


def train_binary_lightgbm(n_select=128, max_epochs_per_record=None, n_jobs=None):
    """训练二分类 LightGBM（含 mRMR 融合器重建）"""
    print("\n" + "=" * 60)
    print("训练 LightGBM [二分类]")
    print("=" * 60)

    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           verbose=True, label_mode="binary")
    X, y = data["X"], data["y"]

    print("\n[2/5] 提取多域特征...")
    feat_train, dims = extract_multi_domain_features(
        X[data["train_idx"]], n_jobs=n_jobs, verbose=True)
    feat_val, _ = extract_multi_domain_features(
        X[data["val_idx"]], n_jobs=n_jobs, verbose=True)
    print(f"  各Agent特征维度: {dims}  拼接后: {feat_train.shape[1]} 维")
    y_train, y_val = y[data["train_idx"]], y[data["val_idx"]]

    print(f"\n[3/5] mRMR 特征选择（目标 {n_select} 维）...")
    fusion = FeatureFusion(n_select=n_select)
    feat_train_sel = fusion.fit_transform(feat_train, y_train)
    feat_val_sel = fusion.transform(feat_val)
    info = fusion.mrmr_info_
    print(f"  {info['n_input']} 维 → {info['n_select']} 维 "
          f"(准则={info['criterion']}, 选中特征平均MI={info['mean_mi_selected']:.4f})")

    fusion.save(BINARY_FUSION_PATH)
    print(f"  融合器已保存: {BINARY_FUSION_PATH}")

    print("\n[4/5] 训练 LightGBM...")
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )
    model.fit(feat_train_sel, y_train,
              eval_set=[(feat_val_sel, y_val)],
              eval_metric="binary_logloss")

    print("\n[5/5] 评估与保存...")
    val_acc = float(np.mean(model.predict(feat_val_sel) == y_val))
    val_prob = model.predict_proba(feat_val_sel)[:, 1]
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, val_prob)
    print(f"  验证集准确率: {val_acc:.4f}  AUC: {val_auc:.4f}")

    ensure_dir(os.path.dirname(BINARY_MODEL_PATHS["lightgbm"]))
    import joblib
    joblib.dump(model, BINARY_MODEL_PATHS["lightgbm"])
    print(f"  模型已保存: {BINARY_MODEL_PATHS['lightgbm']}")

    save_val_accuracies({"lightgbm_binary": val_acc})
    return val_acc


def train_binary_deep(name, epochs=30, batch_size=None, device=None,
                      max_epochs_per_record=None):
    """训练二分类深度模型"""
    from training.common import train_deep_agent
    return train_deep_agent(
        name, epochs=epochs, batch_size=batch_size, device=device,
        max_epochs_per_record=max_epochs_per_record, label_mode="binary",
    )


def main():
    parser = argparse.ArgumentParser(description="训练二分类诊断Agent")
    parser.add_argument("--agents", default="all",
                        help="逗号分隔的Agent列表，默认all=lightgbm,eegnet,tcn,transformer")
    parser.add_argument("--epochs", type=int, default=30, help="深度模型训练轮次")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-select", type=int, default=128, help="LightGBM mRMR选择维度")
    parser.add_argument("--max-epochs-per-record", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=None, help="特征提取并行进程数")
    args = parser.parse_args()

    if args.device is None:
        from config import ensure_device
        args.device = ensure_device()

    if args.agents == "all":
        agent_list = ["lightgbm", "eegnet", "tcn", "transformer"]
    else:
        agent_list = [a.strip() for a in args.agents.split(",")]

    print("=" * 60)
    print("二分类诊断Agent训练")
    print(f"  Agents: {agent_list}")
    print(f"  映射: Normal(0)→0, Borderline(1)+Abnormal(2)→1")
    print(f"  设备: {args.device}")
    print("=" * 60)

    results = {}
    for name in agent_list:
        if name == "lightgbm":
            acc = train_binary_lightgbm(
                n_select=args.n_select,
                max_epochs_per_record=args.max_epochs_per_record,
                n_jobs=args.n_jobs,
            )
        else:
            acc = train_binary_deep(
                name, epochs=args.epochs, device=args.device,
                max_epochs_per_record=args.max_epochs_per_record,
            )
        results[name] = acc

    print("\n" + "=" * 60)
    print("训练完成汇总")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name}: val_acc = {acc:.4f}")
    print(f"\n模型保存目录: checkpoints/*_binary/")


if __name__ == "__main__":
    main()
