# training/train_lightgbm.py
"""
LightGBM 诊断Agent 训练脚本

流程：合成/真实数据 → 三个特征Agent提取多域特征 → mRMR 融合选择 → 训练 LightGBM

用法：
    python training/train_lightgbm.py [--epochs-per-subject 5] [--n-select 128]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import MODEL_PATHS, N_CLASSES, resolve_path
from features.fusion import FeatureFusion
from training.common import (
    prepare_dataset, extract_multi_domain_features, set_seed,
    save_val_accuracies, accuracies_to_weights,
)
from utils.io_utils import ensure_dir


FUSION_PATH = resolve_path("checkpoints/fusion/fusion.joblib")


def main(n_subjects=90, epochs_per_subject=5, n_select=128, seed=None,
         max_epochs_per_record=None, n_jobs=None):
    print("=" * 60)
    print("训练 LightGBM 诊断Agent")
    print("=" * 60)

    set_seed(seed)

    print("\n[1/5] 准备数据...")
    data = prepare_dataset(n_subjects=n_subjects, epochs_per_subject=epochs_per_subject,
                           max_epochs_per_record=max_epochs_per_record, verbose=True)
    X, y = data["X"], data["y"]
    print(f"  数据来源: {data.get('source')}")
    print(f"  样本: {len(X)}  形状: {X.shape[1:]}  "
          f"train/val/test = {len(data['train_idx'])}/{len(data['val_idx'])}/{len(data['test_idx'])}")

    print("\n[2/5] 提取多域特征（时域 + 频域 + 时频域）...")
    feat_train, dims = extract_multi_domain_features(X[data["train_idx"]],
                                                     n_jobs=n_jobs, verbose=True)
    feat_val, _ = extract_multi_domain_features(X[data["val_idx"]],
                                                n_jobs=n_jobs, verbose=True)
    print(f"  各Agent特征维度: {dims}  拼接后: {feat_train.shape[1]} 维")
    y_train, y_val = y[data["train_idx"]], y[data["val_idx"]]

    print(f"\n[3/5] mRMR 特征选择（目标 {n_select} 维）...")
    fusion = FeatureFusion(n_select=n_select)
    feat_train_sel = fusion.fit_transform(feat_train, y_train)
    feat_val_sel = fusion.transform(feat_val)
    info = fusion.mrmr_info_
    print(f"  {info['n_input']} 维 → {info['n_select']} 维 "
          f"(准则={info['criterion']}, 选中特征平均MI={info['mean_mi_selected']:.4f})")

    fusion.save(FUSION_PATH)
    print(f"  融合器已保存: {FUSION_PATH}")

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
        objective="multiclass",
        num_class=N_CLASSES,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",     # Borderline 仅约 10%，必须平衡
    )
    model.fit(feat_train_sel, y_train,
              eval_set=[(feat_val_sel, y_val)],
              eval_metric="multi_logloss")

    print("\n[5/5] 评估与保存...")
    val_acc = float(np.mean(model.predict(feat_val_sel) == y_val))
    print(f"  验证集准确率: {val_acc:.4f}")

    ensure_dir(os.path.dirname(MODEL_PATHS["lightgbm"]))
    import joblib
    joblib.dump(model, MODEL_PATHS["lightgbm"])
    print(f"  模型已保存: {MODEL_PATHS['lightgbm']}")

    path = save_val_accuracies({"lightgbm": val_acc})
    print(f"  验证准确率已记录: {path}  (权重 = {accuracies_to_weights({'lightgbm': val_acc})})")
    print("\n训练完成！")
    return val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=90)
    parser.add_argument("--epochs-per-subject", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs-per-record", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None, help="特征提取并行进程数")
    args = parser.parse_args()
    main(n_subjects=args.n_subjects, epochs_per_subject=args.epochs_per_subject,
         n_select=args.n_select, seed=args.seed,
         max_epochs_per_record=args.max_epochs_per_record, n_jobs=args.n_jobs)
