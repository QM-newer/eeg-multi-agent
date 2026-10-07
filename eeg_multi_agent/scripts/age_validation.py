# scripts/age_validation.py
"""
年龄校正验证实验（两步走方案 第一步）

在有年龄的子集上：
  A: 不加年龄特征（基线）
  B: 加年龄作为额外特征

如果 B 比 A 提升显著（+0.02 以上），说明年龄确实有价值。

用法：
    python scripts/age_validation.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from training.common import (
    prepare_dataset, extract_multi_domain_features, set_seed, remap_labels_binary,
)
from data.epoch_cache import compute_class_weights
from features.fusion import FeatureFusion
from config import resolve_path


def load_age_map():
    """SUBJ_XXXX → age"""
    sm = pd.read_csv(resolve_path("data/subject_master.csv"))
    age_map = {}
    for _, row in sm.iterrows():
        aid = row["anonymous_id"]
        age = row["age"]
        if pd.notna(age):
            num = aid.split("_")[-1] if "_" in str(aid) else str(aid)
            age_map[f"SUBJ_{num}"] = float(age)
    return age_map


def main(max_epochs_per_record=10):
    print("=" * 74)
    print("年龄校正验证实验（第一步：有年龄子集）")
    print("=" * 74)

    set_seed()
    age_map = load_age_map()

    # 加载数据
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           verbose=True, label_mode="3class")
    X = data["X"]
    y_3class = data["y"]
    y_binary = remap_labels_binary(y_3class)
    subject_ids = np.asarray(data["subject_ids"])

    # 筛选有年龄的 epoch
    has_age = np.array([sid in age_map for sid in subject_ids])
    print(f"\n  有年龄的 epoch: {has_age.sum()}/{len(has_age)} ({100*has_age.mean():.1f}%)")

    # 只保留有年龄的 epoch
    age_mask = has_age
    ages = np.array([age_map.get(sid, 6.0) for sid in subject_ids])

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]

    # 在训练/验证中筛选有年龄的
    train_age_mask = age_mask[train_idx]
    val_age_mask = age_mask[val_idx]
    train_idx_age = train_idx[train_age_mask]
    val_idx_age = val_idx[val_age_mask]

    print(f"  Train with age: {len(train_idx_age)}  Val with age: {len(val_idx_age)}")

    # 提取特征
    print("\n[1/3] 提取特征...")
    feat_train, dims = extract_multi_domain_features(X[train_idx_age], n_jobs=4, verbose=True)
    feat_val, _ = extract_multi_domain_features(X[val_idx_age], n_jobs=4, verbose=True)
    print(f"  特征维度: {feat_train.shape[1]}")

    y_train = y_binary[train_idx_age]
    y_val = y_binary[val_idx_age]
    ages_train = ages[train_idx_age]
    ages_val = ages[val_idx_age]

    # 方案 A：不加年龄
    print("\n[2/3] 方案 A：不加年龄特征...")
    fusion_a = FeatureFusion(n_select=128)
    feat_train_a = fusion_a.fit_transform(feat_train, y_train)
    feat_val_a = fusion_a.transform(feat_val)

    import lightgbm as lgb
    model_a = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced")
    model_a.fit(feat_train_a, y_train, eval_set=[(feat_val_a, y_val)])
    prob_a = model_a.predict_proba(feat_val_a)[:, 1]
    auc_a = roc_auc_score(y_val, prob_a)
    acc_a = accuracy_score(y_val, (prob_a >= 0.5).astype(int))
    print(f"  A: acc={acc_a:.4f}  AUC={auc_a:.4f}")

    # 方案 B：加年龄
    print("\n[3/3] 方案 B：加年龄特征...")
    feat_train_b = np.hstack([feat_train, ages_train.reshape(-1, 1)])
    feat_val_b = np.hstack([feat_val, ages_val.reshape(-1, 1)])
    print(f"  加年龄后维度: {feat_train_b.shape[1]}")

    fusion_b = FeatureFusion(n_select=128)
    feat_train_b_sel = fusion_b.fit_transform(feat_train_b, y_train)
    feat_val_b_sel = fusion_b.transform(feat_val_b)

    model_b = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced")
    model_b.fit(feat_train_b_sel, y_train, eval_set=[(feat_val_b_sel, y_val)])
    prob_b = model_b.predict_proba(feat_val_b_sel)[:, 1]
    auc_b = roc_auc_score(y_val, prob_b)
    acc_b = accuracy_score(y_val, (prob_b >= 0.5).astype(int))
    print(f"  B: acc={acc_b:.4f}  AUC={auc_b:.4f}")

    # 结果
    print("\n" + "=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  A (无年龄): acc={acc_a:.4f}  AUC={auc_a:.4f}")
    print(f"  B (有年龄): acc={acc_b:.4f}  AUC={auc_b:.4f}")
    delta = auc_b - auc_a
    print(f"  Delta AUC: {delta:+.4f}")
    if delta > 0.02:
        print("  ✅ 年龄特征有显著价值，值得做年龄插补")
    elif delta > 0:
        print("  ⚠️ 年龄特征有微弱价值，插补收益可能有限")
    else:
        print("  ❌ 年龄特征无帮助，插补无意义")

    # 检查 mRMR 是否选择了年龄特征
    age_idx = feat_train.shape[1]  # 年龄是最后一个特征
    age_selected = age_idx in fusion_b.selected_
    print(f"  mRMR 选择年龄特征: {age_selected}")

    return {"A": {"acc": float(acc_a), "auc": float(auc_a)},
            "B": {"acc": float(acc_b), "auc": float(auc_b)},
            "delta_auc": float(delta)}


if __name__ == "__main__":
    main()
