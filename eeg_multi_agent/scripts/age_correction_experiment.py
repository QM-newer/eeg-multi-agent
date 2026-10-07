# scripts/age_correction_experiment.py
"""
年龄校正实验

儿童脑电 delta 主导是年龄效应（0-5岁正常），不是异常信号。
本实验：
  1. 对每个特征，用训练集拟合 feature_i = a * age + b + residual
  2. 用 residual 替换原始特征（去除年龄效应）
  3. 重新训练 LightGBM 二分类
  4. 对比校正前后 AUC

用法：
    python scripts/age_correction_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    BINARY_MODEL_PATHS, BINARY_FUSION_PATH, BINARY_LABEL_MAP,
    DATA_CONFIG, RANDOM_SEED, resolve_path,
)
from training.common import (
    prepare_dataset, extract_multi_domain_features, set_seed,
    remap_labels_binary,
)
from features.fusion import FeatureFusion
from data.epoch_cache import compute_class_weights
from utils.io_utils import ensure_dir, save_json


def load_age_map():
    """从 subject_master.csv 加载 subject_id → age 映射

    subject_master 用 ANON_XXXX，epoch 缓存用 SUBJ_XXXX，序号一致。
    同时提供两种键以便查找。
    """
    sm = pd.read_csv(resolve_path("data/subject_master.csv"))
    age_map = {}
    for _, row in sm.iterrows():
        aid = row["anonymous_id"]     # ANON_0001
        age = row["age"]
        if pd.notna(age):
            # 同时注册 ANON_0001 和 SUBJ_0001 两种键
            age_map[aid] = age
            num = aid.split("_")[-1] if "_" in str(aid) else str(aid)
            age_map[f"SUBJ_{num}"] = age
    return age_map


def fit_age_regressors(features, ages, feature_names=None):
    """
    对每个特征拟合 feature_i = a * age + b

    参数:
        features: (n_samples, n_features) float64
        ages: (n_samples,) float64

    返回:
        regressors: list of (slope, intercept) per feature
        residuals: (n_samples, n_features) float64 — 去除年龄效应后的残差
    """
    n_features = features.shape[1]
    regressors = []
    residuals = np.zeros_like(features)
    age_2d = ages.reshape(-1, 1)

    for i in range(n_features):
        reg = LinearRegression()
        reg.fit(age_2d, features[:, i])
        regressors.append((reg.coef_[0], reg.intercept_))
        residuals[:, i] = features[:, i] - reg.predict(age_2d)

    return regressors, residuals


def apply_age_correction(features, ages, regressors):
    """用已有的回归系数去除年龄效应"""
    n_features = features.shape[1]
    residuals = np.zeros_like(features)
    age_2d = ages.reshape(-1, 1)

    for i in range(n_features):
        slope, intercept = regressors[i]
        predicted = slope * age_2d.ravel() + intercept
        residuals[:, i] = features[:, i] - predicted

    return residuals


def main(max_epochs_per_record=100, n_select=128, n_jobs=None):
    print("=" * 74)
    print("年龄校正实验")
    print("=" * 74)

    set_seed()

    # 1. 加载年龄
    age_map = load_age_map()
    print(f"  加载年龄映射: {len(age_map)} 条记录")

    # 2. 准备数据（三分类标签，后映射为二分类）
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           verbose=True, label_mode="3class")
    X, y_3class = data["X"], data["y"]
    y_binary = remap_labels_binary(y_3class)
    subject_ids = np.asarray(data["subject_ids"])

    # 3. 为每个 epoch 映射年龄
    # subject_ids 格式可能是 "SUBJ_0001" 或 "ANON_0001"
    # 需要匹配 subject_master 中的 anonymous_id
    print("\n  映射年龄到 epoch...")
    ages = np.zeros(len(subject_ids), dtype=np.float64)
    missing = 0
    for i, sid in enumerate(subject_ids):
        age = age_map.get(sid, None)
        if age is None or np.isnan(age):
            missing += 1
            age = 6.0  # 中位年龄作为默认值
        ages[i] = age
    print(f"  映射完成: {len(ages)} epoch, 缺失年龄 {missing} 个（用中位年龄6代替）")
    print(f"  年龄分布: mean={ages.mean():.1f}, std={ages.std():.1f}, "
          f"range=[{ages.min():.0f}, {ages.max():.0f}]")

    # 4. 提取特征
    print("\n[1/5] 提取多域特征...")
    train_idx, val_idx = data["train_idx"], data["val_idx"]
    feat_train, dims = extract_multi_domain_features(X[train_idx], n_jobs=n_jobs, verbose=True)
    feat_val, _ = extract_multi_domain_features(X[val_idx], n_jobs=n_jobs, verbose=True)
    print(f"  特征维度: {feat_train.shape[1]}")

    y_train = y_binary[train_idx]
    y_val = y_binary[val_idx]
    ages_train = ages[train_idx]
    ages_val = ages[val_idx]

    # 5. 年龄校正
    print("\n[2/5] 年龄校正（线性回归残差）...")
    regressors, feat_train_corrected = fit_age_regressors(feat_train, ages_train)
    feat_val_corrected = apply_age_correction(feat_val, ages_val, regressors)

    # 检查哪些特征受年龄影响最大
    age_sensitivity = np.array([abs(r[0]) for r in regressors])
    top_age_features = np.argsort(age_sensitivity)[::-1][:10]
    print(f"  受年龄影响最大的特征索引: {top_age_features}")
    print(f"  对应回归系数: {np.round(age_sensitivity[top_age_features], 4)}")

    # 6. 训练未校正的 LightGBM（对照）
    print("\n[3/5] 训练 LightGBM（未校正，对照）...")
    import lightgbm as lgb

    fusion_raw = FeatureFusion(n_select=n_select)
    feat_train_sel = fusion_raw.fit_transform(feat_train, y_train)
    feat_val_sel = fusion_raw.transform(feat_val)

    model_raw = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=10,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        objective="binary", random_state=42, n_jobs=-1,
        verbose=-1, class_weight="balanced",
    )
    model_raw.fit(feat_train_sel, y_train,
                  eval_set=[(feat_val_sel, y_val)],
                  eval_metric="binary_logloss")

    val_pred_raw = model_raw.predict(feat_val_sel)
    val_prob_raw = model_raw.predict_proba(feat_val_sel)[:, 1]
    acc_raw = accuracy_score(y_val, val_pred_raw)
    auc_raw = roc_auc_score(y_val, val_prob_raw)
    print(f"  未校正: val acc={acc_raw:.4f}  AUC={auc_raw:.4f}")

    # 7. 训练年龄校正后的 LightGBM
    print("\n[4/5] 训练 LightGBM（年龄校正后）...")
    fusion_corr = FeatureFusion(n_select=n_select)
    feat_train_sel_corr = fusion_corr.fit_transform(feat_train_corrected, y_train)
    feat_val_sel_corr = fusion_corr.transform(feat_val_corrected)

    model_corr = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=10,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        objective="binary", random_state=42, n_jobs=-1,
        verbose=-1, class_weight="balanced",
    )
    model_corr.fit(feat_train_sel_corr, y_train,
                   eval_set=[(feat_val_sel_corr, y_val)],
                   eval_metric="binary_logloss")

    val_pred_corr = model_corr.predict(feat_val_sel_corr)
    val_prob_corr = model_corr.predict_proba(feat_val_sel_corr)[:, 1]
    acc_corr = accuracy_score(y_val, val_pred_corr)
    auc_corr = roc_auc_score(y_val, val_prob_corr)
    print(f"  校正后: val acc={acc_corr:.4f}  AUC={auc_corr:.4f}")

    # 8. 保存年龄校正的回归器和融合器
    print("\n[5/5] 保存模型...")
    import joblib

    # 保存回归器
    reg_path = resolve_path("checkpoints/age_regressors.joblib")
    ensure_dir(os.path.dirname(reg_path))
    joblib.dump(regressors, reg_path)
    print(f"  年龄回归器已保存: {reg_path}")

    # 保存校正后的融合器和模型
    fusion_corr.save(resolve_path("checkpoints/fusion_binary_agecorrected/fusion.joblib"))
    ensure_dir(os.path.dirname(BINARY_MODEL_PATHS["lightgbm"]).replace("lightgbm_binary", "lightgbm_binary_agecorrected"))
    model_path = resolve_path("checkpoints/lightgbm_binary_agecorrected/model.pkl")
    joblib.dump(model_corr, model_path)
    print(f"  校正后模型已保存: {model_path}")

    # 9. 汇总
    print("\n" + "=" * 74)
    print("年龄校正实验结果（val 集）")
    print("=" * 74)
    print(f"  {'方法':<30s} {'Acc':>6s} {'AUC':>6s}")
    print(f"  {'─'*30} {'─'*6} {'─'*6}")
    print(f"  {'未校正':<30s} {acc_raw:.4f} {auc_raw:.4f}")
    print(f"  {'年龄校正后':<30s} {acc_corr:.4f} {auc_corr:.4f}")
    delta = auc_corr - auc_raw
    if delta > 0:
        print(f"\n  ✅ 年龄校正有效: AUC 提升 +{delta:.4f}")
    else:
        print(f"\n  ❌ 年龄校正无效: AUC 变化 {delta:.4f}")

    report = {
        "uncorrected": {"acc": round(float(acc_raw), 4), "auc": round(float(auc_raw), 4)},
        "age_corrected": {"acc": round(float(acc_corr), 4), "auc": round(float(auc_corr), 4)},
        "delta_auc": round(float(delta), 4),
        "n_features": feat_train.shape[1],
        "top_age_features": top_age_features.tolist(),
    }
    out_path = resolve_path("outputs/age_correction_eval.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(report, out_path)
    print(f"结果已保存: {out_path}")

    return report


if __name__ == "__main__":
    main()
