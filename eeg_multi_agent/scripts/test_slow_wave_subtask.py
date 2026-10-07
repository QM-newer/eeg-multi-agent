# scripts/test_slow_wave_subtask.py
"""
慢波子任务可行性实验

目的：判断瓶颈在"特征层"还是"标签映射层"

逻辑：
    1. 用 delta 频段特征（相对功率、绝对功率、峰频等）构造"慢波指标"
    2. 将三分类标签映射为"慢波异常"二分类：
       - 慢波阳性 = Abnormal 中 delta 占比 > 中位数的那些记录
       - 慢波阴性 = Normal 记录
       - Borderline 排除（定义模糊）
    3. 用 LightGBM (5-fold CV) 测 AUC
       - 若 AUC > 0.70：现有特征能捕获慢波，瓶颈在"慢波→诊断"映射
       - 若 AUC ~0.55：特征/窗口真不行，必须投事件级特征

用法：
    python scripts/test_slow_wave_subtask.py --n-per-class 60
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from config import DATA_CONFIG, RANDOM_SEED
from data.epoch_cache import EpochCache
from training.common import extract_multi_domain_features
from utils.io_utils import ensure_dir, save_json
from config import resolve_path


def auc_simple(y, score):
    """手工 AUC（不依赖 sklearn，便于无 sklearn 环境回退）"""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney
    u = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                u += 1.0
            elif p == n:
                u += 0.5
    return float(u / (len(pos) * len(neg)))


def main(n_per_class=60, epochs_per_record=8, n_jobs=None):
    print("=" * 74)
    print("慢波子任务可行性实验")
    print("=" * 74)

    cache = EpochCache()
    rng = np.random.default_rng(RANDOM_SEED)

    # --- 第 1 步：提取 Normal 和 Abnormal 记录的特征 ---
    # 只取 Normal (0) 和 Abnormal (2)，排除 Borderline (1)
    picks = {}
    for cls, name in ((0, "Normal"), (2, "Abnormal")):
        idx = [i for i, r in enumerate(cache.rows)
               if int(r["label"]) == cls and r["split"] == "train"]
        sel = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        picks[cls] = sorted(sel.tolist())
        print(f"  {name}: 抽取 {len(picks[cls])} 条记录")

    # 逐记录提取多域特征
    rec_features = {}  # rec_idx -> (features_mean, label)
    for cls in picks:
        for i in picks[cls]:
            arr = cache._load(i)
            n = min(epochs_per_record, arr.shape[0])
            sel_ep = np.linspace(0, arr.shape[0] - 1, n).astype(int)
            X = np.stack([np.asarray(arr[k], dtype=np.float32) for k in sel_ep])
            f, dims = extract_multi_domain_features(X, n_jobs=n_jobs)
            # 被试级：各 epoch 特征取平均
            rec_features[i] = (f.mean(axis=0), cls)

    rec_ids = sorted(rec_features.keys())
    F = np.vstack([rec_features[i][0] for i in rec_ids])
    y_raw = np.array([rec_features[i][1] for i in rec_ids])

    n_features = F.shape[1]
    print(f"\n  特征: {n_features} 维, 记录数: {len(rec_ids)}")
    print(f"    Normal: {int((y_raw == 0).sum())}, Abnormal: {int((y_raw == 2).sum())}")

    # --- 第 2 步：构造慢波指标 ---
    # 用 delta 相关特征（频域 Agent 的前 19 个特征 = delta_relative per channel）
    # 频域特征排列: 5 bands × 19 channels + 3 global × 19 channels = 152 per 19 ch
    # 实际排列: delta_rel(19), theta_rel(19), alpha_rel(19), beta_rel(19), gamma_rel(19),
    #           peak_freq(19), spectral_edge(19), spectral_entropy(19) = 152

    # 新版 freq_domain_agent: 每通道 8 个特征 (5 relative + peak_freq + sef95 + entropy)
    # 时域: 每通道 12 个, 共 228
    # 频域: 每通道 8 个, 共 152
    # 时频: 每通道 8 个, 共 152
    # 总计: 532

    # 找 delta 相对功率的列索引（频域 Agent 的前 19 列）
    # 需要先知道各 Agent 的维度分配
    # 简单方法：重新提取一个样本，记录各 Agent 的输出维度
    sample_data = np.zeros((19, 500), dtype=np.float64)
    from agents.feature_agents.time_domain_agent import TimeDomainAgent
    from agents.feature_agents.freq_domain_agent import FreqDomainAgent
    from agents.feature_agents.timefreq_agent import TimeFreqAgent
    td = TimeDomainAgent("td")
    fd = FreqDomainAgent("fd", sfreq=250)
    tf = TimeFreqAgent("tf")
    td_dim = len(td.predict(sample_data)["extra"]["features"])
    fd_dim = len(fd.predict(sample_data)["extra"]["features"])
    tf_dim = len(tf.predict(sample_data)["extra"]["features"])
    print(f"  各Agent维度: time_domain={td_dim}, freq_domain={fd_dim}, timefreq={tf_dim}")

    # delta 相对功率 = 频域 Agent 的第 0~18 列
    delta_cols = list(range(td_dim, td_dim + 19))
    # 也可以用所有频域特征
    freq_cols = list(range(td_dim, td_dim + fd_dim))

    # 被试级 delta 相对功率均值（跨通道）
    delta_power = F[:, delta_cols].mean(axis=1)

    # --- 第 3 步：定义慢波标签 ---
    # 策略 A：Abnormal 中 delta 功率高于中位数的 → 慢波阳性
    abnormal_mask = y_raw == 2
    normal_mask = y_raw == 0
    ab_delta_median = np.median(delta_power[abnormal_mask])

    # 慢波二分类标签
    y_slow = np.zeros(len(y_raw), dtype=int)
    # Normal → 0 (慢波阴性)
    # Abnormal 且 delta > 中位数 → 1 (慢波阳性)
    y_slow[abnormal_mask & (delta_power > ab_delta_median)] = 1

    n_slow_pos = int(y_slow.sum())
    n_slow_neg = int((y_slow == 0).sum())
    print(f"\n  慢波标签分布: 阳性={n_slow_pos}, 阴性={n_slow_neg}")
    slow_baseline = max(n_slow_pos, n_slow_neg) / (n_slow_pos + n_slow_neg)
    print(f"  多数类基线: {slow_baseline:.4f}")

    # --- 第 4 步：LightGBM 5-fold CV ---
    print("\n  LightGBM 5-fold CV（慢波二分类）...")
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs, fold_accs, fold_f1s = [], [], []

    # 用所有多域特征（不限于 delta）
    for fold, (train_idx, val_idx) in enumerate(skf.split(F, y_slow)):
        X_tr, X_val = F[train_idx], F[val_idx]
        y_tr, y_val = y_slow[train_idx], y_slow[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            max_depth=-1, min_child_samples=5, subsample=0.9,
            colsample_bytree=0.8, objective="binary",
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            class_weight="balanced",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric="binary_logloss")
        prob = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)

        auc_val = roc_auc_score(y_val, prob) if len(set(y_val)) > 1 else 0.5
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        fold_aucs.append(auc_val)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"    fold {fold+1}: AUC={auc_val:.4f}  acc={acc:.4f}  macro-F1={f1:.4f}")

    mean_auc = np.mean(fold_aucs)
    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    print(f"\n  5-fold 均值: AUC={mean_auc:.4f}  acc={mean_acc:.4f}  macro-F1={mean_f1:.4f}")

    # --- 第 5 步：对比——用原始三分类标签在同一特征上做 LightGBM ---
    print("\n  对比：同一特征 + 原始三分类标签（Normal vs Abnormal 二分类）...")
    y_binary = np.zeros(len(y_raw), dtype=int)
    y_binary[y_raw == 2] = 1  # Normal=0, Abnormal=1

    fold_aucs_3c, fold_accs_3c = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(F, y_binary)):
        X_tr, X_val = F[train_idx], F[val_idx]
        y_tr, y_val = y_binary[train_idx], y_binary[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            max_depth=-1, min_child_samples=5, subsample=0.9,
            colsample_bytree=0.8, objective="binary",
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            class_weight="balanced",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric="binary_logloss")
        prob = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)

        auc_val = roc_auc_score(y_val, prob) if len(set(y_val)) > 1 else 0.5
        acc = accuracy_score(y_val, pred)
        fold_aucs_3c.append(auc_val)
        fold_accs_3c.append(acc)
        print(f"    fold {fold+1}: AUC={auc_val:.4f}  acc={acc:.4f}")

    mean_auc_3c = np.mean(fold_aucs_3c)
    mean_acc_3c = np.mean(fold_accs_3c)
    print(f"  5-fold 均值: AUC={mean_auc_3c:.4f}  acc={mean_acc_3c:.4f}")

    # --- 第 6 步：诊断结论 ---
    print("\n" + "=" * 74)
    print("诊断结论")
    print("=" * 74)
    print(f"  慢波子任务 AUC:       {mean_auc:.4f}")
    print(f"  原始二分类 AUC:       {mean_auc_3c:.4f}")
    print(f"  差异 (慢波 - 原始):   {mean_auc - mean_auc_3c:+.4f}")

    if mean_auc > 0.70 and mean_auc - mean_auc_3c > 0.05:
        print("\n  ✅ 现有特征能捕获慢波信号，但映射到临床标签时丢失了。")
        print("     瓶颈在「慢波→诊断」映射，建议：")
        print("     - 改为二分类 + 置信度（Borderline 低置信度判可疑）")
        print("     - Stacking 融合（元模型可学到映射规则）")
        print("     - 事件级特征优先级可降低")
    elif mean_auc > 0.60:
        print("\n  ⚡ 现有特征对慢波有中等判别力，但不够强。")
        print("     部分瓶颈在特征层（频段统计量捕获一过性事件能力有限）。")
        print("     建议：事件级特征 + 二分类口径同时推进")
    else:
        print("\n  ❌ 现有特征连慢波都难以预测。")
        print("     瓶颈在特征层——全通道频段统计量无法捕获一过性临床事件。")
        print("     必须投入事件级特征（尖波/棘波检测、不对称度）。")

    report = {
        "slow_wave_auc": round(float(mean_auc), 4),
        "slow_wave_acc": round(float(mean_acc), 4),
        "slow_wave_f1": round(float(mean_f1), 4),
        "original_binary_auc": round(float(mean_auc_3c), 4),
        "original_binary_acc": round(float(mean_acc_3c), 4),
        "n_slow_positive": n_slow_pos,
        "n_slow_negative": n_slow_neg,
        "slow_baseline": round(float(slow_baseline), 4),
        "n_features": n_features,
        "n_records": len(rec_ids),
        "feature_dims": {"time_domain": td_dim, "freq_domain": fd_dim, "timefreq": tf_dim},
    }

    out = resolve_path("outputs/slow_wave_subtask.json")
    ensure_dir(os.path.dirname(out))
    save_json(report, out)
    print(f"\n结果已保存: {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=60,
                    help="Normal 和 Abnormal 各抽取的记录数")
    ap.add_argument("--epochs-per-record", type=int, default=8,
                    help="每条记录取多少 epoch（被试级聚合用）")
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    main(n_per_class=a.n_per_class, epochs_per_record=a.epochs_per_record,
         n_jobs=a.n_jobs)
