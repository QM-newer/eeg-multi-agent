# scripts/train_subject_level.py
"""
被试级建模实验：以"一份记录 = 一个样本"训练 LightGBM

动机：epoch 级（2 秒片段）判别力很弱（最佳单特征 AUC 仅 0.59，模型退化为全预测多数类），
而同一被试多个 epoch 取平均后最佳 AUC 升到 0.65~0.73。临床也是"看一段记录下一个结论"，
因此这里直接以被试为单位建模，用来衡量这套特征体系的真实上限。

流程：
    每记录取 N 个 epoch → 多域特征(361维) → 按被试平均 → mRMR 选 128 维 → LightGBM
    同时给出三分类与"Normal vs Abnormal"二分类（剔除 Borderline）两种口径

用法：
    python scripts/train_subject_level.py --max-epochs-per-record 10 --n-select 128
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import DATA_CONFIG, CLASS_NAMES, RANDOM_SEED
from features.fusion import FeatureFusion
from training.common import extract_multi_domain_features, prepare_dataset
from utils.io_utils import ensure_dir, save_json
from config import resolve_path


def build_subject_features(data, max_epochs_per_record, n_jobs=None):
    """(n_epochs, C, T) → 按被试平均的 (n_subjects, n_features)"""
    X, y, sids = data["X"], data["y"], np.asarray(data["subject_ids"])
    uniq = sorted(set(sids.tolist()))
    print(f"  提取特征: {X.shape[0]} 个 epoch，{len(uniq)} 个被试 ...")
    t0 = time.time()
    feats, dims = extract_multi_domain_features(X, n_jobs=n_jobs, verbose=True)
    print(f"  特征提取完成 {time.time() - t0:.0f}s  维度 {feats.shape[1]}")

    sids = np.asarray(sids)
    F, ys, out_sids = [], [], []
    for u in uniq:
        m = sids == u
        F.append(feats[m].mean(axis=0))
        ys.append(y[m][0])
        out_sids.append(u)
    return np.vstack(F), np.asarray(ys), np.array(out_sids), dims


def main(max_epochs_per_record=10, n_select=128, n_jobs=None):
    print("=" * 74)
    print("被试级建模（一份记录 = 一个样本）")
    print("=" * 74)

    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize="per_channel", verbose=False)
    F, y, sids, dims = build_subject_features(data, max_epochs_per_record, n_jobs)

    # 划分：epoch 级划分本就是按被试做的，直接用 subject_id 归属判断
    tr_ids = set(np.asarray(data["subject_ids"])[data["train_idx"]].tolist())
    va_ids = set(np.asarray(data["subject_ids"])[data["val_idx"]].tolist())
    te_ids = set(np.asarray(data["subject_ids"])[data["test_idx"]].tolist())
    tr = np.array([s in tr_ids for s in sids])
    va = np.array([s in va_ids for s in sids])
    te = np.array([s in te_ids for s in sids])
    print(f"  被试: train={tr.sum()}  val={va.sum()}  test={te.sum()}")

    import lightgbm as lgb

    results = {}
    for tag, classes in (("3class", [0, 1, 2]), ("2class_N_vs_A", [0, 2])):
        print(f"\n---- {tag} ----")
        # 先筛掉不参与的类别，再做标签映射（Borderline 在二分类口径下要剔除）
        sel = np.isin(y, classes)
        ymap = {c: i for i, c in enumerate(classes)}
        Fs = F[sel]
        yy = np.array([ymap[v] for v in y[sel]])
        sf = sids[sel]
        tr = np.array([s in tr_ids for s in sf])
        va = np.array([s in va_ids for s in sf])
        te = np.array([s in te_ids for s in sf])

        fusion = FeatureFusion(n_select=min(n_select, Fs.shape[1]))
        Ftr = fusion.fit_transform(Fs[tr], yy[tr])
        Fva = fusion.transform(Fs[va])
        Fte = fusion.transform(Fs[te])
        yva, yte = yy[va], yy[te]
        print(f"  mRMR: {F.shape[1]} 维 → {Ftr.shape[1]} 维")

        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=RANDOM_SEED, verbose=-1, n_jobs=8)
        model.fit(Ftr, yy[tr],
                  eval_set=[(Fva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        for name, (Fp, yt) in (("val", (Fva, yva)), ("test", (Fte, yte))):
            pred = model.predict(Fp)
            acc = accuracy_score(yt, pred)
            f1m = f1_score(yt, pred, average="macro")
            base = max(np.bincount(yt).tolist()) / len(yt)
            print(f"  [{name}] acc={acc:.4f}  macro-F1={f1m:.4f}  "
                  f"多数类基线={base:.4f}  n={len(yt)}")
            print("     混淆矩阵(行=真实):")
            for row in confusion_matrix(yt, pred).tolist():
                print("      ", row)
            if name == "test":
                print(classification_report(yt, pred, digits=3))
            results[f"{tag}_{name}"] = {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1m), 4),
                "majority_baseline": round(float(base), 4),
                "n": int(len(yt)),
                "confusion_matrix": confusion_matrix(yt, pred).tolist(),
            }

    out = resolve_path("outputs/subject_level_train.json")
    ensure_dir(os.path.dirname(out))
    save_json({"max_epochs_per_record": max_epochs_per_record,
               "n_features": int(F.shape[1]), "results": results}, out)
    print(f"\n结果已保存: {out}")

    # 顺便把"被试级"融合器/模型保存下来，供后续接入（与epoch级分开存放）
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs-per-record", type=int, default=10)
    ap.add_argument("--n-select", type=int, default=128)
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    main(max_epochs_per_record=a.max_epochs_per_record, n_select=a.n_select, n_jobs=a.n_jobs)
