# scripts/run_ablation.py
"""
消融实验（论文计划 5.4）— 特征侧 A1~A3 + 事件特征 on/off

基于 scripts/extract_full_features.py 预提取的 614 维特征（列块布局见其 manifest），
对每个"留一块"变体：mRMR(128) → LightGBM 二分类 → 被试级聚合评估（test@20）。

变体（leave-one-block-out）：
    full          全部 614 维（含事件）      —— 完整特征体系
    no_event      去掉事件块 [532:614)       —— 事件特征贡献（对照基线 = 原 lightgbm_binary 口径）
    A1_no_time    去掉时域块 [0:228)
    A2_no_freq    去掉频域块 [228:380)
    A3_no_timefreq 去掉时频块 [380:532)

full 变体同时把模型与融合器保存为事件版基线（checkpoints/lightgbm_binary_event/
与 fusion_binary_event/），供后续融合实验使用。

前置条件:
    python scripts/extract_full_features.py

输出:
    outputs/ablation_features.json

用法:
    python scripts/run_ablation.py                # 全部变体
    python scripts/run_ablation.py --variants full,no_event
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    BINARY_EVENT_MODEL_PATH, BINARY_EVENT_FUSION_PATH,
    BINARY_LABEL_MAP, resolve_path,
)
from features.fusion import FeatureFusion
from utils.io_utils import ensure_dir, save_json

FEATURE_DIR = os.path.join("E:/eeg_asd_cache", "features")

# 列块布局（与 _make_feature_agents 拼接顺序一致）
BLOCKS = {
    "time_domain": (0, 228),
    "freq_domain": (228, 380),
    "timefreq": (380, 532),
    "event": (532, 614),
}

VARIANTS = {
    "full": [],                          # 不去掉任何块
    "no_event": ["event"],
    "A1_no_time": ["time_domain"],
    "A2_no_freq": ["freq_domain"],
    "A3_no_timefreq": ["timefreq"],
}

LGBM_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=-1,
    min_child_samples=10, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, objective="binary", random_state=42,
    n_jobs=-1, verbose=-1, class_weight="balanced",
)


def load_split(split):
    feat = np.load(os.path.join(FEATURE_DIR, f"{split}_feat.npy"))
    meta = np.load(os.path.join(FEATURE_DIR, f"{split}_meta.npz"), allow_pickle=True)
    y3 = meta["y"]
    y_bin = np.array([BINARY_LABEL_MAP[int(v)] for v in y3], dtype=np.int64)
    return feat, y3, y_bin, meta["subject_ids"]


def drop_blocks(feat, blocks_to_drop):
    if not blocks_to_drop:
        return feat
    drop_idx = np.concatenate([np.arange(BLOCKS[b][0], BLOCKS[b][1]) for b in blocks_to_drop])
    keep = np.setdiff1d(np.arange(feat.shape[1]), drop_idx)
    return feat[:, keep]


def subject_metrics(epoch_prob, y_bin_epoch, sids, method="mean"):
    """epoch 概率 → 被试级聚合 → 二分类指标"""
    uniq = sorted(set(sids))
    if method == "mean":
        p_sub = np.array([epoch_prob[sids == u].mean() for u in uniq])
    else:
        p_sub = np.array([np.percentile(epoch_prob[sids == u], 90) for u in uniq])
    y_sub = np.array([y_bin_epoch[sids == u][0] for u in uniq])
    pred = (p_sub >= 0.5).astype(int)
    return {
        "n_subjects": len(uniq),
        "accuracy": round(float(accuracy_score(y_sub, pred)), 4),
        "macro_f1": round(float(f1_score(y_sub, pred, average="macro")), 4),
        "auc": round(float(roc_auc_score(y_sub, p_sub)), 4),
        "baseline": round(float(max(np.bincount(y_sub)) / len(y_sub)), 4),
    }


def run_variant(name, drop, feat_train, y_train, feat_test, y_test_bin, sids_test,
                n_select=128, save_event_model=False):
    blocks = ", ".join(drop) if drop else "无"
    print(f"\n--- 变体 {name}（去掉: {blocks}）---")
    t0 = time.time()

    f_train = drop_blocks(feat_train, drop)
    f_test = drop_blocks(feat_test, drop)
    print(f"  特征维度: {f_train.shape[1]}")

    fusion = FeatureFusion(n_select=n_select, include_event=not any(
        b == "event" for b in drop))
    f_train_sel = fusion.fit_transform(f_train, y_train)
    f_test_sel = fusion.transform(f_test)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(f_train_sel, y_train)

    val_acc_epoch = float(np.mean(model.predict(f_train_sel) == y_train))
    prob_test = model.predict_proba(f_test_sel)[:, 1]

    res = {
        "n_features": int(f_train.shape[1]),
        "train_epoch_acc": round(val_acc_epoch, 4),
        "subject_mean": subject_metrics(prob_test, y_test_bin, sids_test, "mean"),
        "subject_p90": subject_metrics(prob_test, y_test_bin, sids_test, "p90"),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  被试级(mean): acc={res['subject_mean']['accuracy']}  "
          f"AUC={res['subject_mean']['auc']}  F1={res['subject_mean']['macro_f1']}  "
          f"(基线 {res['subject_mean']['baseline']})")
    print(f"  被试级(p90):  acc={res['subject_p90']['accuracy']}  "
          f"AUC={res['subject_p90']['auc']}")

    if save_event_model:
        ensure_dir(os.path.dirname(BINARY_EVENT_MODEL_PATH))
        import joblib
        joblib.dump(model, BINARY_EVENT_MODEL_PATH)
        fusion.save(BINARY_EVENT_FUSION_PATH)
        print(f"  事件版模型已保存: {BINARY_EVENT_MODEL_PATH}")

    return res


def main(variants=None, n_select=128):
    print("=" * 74)
    print("消融实验 — 特征侧（A1~A3 + 事件特征 on/off）")
    print(f"特征目录: {FEATURE_DIR}")
    print("=" * 74)

    if not os.path.exists(os.path.join(FEATURE_DIR, "test_feat.npy")):
        print("未找到预提取特征，请先运行: python scripts/extract_full_features.py")
        return

    feat_train, y3_train, y_train, _ = load_split("train")
    feat_test, _, y_test, sids_test = load_split("test")
    print(f"train: {feat_train.shape}  test: {feat_test.shape} "
          f"({len(set(sids_test))} 被试)")

    todo = variants if variants else list(VARIANTS.keys())
    results = {"protocol": "LightGBM binary, mRMR-128, subject-level agg on test@20",
               "variants": {}}
    for name in todo:
        drop = VARIANTS[name]
        results["variants"][name] = run_variant(
            name, drop, feat_train, y_train, feat_test, y_test, sids_test,
            n_select=n_select, save_event_model=(name == "full"))

    # 汇总表
    print("\n" + "=" * 74)
    print("汇总（被试级 mean 聚合, test 238 被试）")
    print(f"  {'变体':<16s} {'维度':>5s} {'Acc':>7s} {'AUC':>7s} {'macroF1':>8s}")
    print("  " + "─" * 50)
    base = results["variants"].get("no_event", {}).get("subject_mean", {})
    for name, r in results["variants"].items():
        m = r["subject_mean"]
        print(f"  {name:<16s} {r['n_features']:>5d} {m['accuracy']:>7.4f} "
              f"{m['auc']:>7.4f} {m['macro_f1']:>8.4f}")
    if base:
        print("  (对照 no_event：完整体系应不差于去掉任一块)")

    out_path = resolve_path("outputs/ablation_features.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(results, out_path)
    print(f"\n结果已保存: {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None, help="逗号分隔变体名，默认全部")
    ap.add_argument("--n-select", type=int, default=128)
    a = ap.parse_args()
    vs = a.variants.split(",") if a.variants else None
    main(variants=vs, n_select=a.n_select)
