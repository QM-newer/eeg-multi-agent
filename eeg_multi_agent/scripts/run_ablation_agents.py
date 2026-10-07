# scripts/run_ablation_agents.py
"""
消融实验 — Agent 侧（论文计划 5.4 A4~A6、A9）+ 硬投票基线（Baseline 2）

第一步：收集各诊断 Agent 在 val/test（@20 epochs/record）上的 epoch 级概率并缓存
    - lightgbm / lightgbm_event: 直接用预提取特征（E:/eeg_asd_cache/features/）
      过各自融合器后 predict_proba（与 Agent 内部串行提取等价，快 ~600 倍）
    - eegnet / tcn / transformer: 原始 epoch 逐个 agent.predict（Agent 内部归一化）
    缓存: E:/eeg_asd_cache/features/agent_probs_{split}.npz  (n_epochs, n_agents)

第二步：被试级融合变体（元特征上组合，全部即时计算）
    - stacking_full      完整 stacking（对照 = outputs/stacking_eval.json）
    - A4_no_eegnet       去掉 EEGNet
    - A5_no_tcn          去掉 TCN
    - A6_no_transformer  去掉 Transformer
    - soft_voting        等权平均概率（A9 的"简单平均"对照）
    - hard_voting        被试级多数票（Baseline 2）
    若事件版 LightGBM（checkpoints/lightgbm_binary_event/）存在，则以
    lightgbm_event 替换 lightgbm 再跑一遍全部变体（base=event）。

输出:
    outputs/ablation_agents.json

用法:
    python scripts/run_ablation_agents.py
    python scripts/run_ablation_agents.py --skip-cache   # 复用已缓存的概率
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    BINARY_EVENT_MODEL_PATH, BINARY_EVENT_FUSION_PATH, BINARY_LABEL_MAP,
    BINARY_FUSION_PATH, BINARY_MODEL_PATHS, resolve_path,
)
from data.epoch_cache import EpochCache
from utils.io_utils import ensure_dir, save_json

FEATURE_DIR = os.path.join("E:/eeg_asd_cache", "features")

DEEP_AGENTS = [  # (列名, 模块, 类名, BINARY_MODEL_PATHS 的 key)
    ("eegnet", "agents.diagnosis_agents.eegnet_agent", "EegNetAgent", "eegnet"),
    ("tcn", "agents.diagnosis_agents.tcn_agent", "TcnAgent", "tcn"),
    ("transformer", "agents.diagnosis_agents.transformer_agent", "TransformerAgent",
     "transformer"),
]


# ------------------------------------------------------------------
# 第一步：收集 epoch 级概率
# ------------------------------------------------------------------
def _load_or_extract_feat20(split):
    """按 @20 epochs/record 口径取特征（与 train_stacking.py / 深度Agent 一致）。

    test_feat.npy 本身就是 @20；val_feat.npy 是 @100（特征侧消融用），
    因此 val 单独提取 @20 并缓存为 val20_feat.npy。
    归一化用 norm_stats.json（train@100 统计量，与 LightGBM Agent 推理一致）。
    """
    if split == "test":
        return np.load(os.path.join(FEATURE_DIR, "test_feat.npy"))
    path = os.path.join(FEATURE_DIR, "val20_feat.npy")
    if os.path.exists(path):
        return np.load(path)
    print(f"    提取 val@20 特征（一次性，缓存到 {path}）...")
    from data.epoch_cache import EpochCache, load_norm_stats
    from training.common import extract_multi_domain_features
    cache = EpochCache()
    mean, std = load_norm_stats()
    X, _, _, _ = cache.load_split("val", max_epochs_per_record=20,
                                   normalize="per_channel", norm_stats=(mean, std))
    feats, _ = extract_multi_domain_features(X, n_jobs=8, include_event=True)
    np.save(path, feats.astype(np.float32))
    return feats


def lgbm_probs_from_features(split):
    """lightgbm(532) 与 lightgbm_event(614) 的 epoch 概率，直接从预提取特征计算"""
    feat = _load_or_extract_feat20(split)
    out = {}

    p532 = resolve_path(BINARY_MODEL_PATHS["lightgbm"])
    if os.path.exists(p532):
        fusion = joblib.load(BINARY_FUSION_PATH)
        model = joblib.load(p532)
        out["lightgbm"] = model.predict_proba(fusion.transform(feat[:, :532]))[:, 1]

    if os.path.exists(BINARY_EVENT_MODEL_PATH):
        fusion_ev = joblib.load(BINARY_EVENT_FUSION_PATH)
        model_ev = joblib.load(BINARY_EVENT_MODEL_PATH)
        out["lightgbm_event"] = model_ev.predict_proba(fusion_ev.transform(feat))[:, 1]
    return out


def deep_agent_probs(split):
    """三个深度 Agent 的 epoch 概率（原始 µV 输入，Agent 内部归一化）"""
    cache = EpochCache()
    X, y3, sids, _ = cache.load_split(split, max_epochs_per_record=20, normalize=None)
    out = {}
    for col, module, cls_name, ckpt in DEEP_AGENTS:
        mod = __import__(module, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        agent = cls(col, model_path=resolve_path(BINARY_MODEL_PATHS[ckpt]))
        t0 = time.time()
        probs = np.zeros(len(X))
        for i in range(len(X)):
            r = agent.predict(X[i])
            p = np.asarray(r["prob"], dtype=np.float64)
            probs[i] = p[1] if len(p) == 2 else p[1] + p[2]   # P(Abnormal)
        out[col] = probs
        print(f"    [{col}] {len(X)} epochs, {time.time()-t0:.0f}s")
    # 三分类标签与被试ID随缓存一并保存
    out["_y3"] = y3
    out["_sids"] = sids
    return out


def collect_probs(force=False):
    """收集并缓存 val/test 的全部 Agent 概率"""
    all_probs = {}
    for split in ("val", "test"):
        path = os.path.join(FEATURE_DIR, f"agent_probs_{split}.npz")
        if os.path.exists(path) and not force:
            print(f"  [{split}] 复用缓存: {path}")
            z = np.load(path, allow_pickle=True)
            all_probs[split] = {k: z[k] for k in z.files}
            continue
        print(f"  [{split}] 收集 Agent 概率...")
        t0 = time.time()
        probs = lgbm_probs_from_features(split)
        deep = deep_agent_probs(split)
        n_ref = len(deep["_sids"])
        for k, v in probs.items():
            if len(v) != n_ref:
                raise RuntimeError(
                    f"[{split}] Agent '{k}' 行数 {len(v)} 与深度Agent口径 {n_ref} 不一致，"
                    f"检查特征文件是否为 @20 epochs/record")
        probs.update(deep)
        np.savez(path, **probs)
        all_probs[split] = probs
        print(f"  [{split}] 完成 ({time.time()-t0:.0f}s) -> {path}")
    return all_probs


# ------------------------------------------------------------------
# 第二步：被试级融合变体
# ------------------------------------------------------------------
def subject_meta(probs_dict, agent_cols, method="p90"):
    """epoch 概率 → 被试级 (n_subjects, n_agents) 元特征 + 标签"""
    sids = probs_dict["_sids"]
    uniq = sorted(set(sids))
    P = np.column_stack([probs_dict[c] for c in agent_cols])
    if method == "mean":
        M = np.vstack([P[sids == u].mean(axis=0) for u in uniq])
    else:
        M = np.vstack([np.percentile(P[sids == u], 90, axis=0) for u in uniq])
    y3 = np.array([probs_dict["_y3"][sids == u][0] for u in uniq])
    y_bin = np.array([BINARY_LABEL_MAP[int(v)] for v in y3])
    return M, y_bin, uniq


def eval_pred(y_true, p_abn):
    pred = (p_abn >= 0.5).astype(int)
    return {
        "acc": round(float(accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "auc": round(float(roc_auc_score(y_true, p_abn)), 4),
    }


def stacking_variant(val_M, val_y, test_M, test_y):
    """在 val 上训练 LightGBM 元模型，test 上评估（与 train_stacking.py 同超参）"""
    clf = lgb.LGBMClassifier(
        n_estimators=50, learning_rate=0.1, num_leaves=8,
        objective="binary", random_state=42, verbose=-1, class_weight="balanced")
    clf.fit(val_M, val_y)
    p = clf.predict_proba(test_M)[:, 1]
    return eval_pred(test_y, p)


def hard_voting(test_M, test_y):
    """被试级多数票（各 Agent 二值预测，平票判 Normal）"""
    votes = (test_M >= 0.5).astype(int).sum(axis=1)     # 判 Abnormal 的票数
    pred = (votes * 2 > test_M.shape[1]).astype(int)    # 严格过半才判 Abnormal
    n_agents = test_M.shape[1]
    return {
        "acc": round(float(accuracy_score(test_y, pred)), 4),
        "macro_f1": round(float(f1_score(test_y, pred, average="macro")), 4),
        "auc": None,   # 硬投票无连续分数，不给 AUC
        "note": f"{n_agents} agents majority vote, tie→Normal",
    }


def main(force=False, agg="p90"):
    print("=" * 74)
    print("消融实验 — Agent 侧（A4~A6, A9）+ 硬投票基线")
    print("=" * 74)

    probs = collect_probs(force=force)
    agent_cols = [c for c in probs["test"].keys() if not c.startswith("_")]
    print(f"\n可用 Agent: {agent_cols}")

    val_M, val_y, _ = subject_meta(probs["val"], agent_cols, agg)
    test_M, test_y, _ = subject_meta(probs["test"], agent_cols, agg)
    base_rate = max(np.bincount(test_y)) / len(test_y)

    # 两套基底：lightgbm（原）与 lightgbm_event（若存在）
    bases = []
    if "lightgbm" in agent_cols:
        bases.append(("lightgbm", ["lightgbm", "eegnet", "tcn", "transformer"]))
    if "lightgbm_event" in agent_cols:
        bases.append(("lightgbm_event", ["lightgbm_event", "eegnet", "tcn", "transformer"]))

    results = {
        "protocol": f"subject-level binary, meta=LGBM on val, agg={agg}, test@20",
        "n_test_subjects": int(len(test_y)),
        "baseline_majority": round(float(base_rate), 4),
        "bases": {},
    }

    for base_name, cols in bases:
        print(f"\n--- 基底: {base_name} ({cols}) ---")
        vM, vy, _ = subject_meta(probs["val"], cols, agg)
        tM, ty, _ = subject_meta(probs["test"], cols, agg)

        variants = {
            "stacking_full": stacking_variant(vM, vy, tM, ty),
            "A4_no_eegnet": stacking_variant(vM[:, [0, 2, 3]], vy, tM[:, [0, 2, 3]], ty),
            "A5_no_tcn": stacking_variant(vM[:, [0, 1, 3]], vy, tM[:, [0, 1, 3]], ty),
            "A6_no_transformer": stacking_variant(vM[:, [0, 1, 2]], vy, tM[:, [0, 1, 2]], ty),
            "A9_soft_voting_equal": eval_pred(ty, tM.mean(axis=1)),
            "hard_voting": hard_voting(tM, ty),
        }
        # 各单 Agent（被试级）对照
        for i, c in enumerate(cols):
            variants[f"single_{c}"] = eval_pred(ty, tM[:, i])

        results["bases"][base_name] = {"agents": cols, "variants": variants}
        print(f"  {'变体':<22s} {'Acc':>7s} {'AUC':>7s} {'F1':>7s}")
        for k, v in variants.items():
            auc = f"{v['auc']:.4f}" if v["auc"] is not None else "  —"
            print(f"  {k:<22s} {v['acc']:>7.4f} {auc:>7s} {v['macro_f1']:>7.4f}")

    out_path = resolve_path("outputs/ablation_agents.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(results, out_path)
    print(f"\n结果已保存: {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cache", action="store_true", help="复用已缓存概率")
    ap.add_argument("--agg", default="p90", help="被试级聚合: mean/p90")
    a = ap.parse_args()
    main(force=not a.skip_cache, agg=a.agg)
