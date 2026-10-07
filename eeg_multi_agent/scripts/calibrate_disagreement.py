# scripts/calibrate_disagreement.py
"""
分歧检测阈值校准

问题：当前 DISAGREEMENT_CONFIG（prob_diff=0.15 / majority=0.6 / entropy=0.8）
在 val/test 上触发率 100%，仲裁退化为"永远执行"，失去选择性且推理耗时翻倍。

方法（在 val@20 上校准，避免动 test）：
    1. 逐 epoch 走 Orchestrator 第一轮（质量→动态权重→特征→四Agent），
       缓存三个分歧指标：max_prob_diff / majority_ratio / entropy
    2. 对每个 epoch 无条件跑一次仲裁，缓存两轮投票概率 →
       任意阈值组合可零成本模拟最终结果（触发→仲裁后概率，否则→一轮概率）
    3. 宽范围网格扫描 + 帕累托前沿（触发率 vs 被试精度）+ 推荐配置

已知分布事实（val@20, 4800 epochs）：
    entropy 中位数 1.0878（≈ln3 上限 1.0986）→ 熵规则阈值需 ≥1.09 才有选择性
    max_prob_diff 中位数 0.178 → 0.15 阈值触发 67%
    majority_ratio ∈ {0.5, 0.75, 1.0}，0.5（标签分裂）占 32.6%

输出:
    outputs/disagreement_calibration.json
    E:/eeg_asd_cache/features/disagreement_val.npz  逐epoch原始指标

用法:
    python scripts/calibrate_disagreement.py                 # 全量（~12分钟）
    python scripts/calibrate_disagreement.py --from-cache    # 复用 npz 离线分析
"""
import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import accuracy_score

from config import resolve_path
from utils.io_utils import ensure_dir, save_json

FEATURE_DIR = os.path.join("E:/eeg_asd_cache", "features")
CACHE_PATH = os.path.join(FEATURE_DIR, "disagreement_val.npz")


def epoch_metrics(first_results):
    """与 orchestrator.disagreement.check_disagreement 完全一致的三个指标"""
    probs = np.array([r["prob"] for r in first_results], dtype=np.float64)
    labels = [r["label"] for r in first_results]
    max_prob_diff = float(np.max(probs.max(axis=0) - probs.min(axis=0)))
    majority_ratio = Counter(labels).most_common(1)[0][1] / len(labels)
    avg_prob = probs.mean(axis=0)
    entropy = float(-np.sum(avg_prob * np.log(avg_prob + 1e-10)))
    return max_prob_diff, majority_ratio, entropy


def collect(n_jobs_note=""):
    """跑 val@20 第一轮 + 无条件仲裁，缓存指标与两轮概率"""
    from data.epoch_cache import EpochCache
    from orchestrator.orchestrator import Orchestrator
    from orchestrator.voting import weighted_vote
    from orchestrator.arbitration import arbitrate

    cache = EpochCache()
    X, y, sids, _ = cache.load_split("val", max_epochs_per_record=20,
                                     normalize=None)
    n = len(X)
    uniq = sorted(set(sids))
    sid_idx = np.searchsorted(np.array(uniq), sids)
    y_sub = np.array([y[sids == u][0] for u in uniq])
    print(f"val: {n} epochs, {len(uniq)} 被试")

    print("\n初始化 Orchestrator...")
    system = Orchestrator(device="cpu")

    print(f"逐 epoch 诊断（{n} 个，含无条件仲裁）...")
    t0 = time.time()
    m_diff = np.zeros(n); m_maj = np.zeros(n); m_ent = np.zeros(n)
    p_first = np.zeros((n, 3)); p_arb = np.zeros((n, 3))
    for i in range(n):
        qr = system._assess_quality(X[i])
        dyn_w = system._adjust_weights(qr)
        feats = system._extract_features(X[i])
        first = system._run_diagnosis(X[i], feats)
        m_diff[i], m_maj[i], m_ent[i] = epoch_metrics(first)
        p_first[i] = weighted_vote(first, dyn_w)["prob"]
        arb_res, arb_info = arbitrate(
            X[i], first, feats,
            diagnosis_agents=system.diagnosis_agents, weights=dyn_w,
            sfreq=system.sfreq, snr_db=qr["snr_db"],
            quality_level=qr["quality_level"])
        p_arb[i] = weighted_vote(arb_res, arb_info["weights"])["prob"]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")
    print(f"完成，耗时 {time.time()-t0:.0f}s")

    np.savez(CACHE_PATH, max_prob_diff=m_diff, majority_ratio=m_maj,
             entropy=m_ent, p_first=p_first, p_arb=p_arb, y=y, sid_idx=sid_idx)
    return m_diff, m_maj, m_ent, p_first, p_arb, y, sid_idx


def main(from_cache=False):
    print("=" * 74)
    print("分歧检测阈值校准（val@20，Orchestrator 三分类口径）")
    print("=" * 74)

    if from_cache and os.path.exists(CACHE_PATH):
        print(f"复用缓存: {CACHE_PATH}")
        z = np.load(CACHE_PATH)
        m_diff, m_maj, m_ent = z["max_prob_diff"], z["majority_ratio"], z["entropy"]
        p_first, p_arb = z["p_first"], z["p_arb"]
        y, sid_idx = z["y"], z["sid_idx"]
        uniq_n = int(sid_idx.max()) + 1
        y_sub = np.array([y[sid_idx == u][0] for u in range(uniq_n)])
    else:
        m_diff, m_maj, m_ent, p_first, p_arb, y, sid_idx = collect()
        uniq_n = int(sid_idx.max()) + 1
        y_sub = np.array([y[sid_idx == u][0] for u in range(uniq_n)])
    n = len(y)

    # ---- 分布统计 ----
    def q(a): return {p: round(float(np.quantile(a, p)), 4)
                      for p in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    dist = {"max_prob_diff": q(m_diff), "majority_ratio": q(m_maj),
            "entropy": q(m_ent)}

    # ---- 当前配置的规则触发率 ----
    cur = {"prob_diff_threshold": 0.15, "label_majority_threshold": 0.6,
           "entropy_threshold": 0.8}
    r1 = m_diff > cur["prob_diff_threshold"]
    r2 = m_maj < cur["label_majority_threshold"]
    r3 = m_ent > cur["entropy_threshold"]
    current_fire = {
        "rule1_prob_diff": round(float(r1.mean()), 4),
        "rule2_majority": round(float(r2.mean()), 4),
        "rule3_entropy": round(float(r3.mean()), 4),
        "combined": round(float((r1 | r2 | r3).mean()), 4),
    }

    # ---- 两个极端基线 ----
    def subject_acc(probs):
        counts = np.bincount(sid_idx, minlength=uniq_n)
        p_sub = np.zeros((uniq_n, 3))
        for c in range(3):
            p_sub[:, c] = np.bincount(sid_idx, weights=probs[:, c],
                                      minlength=uniq_n) / counts
        return float(accuracy_score(y_sub, p_sub.argmax(axis=1)))

    base_never = subject_acc(p_first)
    base_always = subject_acc(p_arb)

    # ---- 宽范围网格扫描 ----
    # t2: 0.5=规则关闭（majority≥0.5 恒成立）；0.6=标签分裂(2-2/2-1-1)时触发
    # t3: 熵上限 ln3≈1.0986，网格须覆盖到 1.098 才能关掉该规则
    t1_grid = np.round(np.arange(0.15, 1.001, 0.05), 2)
    t3_grid = np.round(np.arange(0.80, 1.101, 0.01), 3)   # 上限 1.10 > ln3≈1.0986，可完全关闭熵规则
    grid = []
    for t2 in (0.5, 0.6):
        r2b = m_maj < t2
        for t1 in t1_grid:
            r1b = m_diff > t1
            for t3 in t3_grid:
                trig = r1b | r2b | (m_ent > t3)
                rate = float(trig.mean())
                if rate > 0.50:      # 触发过半的组合无选择性，不入网格
                    continue
                probs = np.where(trig[:, None], p_arb, p_first)
                grid.append({
                    "prob_diff_threshold": float(t1),
                    "label_majority_threshold": float(t2),
                    "entropy_threshold": float(t3),
                    "trigger_rate": round(rate, 4),
                    "epoch_acc": round(float((probs.argmax(1) == y).mean()), 4),
                    "subject_acc": round(subject_acc(probs), 4),
                })

    if grid:
        grid.sort(key=lambda g: (-g["subject_acc"], g["trigger_rate"]))
        # 推荐配置：被试精度并列（≤0.005）中触发率最低
        best_acc = grid[0]["subject_acc"]
        tied = [g for g in grid if g["subject_acc"] >= best_acc - 0.005]
        recommend = min(tied, key=lambda g: g["trigger_rate"])
    else:
        grid = []
        recommend = None

    # ---- 帕累托前沿：触发率升序，被试精度单调不降的组合 ----
    pareto = []
    best_so_far = -1.0
    for g in sorted(grid, key=lambda g: g["trigger_rate"]):
        if g["subject_acc"] > best_so_far:
            pareto.append(g)
            best_so_far = g["subject_acc"]

    results = {
        "protocol": "val@20, 3-class Orchestrator, 4 diagnosis agents",
        "n_epochs": int(n), "n_subjects": int(uniq_n),
        "metric_distributions": dist,
        "current_config_firing": current_fire,
        "baselines": {
            "never_arbitrate_subject_acc": round(base_never, 4),
            "always_arbitrate_subject_acc": round(base_always, 4),
        },
        "grid_top10": grid[:10],
        "pareto": pareto,
        "recommend": recommend,
    }

    print("\n---- 指标分布（分位数）----")
    for k, v in dist.items():
        print(f"  {k}: {v}")
    print("\n---- 当前配置规则触发率 ----")
    for k, v in current_fire.items():
        print(f"  {k}: {v:.1%}")
    print(f"\n基线: 从不仲裁 subject_acc={base_never:.4f} | "
          f"总是仲裁 subject_acc={base_always:.4f}")
    if pareto:
        print("\n---- 帕累托前沿（触发率 → 被试精度）----")
        for g in pareto:
            print(f"  触发 {g['trigger_rate']:>5.1%}  subject {g['subject_acc']:.4f}"
                  f"  (diff>{g['prob_diff_threshold']:.2f} "
                  f"maj<{g['label_majority_threshold']:.1f} "
                  f"ent>{g['entropy_threshold']:.3f})")
    print(f"\n推荐配置: {recommend}")

    out_path = resolve_path("outputs/disagreement_calibration.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(results, out_path)
    print(f"\n结果已保存: {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true",
                    help="复用 disagreement_val.npz 离线分析")
    a = ap.parse_args()
    main(from_cache=a.from_cache)
