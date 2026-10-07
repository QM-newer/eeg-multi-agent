# scripts/run_ablation_system.py
"""
消融实验 — 系统机制侧（论文计划 5.4 A7/A8/A9，三分类完整系统口径）

对 test 集逐 epoch 走 Orchestrator 流程并缓存中间产物
（质量报告、一轮结果、动态权重、仲裁结果与可信度），随后零成本重组出：

    full            完整系统：SNR动态权重 + 分歧仲裁（忠实复现 diagnose()）
    A7_no_arbitration  去掉仲裁：动态权重 + 一轮结果直接投票
    A8_fixed_weights   去掉动态权重：静态权重（agent_weights.json）贯穿投票与仲裁
    A9_equal_weights   简单平均：全 Agent 等权（投票与仲裁基底均等权）

同时报告论文要求的协作机制指标：分歧触发率、仲裁改善率。

输出:
    outputs/ablation_system.json

用法:
    python scripts/run_ablation_system.py
    python scripts/run_ablation_system.py --max-epochs-per-record 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import CLASS_NAMES, ORCHESTRATOR_CONFIG, resolve_path
from data.epoch_cache import EpochCache
from orchestrator.orchestrator import Orchestrator
from orchestrator.voting import weighted_vote
from orchestrator.disagreement import check_disagreement
from orchestrator.dynamic_weight import adjust_weights_by_snr
from orchestrator.arbitration import arbitrate
from utils.io_utils import ensure_dir, save_json


def run_epoch(system, x, cfg_dis, arb_enable):
    """单 epoch 完整流程，返回全部中间产物"""
    qr = system._assess_quality(x)
    dyn_w = system._adjust_weights(qr)
    feats = system._extract_features(x)
    first = system._run_diagnosis(x, feats)
    disagree = check_disagreement(first, cfg_dis)
    arb_results, arb_info = None, None
    if disagree and arb_enable:
        arb_results, arb_info = arbitrate(
            x, first, feats,
            diagnosis_agents=system.diagnosis_agents,
            weights=dyn_w,
            sfreq=system.sfreq,
            snr_db=qr["snr_db"],
            quality_level=qr["quality_level"],
        )
    return {"quality": qr, "dyn_weights": dyn_w, "first": first,
            "disagree": disagree, "arb_results": arb_results, "arb_info": arb_info}


def recombine(cached, mode, static_weights):
    """按消融模式重组一轮流程，返回 (label, prob)"""
    names = [r["agent_name"] for r in cached["first"]]
    if mode == "A8_fixed_weights":
        base_w = {n: static_weights.get(n, 1.0 / len(names)) for n in names}
    elif mode == "A9_equal_weights":
        base_w = {n: 1.0 / len(names) for n in names}
    else:
        base_w = cached["dyn_weights"]

    if mode == "A7_no_arbitration" or cached["arb_results"] is None:
        results, weights = cached["first"], base_w
    else:
        # 仲裁路径：A8/A9 需以各自基底权重重算仲裁权重（可信度已缓存，无需重推理）
        if mode in ("A8_fixed_weights", "A9_equal_weights"):
            from orchestrator.arbitration import _arbitration_weights
            arb_w = _arbitration_weights(
                base_w, cached["arb_info"]["credibility"], names)
        else:
            arb_w = cached["arb_info"]["weights"]
        results, weights = cached["arb_results"], arb_w

    vote = weighted_vote(results, weights)
    return vote["label"], np.asarray(vote["prob"])


def metrics_block(y_true_ep, probs_ep, sids):
    """epoch 级 + 被试级（mean 聚合）三分类指标"""
    pred_ep = np.argmax(probs_ep, axis=1)
    uniq = sorted(set(sids))
    p_sub = np.vstack([probs_ep[sids == u].mean(axis=0) for u in uniq])
    y_sub = np.array([y_true_ep[sids == u][0] for u in uniq])
    pred_sub = np.argmax(p_sub, axis=1)
    return {
        "epoch_acc": round(float(accuracy_score(y_true_ep, pred_ep)), 4),
        "epoch_macro_f1": round(float(f1_score(y_true_ep, pred_ep, average="macro")), 4),
        "subject_acc": round(float(accuracy_score(y_sub, pred_sub)), 4),
        "subject_macro_f1": round(float(f1_score(y_sub, pred_sub, average="macro")), 4),
    }


def main(max_epochs_per_record=20):
    print("=" * 74)
    print("消融实验 — 系统机制（A7 仲裁 / A8 动态权重 / A9 等权投票）")
    print("=" * 74)

    cache = EpochCache()
    X, y, sids, _ = cache.load_split("test", max_epochs_per_record=max_epochs_per_record,
                                      normalize=None)
    n = len(X)
    print(f"test: {n} epochs, {len(set(sids))} 被试")

    print("\n初始化 Orchestrator...")
    system = Orchestrator(device="cpu")
    static_weights = dict(system.weights)
    print(f"静态权重(来自 agent_weights.json): "
          + ", ".join(f"{k}={v:.3f}" for k, v in static_weights.items()))

    arb_enable = bool(ORCHESTRATOR_CONFIG["arbitration"]["enable"])
    print(f"仲裁开关: {'开' if arb_enable else '关'}")

    # ---- 逐 epoch 跑完整流程并缓存 ----
    print(f"\n逐 epoch 诊断并缓存中间产物（{n} 个）...")
    t0 = time.time()
    all_cached = []
    for i in range(n):
        all_cached.append(run_epoch(system, X[i], system.disagreement_config, arb_enable))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")
    print(f"完成，耗时 {time.time()-t0:.0f}s")

    # ---- 机制指标 ----
    n_disagree = sum(c["disagree"] for c in all_cached)
    dis_idx = [i for i, c in enumerate(all_cached) if c["disagree"] and c["arb_results"]]
    # 仲裁改善率：分歧样本中，仲裁后判对 vs 仲裁前（一轮加权投票）判对
    correct_before = correct_after = 0
    for i in dis_idx:
        c = all_cached[i]
        before = weighted_vote(c["first"], c["dyn_weights"])["label"]
        after = weighted_vote(c["arb_results"], c["arb_info"]["weights"])["label"]
        correct_before += int(before == y[i])
        correct_after += int(after == y[i])
    mech = {
        "disagreement_rate": round(n_disagree / n, 4),
        "arbitration_improvement": {
            "n_disagreement_arbitrated": len(dis_idx),
            "acc_before_arbitration": round(correct_before / max(len(dis_idx), 1), 4),
            "acc_after_arbitration": round(correct_after / max(len(dis_idx), 1), 4),
        },
    }

    # ---- 消融变体 ----
    results = {
        "protocol": f"3-class full system, test@{max_epochs_per_record}, "
                    "subject=mean epoch prob",
        "n_epochs": int(n),
        "n_subjects": int(len(set(sids))),
        "baseline_majority": round(float(max(np.bincount(y)) / len(y)), 4),
        "mechanism_metrics": mech,
        "variants": {},
    }

    subject_correct = {}          # 各变体的被试级判定正确性（McNemar 配对用）
    epoch_correct = {}            # 各变体的 epoch 级正确性（质量分层用）
    for mode in ("full", "A7_no_arbitration", "A8_fixed_weights", "A9_equal_weights"):
        probs = np.zeros((n, len(CLASS_NAMES)))
        for i, c in enumerate(all_cached):
            label, p = recombine(c, mode, static_weights)
            probs[i] = p
        results["variants"][mode] = metrics_block(y, probs, sids)
        m = results["variants"][mode]
        print(f"  {mode:<22s} epoch_acc={m['epoch_acc']:.4f}  "
              f"subject_acc={m['subject_acc']:.4f}  "
              f"subject_F1={m['subject_macro_f1']:.4f}")
        epoch_correct[mode] = probs.argmax(axis=1) == y
        # 被试级正确性向量（与 metrics_block 同口径：mean 聚合后 argmax）
        uniq = sorted(set(sids))
        p_sub = np.vstack([probs[sids == u].mean(axis=0) for u in uniq])
        y_sub = np.array([y[sids == u][0] for u in uniq])
        subject_correct[mode] = p_sub.argmax(axis=1) == y_sub

    # ---- McNemar 检验：各消融变体 vs full（被试级配对）----
    # b = full 对而变体错的被试数；c = full 错而变体对的被试数
    # 精确 McNemar = 二项检验 Bin(b+c, 0.5) 双侧
    from scipy.stats import binomtest
    mcnemar = {}
    for mode, corr in subject_correct.items():
        if mode == "full":
            continue
        ref = subject_correct["full"]
        b = int(np.sum(ref & ~corr))
        c = int(np.sum(~ref & corr))
        if b + c == 0:
            p_val = 1.0
        else:
            p_val = float(binomtest(min(b, c), b + c, 0.5).pvalue)
        mcnemar[mode] = {
            "full_correct_variant_wrong": b,
            "full_wrong_variant_correct": c,
            "p_value": round(p_val, 4),
            "significant_at_0.05": bool(p_val < 0.05),
        }
        print(f"  McNemar {mode:<22s} b={b} c={c}  p={p_val:.4f}"
              f"{' *' if p_val < 0.05 else ''}")
    results["mcnemar_vs_full"] = mcnemar

    # ---- 信号质量分层分析：动态权重的价值应在低质量子集显现 ----
    q_levels = np.array([c["quality"]["quality_level"] for c in all_cached])
    strat = {"epoch_level": {}, "subject_level": {}}
    print("\n质量分层（epoch 级）:")
    for lvl in ("high", "medium", "low"):
        mask = q_levels == lvl
        if not mask.any():
            continue
        strat["epoch_level"][lvl] = {
            "n_epochs": int(mask.sum()),
            **{m_: round(float(ec[mask].mean()), 4)
               for m_, ec in epoch_correct.items()},
        }
        row = strat["epoch_level"][lvl]
        print(f"  {lvl:<7s} n={mask.sum():>5d}  "
              + "  ".join(f"{m_}={row[m_]:.4f}" for m_ in epoch_correct))
    # 被试级：含中/低质量 epoch 的被试 vs 全高质量
    uniq = sorted(set(sids))
    low_mask = np.isin(q_levels, ["medium", "low"])
    has_low = np.array([bool(low_mask[sids == u].any()) for u in uniq])
    for grp, mask_sub in (("all_high_quality", ~has_low),
                          ("contains_medium_low", has_low)):
        if not mask_sub.any():
            continue
        strat["subject_level"][grp] = {
            "n_subjects": int(mask_sub.sum()),
            **{m_: round(float(sc[mask_sub].mean()), 4)
               for m_, sc in subject_correct.items()},
        }
        row = strat["subject_level"][grp]
        print(f"被试级 {grp:<20s} n={mask_sub.sum():>4d}  "
              + "  ".join(f"{m_}={row[m_]:.4f}" for m_ in subject_correct))
    results["quality_stratified"] = strat

    out_path = resolve_path("outputs/ablation_system.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(results, out_path)
    print(f"\n结果已保存: {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs-per-record", type=int, default=20)
    a = ap.parse_args()
    main(max_epochs_per_record=a.max_epochs_per_record)
