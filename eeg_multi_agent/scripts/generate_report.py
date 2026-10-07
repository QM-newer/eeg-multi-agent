# scripts/generate_report.py
"""
单病例自动报告生成（阶段 6 收尾：文档输出环节）

流程：
    1. 从 epoch 缓存取一名被试的全部 epoch（原始 µV，默认取 test 集被试）
    2. 逐 epoch 走完整多Agent诊断（Orchestrator：质量评估→动态权重→四Agent→
       加权投票→分歧仲裁→一致性/可解释性）
    3. 被试级聚合 → 生成结构化 Markdown 病例报告（含证据链与共识过程统计）

输出:
    outputs/reports/<subject_id>_report.md

用法:
    python scripts/generate_report.py                       # 自动挑一个 test 被试
    python scripts/generate_report.py --subject SUBJ_1234
    python scripts/generate_report.py --split val
"""
import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import CLASS_NAMES, DATA_CONFIG, resolve_path
from data.epoch_cache import EpochCache
from orchestrator.orchestrator import Orchestrator
from utils.io_utils import ensure_dir

BORDERLINE_MARGIN = 0.10   # 被试级首两类概率差 < 该值 → 输出"可疑/界限性倾向"


def pick_subject(cache, split, subject=None, max_epochs=20):
    """返回 (subject_id, X(n,C,T), y3)"""
    X, y, sids, _ = cache.load_split(split, max_epochs_per_record=max_epochs,
                                      normalize=None)
    if subject is None:
        # 优先挑一个 Abnormal 的被试（报告更有代表性），否则第一个
        uniq = sorted(set(sids))
        for u in uniq:
            if y[sids == u][0] == 2:
                subject = u
                break
        else:
            subject = uniq[0]
    mask = sids == subject
    if not mask.any():
        raise SystemExit(f"被试 {subject} 不在 {split} 集中。可用示例: "
                         f"{sorted(set(sids))[:5]} ...")
    return subject, X[mask], y[mask][0]


def subject_label_name(p_mean):
    """被试级结论：首两类概率差过小 → 可疑（对应临床 Borderline 档）"""
    order = np.argsort(p_mean)[::-1]
    top, second = order[0], order[1]
    if p_mean[top] - p_mean[second] < BORDERLINE_MARGIN:
        return (f"可疑/界限性倾向（{CLASS_NAMES[top]} {p_mean[top]:.1%} vs "
                f"{CLASS_NAMES[second]} {p_mean[second]:.1%}）")
    return CLASS_NAMES[top]


def main(split="test", subject=None, max_epochs=20):
    print("=" * 74)
    print("单病例报告生成")
    print("=" * 74)

    cache = EpochCache()
    subject, X, y_true = pick_subject(cache, split, subject, max_epochs)
    n_epochs = len(X)
    print(f"被试: {subject}  epoch 数: {n_epochs}  临床标签: {CLASS_NAMES[y_true]}")

    print("初始化多Agent系统...")
    system = Orchestrator(device="cpu")

    # ---- 逐 epoch 完整诊断 ----
    print(f"逐 epoch 诊断（{n_epochs} 个）...")
    t0 = time.time()
    probs, weights_acc, disagreements, arbitrations = [], [], 0, 0
    consistency_scores, top_feature_counter, quality_levels = [], Counter(), Counter()
    agent_votes = Counter()
    sample_detail = None
    for i in range(n_epochs):
        r = system.diagnose(X[i])
        probs.append(r["final_prob"])
        weights_acc.append(r["weights"])
        disagreements += int(r["disagreement"])
        arbitrations += int(r["second_opinion_triggered"])
        consistency_scores.append(r["consistency"]["consistency_score"])
        quality_levels[r["quality_report"]["quality_level"]] += 1
        for f in r["top_features"][:5]:
            top_feature_counter[int(f)] += 1
        for a in r["agent_results"]:
            agent_votes[(a["agent_name"], CLASS_NAMES[a["label"]])] += 1
        if r["disagreement"] and sample_detail is None:
            sample_detail = r        # 留一份分歧样本的详细过程
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_epochs}")
    elapsed = time.time() - t0

    p_mean = np.mean(probs, axis=0)
    p_std = np.std(probs, axis=0)
    mean_weights = {k: float(np.mean([w[k] for w in weights_acc]))
                    for k in weights_acc[0]}
    final_name = subject_label_name(p_mean)
    cons_mean = float(np.mean(consistency_scores))

    # ---- 生成 Markdown ----
    lines = []
    A = lines.append
    A(f"# 脑电多智能体辅助诊断报告")
    A("")
    A(f"- 被试编号：`{subject}`")
    A(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A(f"- 分析样本：{n_epochs} 个 epoch（{DATA_CONFIG['epoch_length']}s / "
      f"{DATA_CONFIG['target_sfreq']}Hz / {DATA_CONFIG['n_eeg_channels']} 通道，"
      f"每记录最多取 {max_epochs} 个）")
    A(f"- 临床参考标签：**{CLASS_NAMES[y_true]}**（医生判读，仅供对照）")
    A("")
    A("## 一、结论")
    A("")
    A(f"**系统诊断：{final_name}**")
    A("")
    A("| 类别 | 平均概率 | 跨epoch波动(std) |")
    A("|---|---|---|")
    for i, name in enumerate(CLASS_NAMES):
        A(f"| {name} | {p_mean[i]:.1%} | {p_std[i]:.3f} |")
    A("")
    A("## 二、信号质量")
    A("")
    q = "、".join(f"{lvl} {cnt} 个" for lvl, cnt in sorted(quality_levels.items()))
    A(f"- epoch 质量等级分布：{q}")
    A(f"- 质量等级由 SNR 与伪迹比例自动评定（utils/signal_utils），"
      f"并驱动各 Agent 动态权重。")
    A("")
    A("## 三、多Agent意见与共识过程")
    A("")
    A("| 诊断Agent | 平均动态权重 | 主要意见（epoch票数） |")
    A("|---|---|---|")
    for ag in mean_weights:
        votes = [f"{nm} {c}" for (a, nm), c in agent_votes.items() if a == ag]
        A(f"| {ag} | {mean_weights[ag]:.3f} | {'、'.join(votes) if votes else '—'} |")
    A("")
    A(f"- 分歧触发率：{disagreements}/{n_epochs}"
      f"（{disagreements/n_epochs:.0%}，触发即进入补充特征→重推理→可信度仲裁流程）")
    A(f"- 仲裁执行率：{arbitrations}/{n_epochs}")
    A(f"- 跨Agent一致性得分：{cons_mean:.3f}"
      f"（{'建议人工复核' if cons_mean < 0.6 else '一致性良好'}）")
    A("")
    if sample_detail is not None:
        arb = sample_detail["arbitration"]
        A("分歧仲裁示例（首个分歧 epoch）：")
        A("")
        if arb:
            A(f"- 补充空间域特征 {arb['spatial_dim']} 维（扩充后 {arb['augmented_dim']} 维）"
              f"后二次诊断；特征可信度："
              + "、".join(f"{k}={v:.2f}" for k, v in arb["credibility"].items()))
        A("")
    A("## 四、证据链（可解释性）")
    A("")
    A("各 epoch 可解释性 Agent 输出的高频 Top 特征（特征索引 @ 532 维多域特征）：")
    A("")
    A("| 特征索引 | 出现次数 |")
    A("|---|---|")
    for fidx, cnt in top_feature_counter.most_common(8):
        A(f"| {fidx} | {cnt} |")
    A("")
    A("> 特征索引按 [时域 0-227 | 频域 228-379 | 时频 380-531] 分块，"
      "可对照 `training/common.py::_make_feature_agents` 定位特征含义。")
    A("")
    A("## 五、说明与免责")
    A("")
    A("- 本报告由 EEG-MAS 多智能体系统自动生成，仅供研究/辅助参考，"
      "不构成临床诊断结论。")
    A(f"- 单被试推理耗时 {elapsed:.1f}s（{n_epochs} epoch，CPU）。")
    A("")

    out_dir = resolve_path("outputs/reports")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{subject}_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已生成: {out_path}")
    print(f"结论: {final_name}  (临床标签: {CLASS_NAMES[y_true]})")
    print(f"概率: " + "  ".join(f"{n}={p:.1%}" for n, p in zip(CLASS_NAMES, p_mean)))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="从哪个 split 挑被试")
    ap.add_argument("--subject", default=None, help="被试ID，如 SUBJ_1234")
    ap.add_argument("--max-epochs", type=int, default=20)
    a = ap.parse_args()
    main(split=a.split, subject=a.subject, max_epochs=a.max_epochs)
