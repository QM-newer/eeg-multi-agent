# scripts/analyze_label_metadata.py
"""
标签可行性分析：把"医生报告里的结构化描述"与三分类标签对齐

为什么需要：
    模型当前 AUC 只有 0.63，判断不了是"信号本身弱"还是"信息没被利用"。
    病例表里其实带了报告级描述（slow_wave / sharp_wave / spike_wave /
    epileptiform_count / 脑区定位 / age），把它们与标签对齐即可回答两件事：
      1) 标签在临床上是否自洽（Abnormal 是否 ≈ 报告里写了慢波/放电）
      2) 这些描述能否作为"中间监督"（辅助任务）来提升主任务

注意：这些字段属于报告级信息，**不能作为推理特征**（等于标签泄漏），
只能用于辅助监督目标、误差分析与可行性评估。

用法：
    python scripts/analyze_label_metadata.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import resolve_path
from utils.io_utils import ensure_dir, save_json

WAVE_COLS = ["slow_wave", "sharp_wave", "spike_wave", "spindle_wave",
             "epileptiform_count", "region_count", "multi_region",
             "frontal", "temporal", "central", "parietal", "occipital", "generalized"]
META_COLS = ["age", "sex", "sleep", "awake", "preterm", "febrile_seizure",
             "birth_asphyxia", "family_history", "medication", "is_asd"]


def normalize_name(s):
    """与 data/build_record_index.normalize_name 保持一致"""
    if not isinstance(s, str):
        return ""
    for ch in (" ", "\u3000", "\u00b7", "~", "-", "_", ".", ","):
        s = s.replace(ch, "")
    return s.strip().lower()


def load_joined():
    """记录表（标签 + 划分）左连接病例表（报告描述 + 年龄）"""
    rec = pd.read_csv(resolve_path("data/record_index.csv"))
    mas = pd.read_csv(resolve_path("data/subject_master.csv"))
    rec["_k"] = rec["folder"].map(normalize_name)
    mas["_k"] = mas["name_raw"].map(normalize_name)
    mas = mas.drop_duplicates("_k").set_index("_k")
    cols = [c for c in WAVE_COLS + META_COLS if c in mas.columns]
    out = rec.join(mas[cols], on="_k", how="left")
    out["_matched"] = out["age"].notna() | out["slow_wave"].notna()
    return out


def to_pos(series):
    """转成 0/1 阳性指示（空值按 0 处理）"""
    return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)


def main():
    df = load_joined()
    print("=" * 74)
    print("标签元数据分析（病例表报告描述 ↔ 三分类标签）")
    print("=" * 74)
    print(f"记录总数 {len(df)}   能匹配到病例表 {int(df['_matched'].sum())} "
          f"({df['_matched'].mean() * 100:.1f}%)")

    report = {"n_records": int(len(df)), "n_matched": int(df["_matched"].sum())}

    print("\n[1] 各划分的病例表字段覆盖率（决定辅助监督可用样本量）:")
    cov = df.groupby("split").apply(
        lambda d: pd.Series({"记录数": len(d),
                             "有病例字段": int(d["_matched"].sum()),
                             "覆盖率%": round(d["_matched"].mean() * 100, 1)}),
        include_groups=False)
    print(cov.to_string())
    report["coverage_by_split"] = cov.to_dict(orient="index")

    sub = df[df["_matched"]].copy()
    sub["age"] = pd.to_numeric(sub["age"], errors="coerce")

    print("\n[2] 年龄分布（决定是否需要年龄校正）:")
    print(f"  中位 {sub['age'].median():.1f} 岁，范围 "
          f"{sub['age'].min():.0f}–{sub['age'].max():.0f}，缺失 {int(sub['age'].isna().sum())}")
    print(sub.groupby("label_name")["age"]
          .describe()[["count", "mean", "std", "min", "50%", "max"]].round(1).to_string())
    report["age_by_label"] = (sub.groupby("label_name")["age"]
                              .describe()[["count", "mean", "std", "min", "50%", "max"]]
                              .round(2).to_dict(orient="index"))

    print("\n[3] 报告描述 ↔ 标签（关键：判断标签是否临床自洽）:")
    ct = {}
    for c in ["slow_wave", "sharp_wave", "spike_wave", "epileptiform_count",
              "region_count", "generalized"]:
        if c not in sub.columns:
            continue
        pos = to_pos(sub[c])
        t = pd.crosstab(sub["label_name"], pos)
        t.columns = ["阴性", "阳性"] if list(t.columns) == [0, 1] else t.columns
        rate = sub.groupby("label_name").apply(lambda d: to_pos(d[c]).mean() * 100,
                                              include_groups=False).round(1)
        print(f"\n  -- {c}  阳性率: "
              + "  ".join(f"{k}={v}%" for k, v in rate.items()))
        print(t.to_string())
        ct[c] = {"pos_rate_by_label": rate.to_dict(),
                 "counts": t.to_dict()}

    report["wave_vs_label"] = ct

    print("\n[4] 结论提示:")
    if "slow_wave" in ct:
        r = ct["slow_wave"]["pos_rate_by_label"]
        print(f"  Normal 的慢波阳性率 = {r.get('Normal')}%  → "
              "若接近 0，说明标签与报告描述高度自洽，数据本身信息充足，")
        print("  当前 AUC 0.63 主要来自'信息利用不足'而非'信号本身弱'。")
        print("  这些字段可作辅助监督目标，但不可直接作特征（标签泄漏）。")

    out = resolve_path("outputs/label_metadata_analysis.json")
    ensure_dir(os.path.dirname(out))
    save_json(report, out)
    print(f"\n结果已保存: {out}")
    return report


if __name__ == "__main__":
    main()
