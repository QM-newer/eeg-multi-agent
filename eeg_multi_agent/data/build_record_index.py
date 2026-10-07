# data/build_record_index.py
"""
构建「记录级主表」 data/record_index.csv

作用：把两份独立来源合成为训练管线的唯一入口
  1) 文件索引：E 盘 organize 阶段产出的记录索引（folder / record_id / eeg_file / 时长 / 采样率 / 原始编码标签）
  2) 病例标签：data/subject_master.csv（由 脑电+病例_原始备份.xlsx 清洗，类别索引标签，本项目的金标准）

对齐规则（保守优先，宁可少匹配也不错匹配）：
  - 姓名归一化后精确相等：空格 / · / ~ / 全角空格会被剔除
  - 归一化后不等时，仅在「两侧检查日期完全相等」且相似度 ≥ 0.6 时才允许模糊匹配（用于 皓/晧 这类异体字）
  - 同一姓名有多条病例记录时，取日期最接近的一条；日期差 > 7 天会打上 date_mismatch 标记

标签优先级：项目病例表（金标准） > organize 索引标签（原始编码转类别索引）

输出字段见 FIELDS 常量。

用法：
    python -m data.build_record_index                # 构建并打印报告
    python -m data.build_record_index --strict       # 额外校验文件是否真实存在（较慢）
"""

import os
import csv
import sys
import glob
import difflib
import datetime
import argparse
from collections import defaultdict, Counter

import numpy as np

from config import DATA_CONFIG, CLASS_NAMES, RANDOM_SEED
from data.label_mapper import map_raw_code

FIELDS = [
    "record_id",        # 记录唯一ID（文件名主干，不含扩展名）
    "subject_id",       # 受试者ID（按归一化姓名归并，用于防泄露划分）
    "folder",           # E 盘目录名（即设备端登记的姓名）
    "eeg_path",         # .eeg 头/蒙太奇文件路径
    "erd_paths",        # 该记录的 .erd 原始信号分段，按时间顺序用 ; 分隔
    "n_segments",       # .erd 分段数
    "duration_s",       # 记录总时长（秒）
    "sfreq",            # 采样率
    "n_channels",       # 总通道数（AC + DC）
    "eeg_date",         # 记录检查日期
    "label",            # 最终标签（类别索引 0/1/2），无标签为空
    "label_name",       # 标签名
    "label_raw",        # 原始数据「异常」列编码（0/1/2）
    "label_source",     # project / raw_index / none
    "label_conflict",   # 两个来源不一致时置 1
    "match_method",     # exact / fuzzy_same_date / none
    "date_diff_days",   # 设备日期与病例日期之差（天）
    "date_mismatch",    # 日期差 > 7 天置 1
    "first_record",     # 同一受试者的首次记录（默认只用首次，避免同一人重复计入）
    "n_records_subject",  # 该受试者的记录数
    "flags",            # 上游索引给出的其它标记
    "split",            # 受试者级分层划分：train / val / test
]

DATE_FORMATS = ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y年%m月%d日")


# ============================================================
# 工具函数
# ============================================================
def normalize_name(s):
    """姓名归一化：去空格、·、~、全角空格等，只保留汉字与字母数字"""
    if not s:
        return ""
    for ch in (" ", "\u3000", "\u00b7", "~", "-", "_", ".", ","):
        s = s.replace(ch, "")
    return s.strip().lower()


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    for f in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None


def find_erd_segments(folder_path, record_id):
    """按记录ID前缀找出该记录的所有 .erd 分段，按 _001/_002 顺序排列"""
    try:
        names = os.listdir(folder_path)
    except OSError:
        return []
    segs = []
    for n in names:
        if not n.lower().endswith(".erd"):
            continue
        stem = n[:-4]
        # 分段命名：<record_id>.erd, <record_id>_001.erd, ...
        if stem == record_id:
            segs.append((0, n))
        elif stem.startswith(record_id + "_"):
            suffix = stem[len(record_id) + 1:]
            try:
                segs.append((int(suffix), n))
            except ValueError:
                segs.append((999, n))
    segs.sort(key=lambda x: x[0])
    return [os.path.join(folder_path, n) for _, n in segs]


def stratified_subject_split(subject_label, train_ratio, val_ratio, seed):
    """按受试者做分层划分（同一受试者的所有记录落在同一集合，避免泄露）"""
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for sid, lab in subject_label.items():
        by_label[lab].append(sid)

    split_of = {}
    for lab, sids in by_label.items():
        sids = sorted(sids)
        rng.shuffle(sids)
        n = len(sids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for i, sid in enumerate(sids):
            if i < n_train:
                split_of[sid] = "train"
            elif i < n_train + n_val:
                split_of[sid] = "val"
            else:
                split_of[sid] = "test"
    return split_of


# ============================================================
# 主流程
# ============================================================
def build(raw_index_csv=None, subject_master_csv=None, data_root=None,
          check_files=False, verbose=True):
    raw_index_csv = raw_index_csv or DATA_CONFIG["raw_index_csv"]
    subject_master_csv = subject_master_csv or DATA_CONFIG["subject_master"]
    data_root = data_root or DATA_CONFIG["data_root"]

    if not os.path.exists(raw_index_csv):
        raise FileNotFoundError(
            f"记录索引不存在: {raw_index_csv}\n"
            "该表由 organize 阶段产出（E:\\eeg_organize_tmp\\subject_master.csv），"
            "若路径变化请修改 config.DATA_CONFIG['raw_index_csv']"
        )
    if not os.path.exists(subject_master_csv):
        raise FileNotFoundError(
            f"病例主表不存在: {subject_master_csv}，请先运行 python -m data.cleaner"
        )

    with open(raw_index_csv, encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))
    with open(subject_master_csv, encoding="utf-8-sig") as f:
        proj_rows = list(csv.DictReader(f))

    # ---------- 病例表按归一化姓名建索引 ----------
    proj_by_name = defaultdict(list)
    for r in proj_rows:
        key = normalize_name(r.get("name_raw"))
        if key:
            proj_by_name[key].append(r)

    # ---------- 逐条记录匹配 ----------
    rows = []
    stats = Counter()
    conflicts = []

    for r in raw_rows:
        folder = (r.get("folder") or "").strip()
        record_id = (r.get("record_id") or "").strip()
        if not folder or not record_id:
            stats["skip_no_id"] += 1
            continue

        folder_path = os.path.join(data_root, folder)
        eeg_path = os.path.join(folder_path, record_id + ".eeg")

        # 兼容上游索引里写死的 F 盘路径
        if not os.path.exists(eeg_path):
            legacy = (r.get("eeg_file") or "").replace("\\", "/")
            if legacy:
                cand = os.path.join(data_root, folder, os.path.basename(legacy))
                if os.path.exists(cand):
                    eeg_path = cand

        erd_paths = find_erd_segments(folder_path, record_id)
        if check_files and not os.path.exists(eeg_path):
            stats["missing_eeg"] += 1
        if check_files and not erd_paths:
            stats["missing_erd"] += 1

        rec_date = parse_date(r.get("eeg_date")) or parse_date(r.get("excel_date"))
        key = normalize_name(folder)

        # ---- 1) 精确姓名匹配 ----
        cands = proj_by_name.get(key, [])
        match_method = "exact" if cands else "none"
        chosen, diff_days = None, None

        if cands:
            best = None
            for p in cands:
                d = None
                p_date = parse_date(p.get("exam_date"))
                if rec_date and p_date:
                    d = abs((rec_date - p_date).days)
                score = (d is None, d if d is not None else 0)
                if best is None or score < best[0]:
                    best = (score, p, d)
            chosen, diff_days = best[1], best[2]

        # ---- 2) 严格模糊匹配：必须同日期 ----
        if chosen is None and rec_date is not None:
            best = None
            for pkey, plist in proj_by_name.items():
                if len(pkey) < 2:
                    continue
                ratio = difflib.SequenceMatcher(None, key, pkey).ratio()
                if ratio < 0.6:
                    continue
                for p in plist:
                    p_date = parse_date(p.get("exam_date"))
                    if p_date != rec_date:
                        continue
                    if best is None or ratio > best[0]:
                        best = (ratio, p, 0)
            if best is not None:
                chosen, diff_days = best[1], best[2]
                match_method = "fuzzy_same_date"

        # ---- 标签融合 ----
        proj_label = None
        if chosen is not None and chosen.get("label") not in ("", None):
            try:
                proj_label = int(float(chosen["label"]))
            except (TypeError, ValueError):
                proj_label = None

        raw_code = r.get("label")
        raw_label = map_raw_code(int(float(raw_code))) if raw_code not in ("", None) else None

        date_mismatch = int(diff_days is not None and diff_days > 7)
        use_project = proj_label is not None and not date_mismatch

        if use_project:
            label, source = proj_label, "project"
        elif raw_label is not None:
            label, source = raw_label, "raw_index"
        elif proj_label is not None:
            label, source = proj_label, "project"   # 日期对不上但只有项目有标签
        else:
            label, source = None, "none"

        conflict = int(proj_label is not None and raw_label is not None and proj_label != raw_label)
        if conflict:
            conflicts.append((record_id, folder, proj_label, raw_label, str(rec_date)))

        stats["total"] += 1
        stats["match_" + match_method] += 1
        stats["src_" + source] += 1
        stats["conflict"] += conflict
        stats["date_mismatch"] += date_mismatch

        rows.append({
            "record_id": record_id,
            "subject_id": "",                       # 稍后按归一化姓名统一编号
            "folder": folder,
            "eeg_path": eeg_path,
            "erd_paths": ";".join(erd_paths),
            "n_segments": len(erd_paths),
            "duration_s": r.get("record_seconds") or "",
            "sfreq": r.get("sf") or "",
            "n_channels": (int(r["num_ac"]) + int(r["num_dc"]))
                          if r.get("num_ac") and r.get("num_dc") else "",
            "eeg_date": str(rec_date) if rec_date else "",
            "label": "" if label is None else label,
            "label_name": CLASS_NAMES[label] if label is not None else "",
            "label_raw": "" if raw_code in ("", None) else raw_code,
            "label_source": source,
            "label_conflict": conflict,
            "match_method": match_method,
            "date_diff_days": "" if diff_days is None else diff_days,
            "date_mismatch": date_mismatch,
            "first_record": "",
            "n_records_subject": "",
            "flags": r.get("flags") or "",
            "split": "",
        })

    # ---------- 受试者归并 + 首次记录标记 ----------
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[normalize_name(row["folder"])].append(row)

    subject_id_of = {}
    for i, skey in enumerate(sorted(by_subject.keys()), start=1):
        subject_id_of[skey] = f"SUBJ_{i:04d}"

    subject_label = {}
    for skey, srows in by_subject.items():
        sid = subject_id_of[skey]
        srows.sort(key=lambda x: (x["eeg_date"] or "9999", x["record_id"]))
        for j, row in enumerate(srows):
            row["subject_id"] = sid
            row["first_record"] = int(j == 0)
            row["n_records_subject"] = len(srows)
        # 受试者标签：取多数（如有冲突取首条有标签记录）
        labs = [r["label"] for r in srows if r["label"] != ""]
        subject_label[sid] = int(labs[0]) if labs else None

    # ---------- 受试者级分层划分 ----------
    labeled_subjects = {k: v for k, v in subject_label.items() if v is not None}
    split_of = stratified_subject_split(
        labeled_subjects,
        DATA_CONFIG["train_ratio"], DATA_CONFIG["val_ratio"], RANDOM_SEED,
    )
    for row in rows:
        row["split"] = split_of.get(row["subject_id"], "")

    # ---------- 写出 ----------
    out_path = DATA_CONFIG["record_index"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    if verbose:
        _print_report(rows, stats, conflicts, out_path)
    return rows


def _print_report(rows, stats, conflicts, out_path):
    n = len(rows)
    print("=" * 68)
    print("记录级主表构建完成 ->", out_path)
    print("=" * 68)
    print(f"记录总数            : {n}")
    print(f"姓名精确匹配        : {stats['match_exact']}")
    print(f"同日期模糊匹配      : {stats['match_fuzzy_same_date']}")
    print(f"未匹配到病例表      : {stats['match_none']}")
    print(f"标签来源 project    : {stats['src_project']}")
    print(f"标签来源 raw_index  : {stats['src_raw_index']}")
    print(f"无标签（将剔除）    : {stats['src_none']}")
    print(f"两来源冲突          : {stats['conflict']}")
    print(f"日期差 > 7 天       : {stats['date_mismatch']}")
    if stats.get("missing_eeg"):
        print(f"[警告] .eeg 缺失     : {stats['missing_eeg']}")
    if stats.get("missing_erd"):
        print(f"[警告] .erd 缺失     : {stats['missing_erd']}")

    labeled = [r for r in rows if r["label"] != ""]
    first = [r for r in labeled if r["first_record"] == 1]
    print("\n有标签记录          :", len(labeled))
    print("其中受试者首次记录  :", len(first), "（默认训练口径）")

    for title, subset in (("全部有标签记录", labeled), ("仅首次记录", first)):
        cnt = Counter(int(r["label"]) for r in subset)
        tot = sum(cnt.values()) or 1
        print(f"\n{title} 类别分布：")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:<11s} {cnt.get(i, 0):5d}  ({cnt.get(i, 0)/tot*100:5.1f}%)")

        sc = Counter(r["split"] for r in subset)
        ns = Counter(r["subject_id"] for r in subset)
        print(f"  划分（受试者级）: train={sc.get('train',0)} val={sc.get('val',0)} "
              f"test={sc.get('test',0)}  |  受试者数={len(ns)}")

    if conflicts:
        print(f"\n[需人工核查] 标签冲突 {len(conflicts)} 条（前 10 条）：")
        for c in conflicts[:10]:
            print("   ", c)


def main():
    ap = argparse.ArgumentParser(description="构建记录级主表 record_index.csv")
    ap.add_argument("--strict", action="store_true", help="校验每个记录的 .eeg/.erd 是否真实存在")
    args = ap.parse_args()
    build(check_files=args.strict)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
