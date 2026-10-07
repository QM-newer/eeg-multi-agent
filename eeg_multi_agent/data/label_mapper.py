# data/label_mapper.py
"""
病例标签映射

将「脑电+病例_工作文件.xlsx」中医生的结构化标注映射为本项目三分类标签。

已验证的映射关系（1037 条记录，与「脑电图印象」文本交叉验证 0 冲突）：
    原始「异常」列 = 0  -> Normal     正常      (508 条，印象"正常范围…")
    原始「异常」列 = 1  -> Abnormal   异常      (321 条，印象"异常…")
    原始「异常」列 = 2  -> Borderline 界限性    ( 92 条，印象"界限性…")
    原始「异常」列 为空 -> 无标签，需剔除       (116 条，印象也为空)

注意：原始编码到类别索引【不是恒等映射】，
      config.CLASS_NAMES = ["Normal", "Borderline", "Abnormal"]
      因此 1 -> 2(Abnormal)、2 -> 1(Borderline)。

用法：
    from data.label_mapper import load_subject_master, summarize_labels
    df = load_subject_master()          # 只保留有标签的样本
    print(summarize_labels(df))
"""

import os

import numpy as np
import pandas as pd

from config import CLASS_NAMES, CLASS_TO_IDX, N_CLASSES

# 病例数据候选文件名（按优先级；原始备份优先，因其保留最完整的表头注释）
CASE_XLSX_CANDIDATES = [
    "脑电+病例_原始备份.xlsx",
    "脑电+病例_工作文件.xlsx",
    "病历数据-提取结果.xlsx",
]
CASE_XLSX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
)
DEFAULT_XLSX = os.path.join(CASE_XLSX_DIR, CASE_XLSX_CANDIDATES[0])


def find_case_xlsx(search_dir=None):
    """
    在指定目录（默认项目上级目录）中查找病例数据文件

    返回:
        str or None: 找到的文件路径
    """
    search_dir = search_dir or CASE_XLSX_DIR
    for name in CASE_XLSX_CANDIDATES:
        p = os.path.join(search_dir, name)
        if os.path.exists(p):
            return p
    return None


def normalize_columns(df):
    """
    规范化列名：去掉原始表头中的换行与括号注释

    例：'异常\\n（正常=0，异常=1，界限=2）' -> '异常'
        '脑电图印象\\n（抄整段文字）'      -> '脑电图印象'

    这样就同时兼容「原始备份」（带注释表头）和「工作文件」（简化表头）两个版本。
    """
    df = df.copy()
    new_cols = {}
    for c in df.columns:
        name = str(c)
        # 去掉换行及其后的注释
        name = name.split("\n")[0].split("\r")[0]
        # 去掉行内括号注释（中文括号）
        for lb, rb in [("（", "）"), ("(", ")")]:
            i = name.find(lb)
            if i != -1 and name.find(rb, i) != -1:
                name = name[:i]
        name = name.strip()
        # 处理 pandas 生成的 Unnamed 空列
        if name.startswith("Unnamed:"):
            name = ""
        new_cols[c] = name
    df = df.rename(columns=new_cols)

    # 处理重命名后产生的重复列名（如「用药」在原始备份中有 0/1 编码与药名两列），
    # 保留首个原名，后续依次加 _2 / _3 后缀
    seen = {}
    final_cols = []
    for c in df.columns:
        if c == "":
            final_cols.append(c)
            continue
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 1
            final_cols.append(c)
    df.columns = final_cols

    # 丢弃无名列
    df = df.loc[:, [c for c in df.columns if c != ""]]
    return df

# 原始「异常」列编码 -> 类别名（已交叉验证，勿随意改动）
RAW_CODE_TO_CLASS = {
    0: "Normal",
    1: "Abnormal",
    2: "Borderline",
}

# 文本兜底关键词（按优先级顺序匹配）
TEXT_RULES = [
    ("界限性", "Borderline"),
    ("界限", "Borderline"),
    ("边缘", "Borderline"),
    ("异常", "Abnormal"),
    ("正常范围", "Normal"),
    ("正常", "Normal"),
]


def map_raw_code(code):
    """
    将原始「异常」列编码映射为类别索引

    参数:
        code: 0 / 1 / 2 / None / NaN

    返回:
        int or None: 类别索引（config.CLASS_TO_IDX），无标签返回 None
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    class_name = RAW_CODE_TO_CLASS.get(code)
    return CLASS_TO_IDX.get(class_name) if class_name else None


def map_from_text(text):
    """
    从「脑电图印象」文本兜底判定类别（仅在「异常」列缺失时使用）

    参数:
        text: str，脑电图印象文本

    返回:
        int or None: 类别索引，无法判定返回 None
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    text = str(text).strip()
    if not text:
        return None
    for keyword, class_name in TEXT_RULES:
        if keyword in text:
            return CLASS_TO_IDX.get(class_name)
    return None


def _make_subject_id(row):
    """生成受试者ID：优先用姓名（按人划分，避免同一人跨集泄露）"""
    name = row.get("姓名")
    if name is not None and not (isinstance(name, float) and np.isnan(name)):
        return str(name).strip()
    return None


def load_subject_master(xlsx_path=None, use_text_fallback=True, drop_unlabeled=True):
    """
    读取病例表并构建受试者主表

    参数:
        xlsx_path: xlsx 路径，默认取项目上级目录的「脑电+病例_工作文件.xlsx」
        use_text_fallback: 「异常」列缺失时是否用脑电图印象文本兜底
        drop_unlabeled: 是否丢弃无标签样本

    返回:
        DataFrame，含列：
            subject_id   受试者ID（姓名）
            name         姓名
            exam_date    检查日期
            raw_code     原始「异常」列编码
            label        类别索引（0/1/2）
            label_name   类别名
            impression   脑电图印象原文
            epileptiform 癫痫波计数（可辅助分析）
            has_spike    棘波标记
            has_slow     慢波标记
    """
    xlsx_path = xlsx_path or find_case_xlsx() or DEFAULT_XLSX
    df = pd.read_excel(xlsx_path)
    # 规范化列名（兼容原始备份的带注释表头）
    df = normalize_columns(df)

    # 主标签：来自「异常」列
    df["label"] = df["异常"].apply(map_raw_code)

    # 兜底：文本判定
    if use_text_fallback and "脑电图印象" in df.columns:
        fallback = df["脑电图印象"].apply(map_from_text)
        df["label_source"] = np.where(df["label"].notna(), "raw_code",
                                      np.where(fallback.notna(), "text", "none"))
        df["label"] = df["label"].fillna(fallback)
    else:
        df["label_source"] = np.where(df["label"].notna(), "raw_code", "none")

    # 类别名
    df["label_name"] = df["label"].apply(
        lambda x: CLASS_NAMES[int(x)] if x is not None and not np.isnan(x) else None
    )

    # 受试者与检查信息
    df["subject_id"] = df.apply(_make_subject_id, axis=1)
    df["name"] = df["姓名"] if "姓名" in df.columns else None
    df["exam_date"] = df["检查日期"] if "检查日期" in df.columns else None
    df["impression"] = df["脑电图印象"] if "脑电图印象" in df.columns else None
    df["epileptiform"] = df["癫痫波"] if "癫痫波" in df.columns else None
    df["has_spike"] = df["棘波"] if "棘波" in df.columns else None
    df["has_slow"] = df["慢波"] if "慢波" in df.columns else None
    df["raw_code"] = df["异常"]

    if drop_unlabeled:
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)

    keep_cols = [
        "subject_id", "name", "exam_date", "raw_code", "label", "label_name",
        "label_source", "impression", "epileptiform", "has_spike", "has_slow",
    ]
    return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)


def summarize_labels(df):
    """
    统计标签分布

    返回:
        dict: {total, n_subjects, counts:{类名:数量}, ratios:{类名:占比}, imbalance_ratio}
    """
    counts = {name: int((df["label"] == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    total = int(len(df))
    ratios = {k: (v / total if total else 0.0) for k, v in counts.items()}
    minority = min(v for v in counts.values() if v > 0) if counts else 0
    majority = max(counts.values()) if counts else 0
    return {
        "total": total,
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "counts": counts,
        "ratios": {k: round(v, 4) for k, v in ratios.items()},
        "imbalance_ratio": round(majority / minority, 2) if minority else None,
    }


def print_label_summary(df):
    """美观打印标签分布"""
    s = summarize_labels(df)
    print("=" * 60)
    print("病例标签分布")
    print("=" * 60)
    print(f"样本总数: {s['total']}    受试者数: {s['n_subjects']}")
    print(f"类别不平衡比(最大/最小): {s['imbalance_ratio']}")
    print()
    for i, name in enumerate(CLASS_NAMES):
        c = s["counts"][name]
        r = s["ratios"][name]
        bar = "█" * int(r * 40)
        print(f"  {i} {name:12s}: {c:5d}  ({r:6.2%})  {bar}")
    print()
    if "label_source" in df.columns:
        print("标签来源:", df["label_source"].value_counts().to_dict())
    print("=" * 60)
    return s
