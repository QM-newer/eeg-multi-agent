# data/cleaner.py
"""
数据清洗流水线（对应设计文档 阶段一：数据准备与基线建立）

处理步骤：
1. 列名规范化（兼容原始备份的带注释表头）
2. 匿名化处理（anonymous_id）
3. 标记重复EEG（eeg_sequence / first_record）
4. 统一日期与年龄计算
5. EEG三分类编码（复用 data.label_mapper）
6. 睡眠/脑区/波形等变量标准化编码

输出：data/subject_master.csv（样本-标签映射主表）

用法：
    python -c "from data.cleaner import clean_and_save; clean_and_save()"
"""

import os

import numpy as np
import pandas as pd

from config import CLASS_NAMES, DATA_CONFIG
from data.label_mapper import find_case_xlsx, load_subject_master, normalize_columns
from utils.io_utils import ensure_dir

# 二值字段映射：原始列名 -> 标准名
BINARY_FIELDS = {
    "睡眠": "sleep",
    "清醒": "awake",
    "左侧": "left",
    "右侧": "right",
    "额区": "frontal",
    "颞区": "temporal",
    "中央区": "central",
    "顶区": "parietal",
    "枕区": "occipital",
    "全导": "generalized",
    "是否多个脑区": "multi_region",
    "慢波": "slow_wave",
    "尖波": "sharp_wave",
    "纺锤波": "spindle_wave",
    "棘波": "spike_wave",
    "癫痫波是否为多个": "epileptiform_multi",
}

# 计数字段映射
COUNT_FIELDS = {
    "癫痫波": "epileptiform_count",
    "脑区个数": "region_count",
}

# 保留的临床协变量
CLINICAL_FIELDS = {
    "诊断": "diagnosis",
    "是否为ASD": "is_asd",
    "性别": "sex",
    "居住地": "residence",
    "用药": "medication",        # 0/1 编码列（原始备份中该名称首次出现者）
    "用药_2": "medication_name",  # 具体药名列（重复列名加后缀后）
    "早产": "preterm",
    "高热惊厥": "febrile_seizure",
    "出生缺氧": "birth_asphyxia",
    "家族史": "family_history",
    "剖宫产": "cesarean",
}

# 最终输出列顺序
OUTPUT_COLUMNS = [
    "anonymous_id", "name_raw", "exam_date", "birth_date", "age", "age_source",
    "age_calc", "age_check_flag", "sex",
    "label", "label_name", "label_source",
    "eeg_sequence", "first_record", "n_records",
    "diagnosis", "is_asd", "residence", "medication", "medication_name",
    "sleep", "awake", "left", "right",
    "frontal", "temporal", "central", "parietal", "occipital", "generalized",
    "region_count", "multi_region",
    "slow_wave", "sharp_wave", "spindle_wave", "spike_wave",
    "epileptiform_count", "epileptiform_multi",
    "preterm", "febrile_seizure", "birth_asphyxia", "family_history", "cesarean",
    "impression",
]


# 出生日期的合理下限：早于此视为占位值
# （数据中 199 条出生日期为 1970-01-01，是 Excel 空日期被解析为 Unix epoch 起点所致）
MIN_VALID_BIRTH_YEAR = 1980
MAX_VALID_AGE = 30.0


def compute_age(birth_date, exam_date):
    """
    根据出生日期与检查日期计算年龄（岁）

    对不合理值返回 NaN：
    - 出生日期缺失或早于 MIN_VALID_BIRTH_YEAR（占位值）
    - 出生日期晚于检查日期（负年龄）
    - 年龄超过 MAX_VALID_AGE
    """
    b = pd.to_datetime(birth_date, errors="coerce")
    e = pd.to_datetime(exam_date, errors="coerce")
    if pd.isna(b) or pd.isna(e):
        return np.nan
    if b.year < MIN_VALID_BIRTH_YEAR:
        return np.nan
    age = (e - b).days / 365.25
    if age < 0 or age > MAX_VALID_AGE:
        return np.nan
    return age


def _to_binary(series):
    """将 0/1 浮点列规范为可空整数（保持缺失为 NaN，不臆造 0）"""
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s.isna(), s.round().astype("Int64"))


def _to_count(series):
    """计数列规范为可空整数"""
    return _to_binary(series)


def build_clean_dataset(xlsx_path=None):
    """
    执行完整清洗流程，返回清洗后的主表

    参数:
        xlsx_path: 病例xlsx路径，默认自动查找（优先原始备份）

    返回:
        DataFrame: 清洗后的受试者主表
    """
    xlsx_path = xlsx_path or find_case_xlsx()
    if not xlsx_path or not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"未找到病例数据文件，已尝试目录: {os.path.dirname(xlsx_path or '')}")

    # 读取原始表（保留全部列用于协变量提取）
    raw = normalize_columns(pd.read_excel(xlsx_path))

    # 标签映射（复用已验证的三分类规则）
    labeled = load_subject_master(xlsx_path=xlsx_path, drop_unlabeled=True)
    df = raw.loc[raw.index.isin(labeled.index)].copy() if len(labeled) < len(raw) else raw.copy()

    # 若 load_subject_master 丢弃了无标签样本，需要对齐
    # 用「姓名+检查日期」作为对齐键，保证标签与原始行对应
    if len(labeled) < len(raw):
        key_raw = raw["姓名"].astype(str) + "|" + raw["检查日期"].astype(str)
        key_lab = labeled["name"].astype(str) + "|" + labeled["exam_date"].astype(str)
        lab_map = dict(zip(key_lab, labeled["label"]))
        src_map = dict(zip(key_lab, labeled["label_source"]))
        df = raw.copy()
        df["label"] = key_raw.map(lab_map)
        df["label_source"] = key_raw.map(src_map)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = labeled["label"].values
        df["label_source"] = labeled["label_source"].values

    df["label_name"] = df["label"].apply(lambda x: CLASS_NAMES[int(x)])

    # --- 匿名化 ID ---
    df["_name"] = df["姓名"].astype(str).str.strip()
    unique_names = sorted(df["_name"].dropna().unique())
    anon_map = {n: f"ANON_{i+1:04d}" for i, n in enumerate(unique_names)}
    df["anonymous_id"] = df["_name"].map(anon_map)
    df["name_raw"] = df["_name"]  # 保留原始姓名用于匹配EEG文件目录

    # --- 日期与年龄 ---
    df["exam_date"] = pd.to_datetime(df["检查日期"], errors="coerce")
    df["birth_date"] = pd.to_datetime(df["出生日期"], errors="coerce")
    # 年龄：以医生记录的「年龄」列为准（该列更可靠），
    # 计算值仅作校验；记录缺失时才用计算值回退
    computed_age = df.apply(
        lambda r: compute_age(r["birth_date"], r["exam_date"]), axis=1
    )
    recorded_age = pd.to_numeric(df.get("年龄"), errors="coerce")
    df["age_calc"] = computed_age
    df["age"] = recorded_age.fillna(computed_age)
    df["age_source"] = np.where(
        recorded_age.notna(), "recorded",
        np.where(computed_age.notna(), "calculated", "missing")
    )
    # 两者都有值时，差异 >1 岁标记为待核查
    df["age_check_flag"] = ((recorded_age - computed_age).abs() > 1.0).astype(int)
    df.loc[recorded_age.isna() | computed_age.isna(), "age_check_flag"] = 0

    # --- 重复EEG标记 ---
    df = df.sort_values(["_name", "exam_date"], kind="stable").reset_index(drop=True)
    df["eeg_sequence"] = df.groupby("_name").cumcount() + 1
    df["n_records"] = df.groupby("_name")["_name"].transform("count")
    df["first_record"] = (df["eeg_sequence"] == 1).astype(int)

    # --- 变量标准化编码 ---
    for src, dst in BINARY_FIELDS.items():
        if src in df.columns:
            df[dst] = _to_binary(df[src])
    for src, dst in COUNT_FIELDS.items():
        if src in df.columns:
            df[dst] = _to_count(df[src])
    for src, dst in CLINICAL_FIELDS.items():
        if src in df.columns:
            df[dst] = df[src]

    if "脑电图印象" in df.columns:
        df["impression"] = df["脑电图印象"]

    # 保证所有输出列存在
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def save_subject_master(df, path=None):
    """保存主表为 CSV（utf-8-sig 便于 Excel 直接打开）"""
    path = path or DATA_CONFIG.get("subject_master") or \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "subject_master.csv")
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_cleaning_report(df):
    """打印清洗结果报告"""
    print("=" * 70)
    print("数据清洗报告")
    print("=" * 70)
    print(f"样本数: {len(df)}    受试者数: {df['anonymous_id'].nunique()}")
    print(f"重复检查记录: {(df['eeg_sequence'] > 1).sum()} "
          f"(涉及 {df.loc[df['eeg_sequence'] > 1, 'anonymous_id'].nunique()} 人)")
    print(f"检查日期范围: {df['exam_date'].min()} ~ {df['exam_date'].max()}")
    print(f"年龄: 均值 {df['age'].mean():.1f}  中位 {df['age'].median():.1f}  "
          f"范围 {df['age'].min():.1f}-{df['age'].max():.1f}  "
          f"(缺失 {df['age'].isna().sum()})")
    print(f"  年龄来源: {df['age_source'].value_counts().to_dict()}")
    print(f"  年龄待核查(记录值与计算值差>1岁): {int(df['age_check_flag'].sum())}")
    print()
    print("三分类标签分布:")
    for i, name in enumerate(CLASS_NAMES):
        c = int((df["label"] == i).sum())
        print(f"  {i} {name:12s}: {c:5d}  ({c / len(df):6.2%})")
    print()
    print("波形标记阳性率（非缺失样本中）:")
    for col in ["slow_wave", "sharp_wave", "spindle_wave", "spike_wave",
                "epileptiform_multi", "generalized", "multi_region"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                print(f"  {col:20s}: {int(s.sum()):4d} / {len(s):4d}  ({s.mean():6.2%})")
    print("=" * 70)


def clean_and_save(xlsx_path=None, save_path=None):
    """一步完成清洗与保存"""
    df = build_clean_dataset(xlsx_path)
    print_cleaning_report(df)
    path = save_subject_master(df, save_path)
    print(f"主表已保存: {path}")
    return df


if __name__ == "__main__":
    clean_and_save()
