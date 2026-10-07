# EEG Multi-Agent 诊断系统

基于多Agent协作的脑电自动诊断系统，针对界限性EEG等边界样本设计，通过多模型协作与分歧仲裁提升诊断准确性和一致性。

## 项目结构

```
eeg_multi_agent/
├── config.py                 # 全局配置
├── agents/                   # Agent层（9个Agent）
│   ├── base_agent.py         # Agent基类
│   ├── feature_agents/       # 特征工程组（3个）
│   ├── diagnosis_agents/     # 诊断推理组（4个）
│   └── validation_agents/    # 解释验证组（2个）
├── orchestrator/             # 协调层
│   ├── orchestrator.py       # 协调器主类
│   ├── voting.py             # 加权投票
│   ├── disagreement.py       # 分歧检测与仲裁
│   └── dynamic_weight.py     # 动态权重调整
├── models/                   # 深度学习模型
│   ├── eegnet.py
│   ├── tcn.py
│   └── eeg_transformer.py
├── data/                     # 数据层
│   ├── reader.py             # 数据读取
│   ├── preprocessor.py       # 预处理流水线
│   └── dataset.py            # 数据集构建
├── evaluation/               # 实验评估
│   └── metrics.py            # 评价指标
├── utils/                    # 工具函数
├── scripts/                  # 可执行脚本
│   ├── run_single_diagnosis.py   # 单样本诊断演示
│   └── run_batch_evaluation.py  # 批量评估
├── checkpoints/              # 模型权重
├── results/                  # 实验结果
└── notebooks/                # 探索性notebook
```

## 系统架构

三层九Agent金字塔架构：

- **特征工程层**（3个Agent）：时域、频域、时频域特征提取
- **诊断推理层**（4个Agent）：LightGBM、EEGNet、TCN、Transformer
- **解释验证层**（2个Agent）：可解释性分析、一致性校验
- **协调层**：任务调度、加权投票、分歧仲裁、动态权重

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- NumPy, SciPy, scikit-learn

### 安装依赖

```bash
pip install -r requirements.txt
```

### 单样本诊断演示

```bash
cd eeg_multi_agent
python scripts/run_single_diagnosis.py
```

### 批量评估

```bash
python scripts/run_batch_evaluation.py
```

> 注：当前模型为随机初始化占位模式，需先通过 `training/` 训练并将权重放入 `checkpoints/`，再传入 `Orchestrator(model_paths=...)` 切换为真实推理。

## 核心功能

1. **多Agent协作诊断**：四种不同归纳偏置的模型协作，多样性最大化
2. **分歧仲裁机制**：边界样本自动触发二次诊断
3. **动态权重调整**：根据信号质量自适应调整各Agent权重
4. **可解释性分析**：特征重要性排序与诊断解释
5. **一致性校验**：多Agent结果一致性评估，可疑样本自动标记

## 数据格式

支持格式：
- XLTEK/Natus (.eeg/.erd/.ent)
- EDF (.edf)

分类标签：三分类（0=Normal 正常, 1=Borderline 界限性, 2=Abnormal 异常）

## 数据清洗与标签定义

病例来源：`脑电+病例_原始备份.xlsx`（项目上级目录，1037 条原始记录），医生已提供结构化标注。

清洗流水线：`data/cleaner.py`（对应设计文档阶段一），执行列名规范化 → 匿名化 → 重复EEG标记 → 日期/年龄统一 → 三分类编码 → 波形变量标准化，输出 `data/subject_master.csv`。

运行方式：

```bash
python -m data.cleaner
```

清洗要点：
- 原始备份表头带注释（如 `异常\n（正常=0，异常=1，界限=2）`），已做列名规范化，同时兼容简化表头版本
- **年龄以医生记录为准**：出生日期列有 199 条为 `1970-01-01`（Excel 空日期解析为 Unix epoch），计算值不可靠；差异 >1 岁的 53 条已用 `age_check_flag` 标记待核查
- 重复检查记录 10 人 20 条，已用 `eeg_sequence` / `first_record` 标记（敏感性分析可仅保留首次记录）

「异常」列与「脑电图印象」文本交叉验证 **0 冲突**，映射规则如下：

| 原始「异常」列 | 脑电图印象文本 | 类别 | 类别索引 | 样本数 |
|---|---|---|---|---|
| 0 | 正常范围… | Normal | 0 | 508 (55.2%) |
| 2 | 界限性… | Borderline | 1 | 92 (10.0%) |
| 1 | 异常… | Abnormal | 2 | 321 (34.8%) |
| 空 | 空 | 无标签，剔除 | — | 116 |

> 注意：原始编码到类别索引**不是恒等映射**（1→Abnormal、2→Borderline），规则已固化在 `data/label_mapper.py` 的 `RAW_CODE_TO_CLASS` 中。

可用样本 **921 条 / 911 名受试者**，类别不平衡比 5.52:1（界限性样本最少，是本项目的核心难点）。

## 当前进度

- [x] Agent框架与统一接口（BaseAgent）
- [x] 三个特征Agent（时域/频域/时频域）
- [x] 四个诊断Agent（LightGBM/EEGNet/TCN/Transformer）
- [x] 两个验证Agent（可解释性/一致性）
- [x] 协调层（加权投票/分歧检测/动态权重）
- [x] 深度模型结构定义（models/）
- [x] 评估指标与演示脚本
- [x] 数据层骨架与工具函数
- [x] **病例标签映射（data/label_mapper.py，基于真实病例数据）**
- [x] **数据清洗流水线（data/cleaner.py，输出 data/subject_master.csv）**
- [x] **真实EEG文件解析（data/xltek_reader.py，XLTEK .erd delta 解码，已与参考实现结构级校验通过）**
- [x] **记录级主表（data/build_record_index.py → data/record_index.csv，1830 条记录对齐）**
- [x] **epoch 缓存构建（scripts/build_epoch_cache.py + data/epoch_cache.py）**
- [ ] 模型训练（training/，已接入真实缓存，待缓存构建完成后重训）
- [ ] 消融实验与对比评估

### 真实数据接入说明

- 原始信号在 `E:/脑电数据按姓名整理`（1830 条记录 / 1778 名受试者，663GB），
  采样率 500Hz，32 AC + 3 DC = 35 通道，记录中位时长约 4 小时
- 解码要点（与上游 XltekDataReader 的差异见 `data/xltek_reader.py` 文件头注释）：
  2 字节增量为小端、discardbits 只乘一次、绝对值统一排在包尾
- 快速自检：`python scripts/probe_read.py --seconds 10`
- 记录主表：`python -m data.build_record_index`
- epoch 缓存：`python scripts/build_epoch_cache.py --workers 12`
  （1598 条 × 100 epoch，输出 `E:/eeg_asd_cache/epochs`，约 6GB / 80 分钟）

### 预处理与关键实测参数

- 采样率 500Hz → 带通 0.5-45Hz → **50Hz 陷波**（本数据集工频干扰最高占 63% 能量，
  仅靠 45Hz 四阶 Butterworth 衰减不足 5dB，必须先陷波）→ 降采样 250Hz → 平均参考
- 2 秒 epoch、重叠 50%，按每条记录自身峰峰值中位数的 3 倍做自适应伪迹剔除
  （保留率中位约 0.91），每条记录均匀抽样至多 100 个 epoch
- 通道：默认 Ch1~Ch19（`config.DATA_CONFIG["eeg_channel_indices"]`）。
  Ch1~Ch23 幅值正常，Ch24~Ch27 峰值恰为满 16bit 量程（非 EEG），Ch33~Ch35 为恒定 DC 通道
- 实验结果与结论见上级目录 `项目进展报告.md`（test 集：epoch 级融合 0.537，
  被试级最优单Agent 0.597，多数类基线 0.550）
- 归一化一致性：训练集估计的逐通道 mean/std 存于 `checkpoints/norm_stats.json`，
  训练（`training/common.prepare_dataset`）与推理（`agents/diagnosis_agents/*._preprocess`）
  共用同一变换；没有该文件时 Agent 退回逐样本全局标准化（旧行为）
- 类别不平衡：Borderline 仅约 12%，深度模型用带权交叉熵，LightGBM 用 `class_weight="balanced"`

## 论文引用

待补充...
