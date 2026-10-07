# 脑电多智能体辅助诊断报告

- 被试编号：`SUBJ_0034`
- 生成时间：2026-10-06 23:43
- 分析样本：20 个 epoch（2s / 250Hz / 19 通道，每记录最多取 20 个）
- 临床参考标签：**Abnormal**（医生判读，仅供对照）

## 一、结论

**系统诊断：可疑/界限性倾向（Abnormal 36.7% vs Normal 34.4%）**

| 类别 | 平均概率 | 跨epoch波动(std) |
|---|---|---|
| Normal | 34.4% | 0.049 |
| Borderline | 28.9% | 0.023 |
| Abnormal | 36.7% | 0.046 |

## 二、信号质量

- epoch 质量等级分布：high 20 个
- 质量等级由 SNR 与伪迹比例自动评定（utils/signal_utils），并驱动各 Agent 动态权重。

## 三、多Agent意见与共识过程

| 诊断Agent | 平均动态权重 | 主要意见（epoch票数） |
|---|---|---|
| lightgbm | 0.459 | Abnormal 12、Normal 7、Borderline 1 |
| eegnet | 0.439 | Normal 10、Borderline 1、Abnormal 9 |
| tcn | 0.050 | Normal 8、Abnormal 12 |
| transformer | 0.051 | Normal 19、Borderline 1 |

- 分歧触发率：20/20（100%，触发即进入补充特征→重推理→可信度仲裁流程）
- 仲裁执行率：20/20
- 跨Agent一致性得分：0.515（建议人工复核）

分歧仲裁示例（首个分歧 epoch）：

- 补充空间域特征 43 维（扩充后 171 维）后二次诊断；特征可信度：lightgbm=0.77、eegnet=0.82、tcn=0.78、transformer=0.81

## 四、证据链（可解释性）

各 epoch 可解释性 Agent 输出的高频 Top 特征（特征索引 @ 532 维多域特征）：

| 特征索引 | 出现次数 |
|---|---|
| 7 | 7 |
| 68 | 6 |
| 64 | 6 |
| 98 | 5 |
| 58 | 5 |
| 66 | 5 |
| 117 | 4 |
| 3 | 4 |

> 特征索引按 [时域 0-227 | 频域 228-379 | 时频 380-531] 分块，可对照 `training/common.py::_make_feature_agents` 定位特征含义。

## 五、说明与免责

- 本报告由 EEG-MAS 多智能体系统自动生成，仅供研究/辅助参考，不构成临床诊断结论。
- 单被试推理耗时 3.0s（20 epoch，CPU）。
