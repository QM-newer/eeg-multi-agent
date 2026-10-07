# scripts/run_batch_evaluation.py
"""
批量评估脚本
用法：python scripts/run_batch_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from orchestrator.orchestrator import Orchestrator
from evaluation.metrics import calculate_metrics, print_metrics
from config import CLASS_NAMES, N_CLASSES, DATA_CONFIG, RANDOM_SEED


def load_real_test_set(max_epochs_per_record=None, split="test", normalize=None):
    """
    加载真实测试集（epoch 缓存）

    默认返回 RAW(µV) 数据：各诊断Agent内部会按训练集统计量做归一化，
    与训练时"prepare_dataset(normalize='per_channel')"的输入口径一致。
    若这里先归一化再交给Agent，就会二次归一化，预测会退化为单一类。
    """
    from data.epoch_cache import EpochCache
    from training.common import prepare_dataset

    if max_epochs_per_record is None:
        max_epochs_per_record = DATA_CONFIG.get("max_epochs_per_record")
    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize=normalize, verbose=False)
    idx = data["test_idx"] if split == "test" else data["val_idx"]
    return data["X"][idx], data["y"][idx], np.asarray(data["subject_ids"])[idx]


def accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def main(max_epochs_per_record=None, n_samples=None, split="test"):
    print("=" * 60)
    print("多Agent脑电诊断系统 - 批量评估")
    print("=" * 60)

    # 初始化系统
    print("\n初始化系统...")
    system = Orchestrator(device="cpu")
    print(f"  诊断Agent数量: {len(system.diagnosis_agents)}")

    # 真实测试集（epoch 缓存）
    from training.common import cache_available
    subject_ids = None
    if not cache_available():
        print("  [警告] 未找到 epoch 缓存，回退到模拟数据（结果无意义）")
        X, y_true = generate_mock_dataset(n_samples=50)
    else:
        X, y_true, subject_ids = load_real_test_set(
            max_epochs_per_record=max_epochs_per_record, split=split)
        print(f"  真实{split}集: {len(X)} 个 epoch，{len(set(subject_ids))} 个被试")
        if n_samples and len(X) > n_samples:
            rng = np.random.default_rng(RANDOM_SEED)
            sel = np.sort(rng.choice(len(X), n_samples, replace=False))
            X, y_true, subject_ids = X[sel], y_true[sel], subject_ids[sel]
            print(f"  随机抽取 {len(X)} 个用于评估")

    # 批量预测
    print(f"\n批量诊断中... ({len(X)} 个样本)")
    y_pred = []
    y_prob = []
    # 缓存每个样本的单Agent预测结果，避免二次推理
    per_agent_preds = {ag.name: [] for ag in system.diagnosis_agents}
    per_agent_probs = {ag.name: [] for ag in system.diagnosis_agents}

    for i in range(len(X)):
        result = system.diagnose(X[i])
        y_pred.append(result["final_label"])
        y_prob.append(result["final_prob"])

        # 从诊断结果中收集各单Agent的预测（无需重新推理）
        for r in result["agent_results"]:
            per_agent_preds[r["agent_name"]].append(r["label"])
            if "prob" in r:
                per_agent_probs[r["agent_name"]].append(r["prob"])

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(X)} 个样本")

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 计算评价指标
    print("\n计算评价指标...")
    metrics = calculate_metrics(y_true, y_pred, y_prob, average="weighted")

    # 打印结果
    print()
    print_metrics(metrics, CLASS_NAMES)

    # 各单Agent对比
    print("\n各单Agent性能对比（epoch 级）:")
    print("-" * 60)
    for agent in system.diagnosis_agents:
        agent_preds = np.array(per_agent_preds[agent.name])
        agent_acc = accuracy_score(y_true, agent_preds)
        print(f"  {agent.name:15s}: 准确率 = {agent_acc:.4f}")
    print("-" * 60)

    # ---- 被试级聚合（关键）----
    # 实测：epoch 级最佳单特征 AUC 仅 0.59，按被试平均后升到 0.65，
    # 说明判别信息主要在"被试整体"而非单个 2 秒片段上，故必须按被试聚合再评估。
    if subject_ids is not None and len(set(subject_ids)) > 1:
        print("\n被试级聚合评估（同一被试所有 epoch 的概率取平均后投票）:")
        print("-" * 60)
        uniq = sorted(set(subject_ids))
        sub_y = np.array([y_true[subject_ids == u][0] for u in uniq])

        def aggregate(pred_labels, probs=None):
            """按被试聚合：优先平均概率，其次多数投票"""
            if probs is not None:
                P = np.vstack([probs[subject_ids == u].mean(axis=0) for u in uniq])
                return P.argmax(axis=1)
            return np.array([np.bincount(pred_labels[subject_ids == u]).argmax()
                             for u in uniq])

        print(f"  被试数: {len(uniq)}   融合系统(epoch级投票后再聚合): "
              f"{accuracy_score(sub_y, aggregate(y_pred, y_prob)):.4f}")
        for agent in system.diagnosis_agents:
            ap = np.array(per_agent_preds[agent.name])
            print(f"    {agent.name:15s}: 被试级准确率 = {accuracy_score(sub_y, aggregate(ap)):.4f}")

        # 被试级融合：每个Agent先对被试给出"整体意见"（概率平均），再按权重投票。
        # 这比"epoch级投票后再聚合"更符合临床口径，也避免弱模型在逐片段投票中稀释强模型。
        if all(len(per_agent_probs[n]) == len(X) for n in per_agent_probs):
            w = {n: v for n, v in system.weights.items() if n in per_agent_probs}
            tot = sum(w.values()) or 1.0
            w = {n: v / tot for n, v in w.items()}
            fused = np.zeros((len(uniq), N_CLASSES))
            for n, wv in w.items():
                P = np.vstack([np.asarray(per_agent_probs[n])[subject_ids == u].mean(axis=0)
                               for u in uniq])
                fused += wv * P
            sub_fused_pred = fused.argmax(axis=1)
            acc_fused_sub = accuracy_score(sub_y, sub_fused_pred)
            f1_fused_sub = f1_score(sub_y, sub_fused_pred, average="macro")
            print(f"  >> 被试级融合（各Agent先出被试级意见再加权投票）: "
                  f"acc={acc_fused_sub:.4f}  macro-F1={f1_fused_sub:.4f}")
        print("-" * 60)

    print("\n评估完成！")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs-per-record", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="评估样本数（默认全测）")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    a = ap.parse_args()
    main(max_epochs_per_record=a.max_epochs_per_record, n_samples=a.n_samples,
         split=a.split)
