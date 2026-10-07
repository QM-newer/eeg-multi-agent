# scripts/compute_subject_weights.py
"""
按"被试级"验证准确率重新计算四个诊断Agent的融合权重

为什么需要：
    权重原本按 epoch 级验证准确率计算。但实测 epoch 级判别力很弱且各Agent排序与
    被试级不一致（epoch 级 LightGBM 最低 0.51，被试级反而最高 0.62），
    导致融合结果被"被试级更差的模型"拖累（融合 0.559 < 单Agent 0.618）。
    临床决策单位是"一份记录一个结论"，权重口径必须与之对齐。

用法：
    python scripts/compute_subject_weights.py --max-epochs-per-record 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import accuracy_score

from config import resolve_path
from training.common import accuracies_to_weights
from training.common import prepare_dataset
from utils.io_utils import save_json, load_json

from agents.diagnosis_agents.lightgbm_agent import LightGbmAgent
from agents.diagnosis_agents.eegnet_agent import EegNetAgent
from agents.diagnosis_agents.tcn_agent import TcnAgent
from agents.diagnosis_agents.transformer_agent import TransformerAgent

AGENT_PATHS = {
    "lightgbm": (LightGbmAgent, "checkpoints/lightgbm/model.pkl"),
    "eegnet": (EegNetAgent, "checkpoints/eegnet/best.pth"),
    "tcn": (TcnAgent, "checkpoints/tcn/best.pth"),
    "transformer": (TransformerAgent, "checkpoints/transformer/best.pth"),
}


def main(max_epochs_per_record=20, split="val", write=True):
    print("=" * 70)
    print(f"被试级权重计算（{split} 集，每记录最多 {max_epochs_per_record} epoch）")
    print("=" * 70)

    data = prepare_dataset(max_epochs_per_record=max_epochs_per_record,
                           normalize=None, verbose=False)
    idx = data["val_idx"] if split == "val" else data["test_idx"]
    X, y = data["X"][idx], data["y"][idx]
    sids = np.asarray(data["subject_ids"])[idx]
    uniq = sorted(set(sids.tolist()))
    sub_y = np.array([y[sids == u][0] for u in uniq])
    print(f"  {X.shape[0]} 个 epoch，{len(uniq)} 个被试")

    accs = {}
    for name, (cls, rel) in AGENT_PATHS.items():
        path = resolve_path(rel)
        if not os.path.exists(path):
            print(f"  [跳过] {name} 无权重文件")
            continue
        agent = cls(name, model_path=path)
        probs = np.zeros((X.shape[0], 3), dtype=np.float64)
        for i in range(X.shape[0]):
            probs[i] = agent.predict(X[i])["prob"]
        ep_acc = accuracy_score(y, probs.argmax(axis=1))
        P = np.vstack([probs[sids == u].mean(axis=0) for u in uniq])
        sub_acc = accuracy_score(sub_y, P.argmax(axis=1))
        accs[name] = round(float(sub_acc), 4)
        print(f"  {name:12s} epoch级={ep_acc:.4f}   被试级={sub_acc:.4f}")

    # 基线用"多数类基线"而非随机水平：类别不平衡（Normal 55%）时，
    # "全猜多数类"的退化模型也有 0.55 准确率，必须扣掉这部分才反映真实能力
    baseline = float(max(np.bincount(sub_y).tolist()) / len(sub_y))
    weights = accuracies_to_weights(accs, floor=baseline)
    print(f"\n被试级准确率: {accs}   多数类基线: {baseline:.4f}")
    print("融合权重:", {k: round(v, 4) for k, v in weights.items()})

    if write:
        wp = resolve_path("checkpoints/agent_weights.json")
        prev = load_json(wp) or {}
        save_json({"accuracies": accs, "weights": weights,
                   "level": "subject", "split": split,
                   "prev_epoch_level_accuracies": prev.get("accuracies")}, wp)
        vp = resolve_path("outputs/subject_val_accuracies.json")
        save_json(accs, vp)
        print(f"\n已写入: {wp}")
        print(f"已写入: {vp}")
    return accs, weights


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs-per-record", type=int, default=20)
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    main(max_epochs_per_record=a.max_epochs_per_record, split=a.split, write=not a.no_write)
