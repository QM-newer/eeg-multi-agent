# scripts/_eval_event_features.py — 事件级特征 A/B 快速评估（小规模）
# 输出: outputs/event_feature_ab.json
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from training.common import (prepare_dataset, extract_multi_domain_features, set_seed)
from features.fusion import FeatureFusion
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.io_utils import save_json
from config import resolve_path


def main():
    set_seed()
    print("Loading data...")
    data = prepare_dataset(max_epochs_per_record=5, verbose=False, label_mode="binary")
    X, y = data["X"], data["y"]

    # 只取 5000 训练 + 2000 验证
    train_idx = data["train_idx"][:5000]
    val_idx = data["val_idx"][:2000]

    print(f"Extracting features with event (train={len(train_idx)}, val={len(val_idx)})...")
    t0 = time.time()
    feat_train, dims = extract_multi_domain_features(X[train_idx], include_event=True)
    feat_val, _ = extract_multi_domain_features(X[val_idx], include_event=True)
    print(f"  {time.time()-t0:.0f}s  Dims: {dims}  Total: {feat_train.shape[1]}")

    y_train, y_val = y[train_idx], y[val_idx]

    fusion = FeatureFusion(n_select=128, include_event=True)
    feat_train_sel = fusion.fit_transform(feat_train, y_train)
    feat_val_sel = fusion.transform(feat_val)

    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced")
    model.fit(feat_train_sel, y_train)

    val_prob = model.predict_proba(feat_val_sel)[:, 1]
    auc = roc_auc_score(y_val, val_prob)
    acc = accuracy_score(y_val, (val_prob >= 0.5).astype(int))
    print(f"\nWith event ({feat_train.shape[1]}d -> 128): acc={acc:.4f}  AUC={auc:.4f}")

    # 对比：不用 event 特征（同一批样本，切掉末尾事件维度，mRMR 重新选）
    feat_train_no_ev = feat_train[:, :532]
    feat_val_no_ev = feat_val[:, :532]
    fusion2 = FeatureFusion(n_select=128)
    feat_train_sel2 = fusion2.fit_transform(feat_train_no_ev, y_train)
    feat_val_sel2 = fusion2.transform(feat_val_no_ev)
    model2 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced")
    model2.fit(feat_train_sel2, y_train)
    val_prob2 = model2.predict_proba(feat_val_sel2)[:, 1]
    auc2 = roc_auc_score(y_val, val_prob2)
    acc2 = accuracy_score(y_val, (val_prob2 >= 0.5).astype(int))
    print(f"Without event (532d -> 128): acc={acc2:.4f}  AUC={auc2:.4f}")
    print(f"Delta: AUC {auc-auc2:+.4f}  acc {acc-acc2:+.4f}")

    event_start = dims.get("time_domain", 228) + dims.get("freq_domain", 152) + dims.get("timefreq", 152)
    n_event_sel = sum(1 for s in fusion.selected_ if event_start <= s < event_start + dims.get("event", 82))
    print(f"mRMR: {n_event_sel}/{dims.get('event', 82)} event features selected")

    report = {
        "protocol": "quick ab: 5000 train / 2000 val epochs, 5 epochs/record, LightGBM(128 mRMR)",
        "dims": dims,
        "with_event": {"acc": round(float(acc), 4), "auc": round(float(auc), 4)},
        "without_event": {"acc": round(float(acc2), 4), "auc": round(float(auc2), 4)},
        "delta": {"acc": round(float(acc - acc2), 4), "auc": round(float(auc - auc2), 4)},
        "mrmr_event_selected": f"{n_event_sel}/{dims.get('event', 82)}",
    }
    out = resolve_path("outputs/event_feature_ab.json")
    save_json(report, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
