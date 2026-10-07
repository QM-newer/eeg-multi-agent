# training/common.py
"""
训练公共工具：数据准备、模型构建、PyTorch 训练循环、评估与保存

真实数据接入后，只需替换 prepare_dataset() 中的数据源（目前用 data.simulator），
下游训练脚本无需改动。
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    MODEL_CONFIG, MODEL_PATHS, N_CLASSES, N_TIMES, NUM_WORKERS,
    DATA_CONFIG, RANDOM_SEED, TRAIN_CONFIG, resolve_path,
    BINARY_LABEL_MAP, BINARY_MODEL_PATHS,
)
from data.dataset import EEGDataset, split_by_subject
from data.epoch_cache import compute_class_weights
from data.simulator import generate_dataset
from models.eegnet import EEGNet
from models.tcn import TCN
from models.eeg_transformer import EEGTransformer
from utils.io_utils import ensure_dir, save_json


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed=None):
    seed = RANDOM_SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ============================================================
# 数据准备
# ============================================================
def cache_available():
    """epoch 缓存是否已构建（scripts/build_epoch_cache.py 的产物）"""
    path = os.path.join(DATA_CONFIG["epoch_cache_dir"], "manifest.csv")
    return os.path.exists(path)


def remap_labels_binary(y):
    """将三分类标签映射为二分类：Normal(0)→0, Borderline(1)→1, Abnormal(2)→1"""
    y = np.asarray(y)
    return np.array([BINARY_LABEL_MAP[int(v)] for v in y], dtype=np.int64)


def prepare_dataset(n_subjects=90, epochs_per_subject=5, seed=None, use_cache=None,
                    max_epochs_per_record=None, normalize="per_channel", verbose=False,
                    label_mode="3class"):
    """
    准备数据集：优先加载真实 epoch 缓存，否则回退到模拟数据

    真实数据路径（阶段 2 之后）：
        - 直接沿用 record_index.csv 里按受试者做好的 train/val/test 划分（不会泄露）
        - 逐通道 z-score 的统计量只用训练集估计

    参数:
        use_cache: None=自动（有缓存就用），True=强制用缓存，False=强制用模拟数据
        max_epochs_per_record: 每条记录最多取多少 epoch（控制内存与训练时长）
        normalize: "per_channel" / None
        label_mode: "3class" (原始三分类) / "binary" (Normal vs Abnormal, Borderline→Abnormal)

    返回:
        dict: {X, y, subject_ids, train_idx, val_idx, test_idx, source, ...}
    """
    seed = RANDOM_SEED if seed is None else seed
    if use_cache is None:
        use_cache = cache_available()

    if use_cache:
        from data.epoch_cache import EpochCache

        cache = EpochCache()
        if max_epochs_per_record is None:
            max_epochs_per_record = DATA_CONFIG.get("max_epochs_per_record")
        if verbose:
            print(cache.summary())
            print(f"  每条记录最多取 {max_epochs_per_record} 个 epoch")

        mean, std = cache.fit_normalizer("train", max_epochs_per_record)
        from data.epoch_cache import save_norm_stats
        norm_path = save_norm_stats(mean, std)   # 供诊断Agent推理时复用同一变换
        Xs, ys, sids, bounds = [], [], [], []
        for sp in ("train", "val", "test"):
            X_sp, y_sp, sid_sp, _ = cache.load_split(
                sp, max_epochs_per_record=max_epochs_per_record,
                normalize=normalize, norm_stats=(mean, std), verbose=verbose)
            start = sum(len(y) for y in ys)
            bounds.append((start, start + len(y_sp)))
            Xs.append(X_sp)
            ys.append(y_sp)
            sids.append(sid_sp)
        X = np.concatenate(Xs) if Xs else np.zeros((0, DATA_CONFIG["n_eeg_channels"], N_TIMES),
                                                   dtype=np.float32)
        y = np.concatenate(ys) if ys else np.zeros(0, np.int64)
        subject_ids = np.concatenate(sids) if sids else np.zeros(0)
        # 二分类模式：Borderline(1) → Abnormal(1)
        if label_mode == "binary":
            y = remap_labels_binary(y)
        train_idx = np.arange(bounds[0][0], bounds[0][1])
        val_idx = np.arange(bounds[1][0], bounds[1][1])
        test_idx = np.arange(bounds[2][0], bounds[2][1])
        return {
            "X": X, "y": y, "subject_ids": subject_ids,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "source": "epoch_cache", "cache": cache, "norm_stats": (mean, std),
            "norm_path": norm_path, "label_mode": label_mode,
        }

    # ---- 回退：模拟数据 ----
    X, y, subject_ids = generate_dataset(
        n_subjects=n_subjects,
        epochs_per_subject=epochs_per_subject,
        n_channels=DATA_CONFIG["n_eeg_channels"],
        n_times=N_TIMES,
        sfreq=DATA_CONFIG["target_sfreq"],
        seed=seed,
    )
    train_idx, val_idx, test_idx = split_by_subject(
        subject_ids, y,
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        seed=seed,
    )
    return {
        "X": X, "y": y, "subject_ids": subject_ids,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "source": "simulator",
    }


def make_loaders(X, y, train_idx, val_idx, batch_size=None, num_workers=None):
    """构建训练/验证 DataLoader"""
    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    num_workers = NUM_WORKERS if num_workers is None else num_workers

    train_set = EEGDataset(X[train_idx], y[train_idx])
    val_set = EEGDataset(X[val_idx], y[val_idx])

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return (
        DataLoader(train_set, shuffle=TRAIN_CONFIG["shuffle"], **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
    )


# ============================================================
# 模型构建（超参统一来自 config.MODEL_CONFIG）
# ============================================================
def build_model(name, n_classes=None, n_channels=None):
    """按 config.MODEL_CONFIG 构建模型，保证与诊断Agent推理时结构一致"""
    n_classes = n_classes or N_CLASSES
    n_channels = n_channels or DATA_CONFIG["n_eeg_channels"]

    if name == "eegnet":
        cfg = MODEL_CONFIG["eegnet"]
        return EEGNet(
            n_classes=n_classes, n_channels=n_channels, n_times=N_TIMES,
            dropout_rate=cfg["dropout"], filters=cfg["filters"],
            kernel_time=cfg["kernel_time"], pool=(cfg["pool"], cfg["pool2"]),
        )
    if name == "tcn":
        cfg = MODEL_CONFIG["tcn"]
        return TCN(
            n_classes=n_classes, n_channels=n_channels,
            num_layers=cfg["n_layers"], hidden_channels=cfg["hidden_channels"],
            kernel_size=cfg["kernel_size"], dropout=cfg["dropout"],
            dilation_base=cfg["dilation_base"],
        )
    if name == "transformer":
        cfg = MODEL_CONFIG["transformer"]
        return EEGTransformer(
            n_classes=n_classes, n_channels=n_channels, n_times=N_TIMES,
            patch_size=cfg["patch_size"], d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        )
    raise ValueError(f"未知模型: {name}")


# ============================================================
# PyTorch 训练 / 评估
# ============================================================
def _run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).long()
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum())
            total += yb.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_torch_model(model, train_loader, val_loader, device="cpu", epochs=30,
                      lr=None, weight_decay=None, patience=None, save_path=None,
                      verbose=True, class_weights=None):
    """
    通用 PyTorch 训练循环（Adam + ReduceLROnPlateau + 早停 + 保存最优）

    参数:
        class_weights: 长度为 n_classes 的数组，用于 CrossEntropyLoss 加权
                       （本数据集 Borderline 仅约 10%，必须做类别平衡）

    返回:
        dict: {best_val_acc, final_val_acc, history, save_path}
    """
    lr = lr or TRAIN_CONFIG["lr"]
    weight_decay = weight_decay if weight_decay is not None else TRAIN_CONFIG["weight_decay"]
    patience = patience or TRAIN_CONFIG["early_stop_patience"]

    model = model.to(device)
    if class_weights is not None:
        w = torch.FloatTensor(np.asarray(class_weights, dtype=np.float32)).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=5)

    best_acc, best_state, no_improve = 0.0, None, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer,
                                           device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer,
                                       device, train=False)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"    早停触发（{patience} 轮无提升），停止于 Epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        torch.save(model.state_dict(), save_path)

    return {
        "best_val_acc": float(best_acc),
        "final_val_acc": float(history["val_acc"][-1]) if history["val_acc"] else 0.0,
        "history": history,
        "save_path": save_path,
    }


@torch.no_grad()
def predict_torch(model, X, device="cpu", batch_size=64):
    """批量推理，返回概率矩阵 (n_samples, n_classes)"""
    model = model.to(device).eval()
    probs = []
    for start in range(0, len(X), batch_size):
        xb = torch.FloatTensor(np.asarray(X[start:start + batch_size])).to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


# ============================================================
# 特征层（供 LightGBM 训练 / 融合）
# ============================================================
def _make_feature_agents(sfreq, include_event=False):
    from agents.feature_agents.time_domain_agent import TimeDomainAgent
    from agents.feature_agents.freq_domain_agent import FreqDomainAgent
    from agents.feature_agents.timefreq_agent import TimeFreqAgent

    agents = [
        TimeDomainAgent("time_domain"),
        FreqDomainAgent("freq_domain", sfreq=sfreq),
        TimeFreqAgent("timefreq"),
    ]
    if include_event:
        # 事件级特征（棘波/尖波形态学指标，82 维）追加在拼接末尾
        from agents.feature_agents.event_agent import EventAgent
        agents.append(EventAgent("event", sfreq=sfreq))
    return agents


def _extract_chunk(args):
    """子进程worker：对一小批样本提取多域特征（模块级函数，保证可 pickle）"""
    X_chunk, sfreq, include_event = args
    agents = _make_feature_agents(sfreq, include_event=include_event)
    rows = []
    for i in range(len(X_chunk)):
        sample = np.asarray(X_chunk[i], dtype=np.float64)
        rows.append(np.hstack([
            np.asarray(agent.predict(sample)["extra"]["features"]).reshape(-1)
            for agent in agents
        ]))
    return np.vstack(rows) if rows else np.zeros((0, 0))


def extract_multi_domain_features(X, sfreq=None, n_jobs=None, verbose=False,
                                  include_event=False):
    """
    用特征Agent逐样本提取多域特征并拼接

    单个样本约 110ms，上万样本必须并行，否则耗时以小时计。

    参数:
        X: (n_samples, n_channels, n_times)
        n_jobs: 并行进程数；None=自动（CPU 核数的 1/4，最多 8），1=串行
        include_event: True 时追加事件级特征Agent（82 维，拼接到末尾）

    返回:
        (features, agent_dims)
            features: (n_samples, n_features)
            agent_dims: dict，各Agent贡献的维度
    """
    sfreq = sfreq or DATA_CONFIG["target_sfreq"]
    X = np.asarray(X)
    n = len(X)
    if n == 0:
        return np.zeros((0, 0)), {}

    agents = _make_feature_agents(sfreq, include_event=include_event)
    agents[0].predict(np.asarray(X[0], dtype=np.float64))   # 触发一次，拿各Agent维度
    dims = {}
    sample0 = np.asarray(X[0], dtype=np.float64)
    for a in agents:
        dims[a.name] = int(np.asarray(a.predict(sample0)["extra"]["features"]).size)

    if n_jobs is None:
        n_jobs = max(1, min(8, (os.cpu_count() or 2) // 4))

    if n_jobs > 1 and n >= 200:
        chunk_size = max(50, int(np.ceil(n / (n_jobs * 4))))
        chunks = [(X[s:s + chunk_size], sfreq, include_event)
                  for s in range(0, n, chunk_size)]
        try:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(n_jobs) as pool:
                parts = pool.map(_extract_chunk, chunks)
            feats = np.vstack(parts)
            if verbose:
                print(f"    并行提取 {n} 样本（{n_jobs} 进程，{len(chunks)} 块）")
            return feats, dims
        except Exception as e:       # 并行失败则退回串行
            if verbose:
                print(f"    并行提取失败（{type(e).__name__}: {e}），退回串行")

    feats_per_agent = [[] for _ in agents]
    for i in range(n):
        sample = np.asarray(X[i], dtype=np.float64)
        for k, agent in enumerate(agents):
            res = agent.predict(sample)
            feats_per_agent[k].append(np.asarray(res["extra"]["features"]).reshape(-1))

    stacked = [np.vstack(f) for f in feats_per_agent]
    agent_dims = {agent.name: int(s.shape[1]) for agent, s in zip(agents, stacked)}
    return np.hstack(stacked), agent_dims


# ============================================================
# 训练结果记录（动态权重回写）
# ============================================================
def train_deep_agent(name, n_subjects=90, epochs_per_subject=5, epochs=30,
                     batch_size=None, device=None, seed=None, verbose=True,
                     max_epochs_per_record=None, label_mode="3class"):
    """
    深度学习诊断Agent 的统一训练入口（EEGNet / TCN / Transformer 共用）

    参数:
        label_mode: "3class" 或 "binary"（Normal vs Abnormal）

    返回:
        float: 验证集准确率
    """
    set_seed(seed)
    device = device or TRAIN_CONFIG["device"]

    n_classes = 2 if label_mode == "binary" else N_CLASSES
    model_paths = BINARY_MODEL_PATHS if label_mode == "binary" else MODEL_PATHS

    if verbose:
        print("=" * 60)
        print(f"训练 {name.upper()} 诊断Agent ({label_mode}, {n_classes}分类)")
        print("=" * 60)
        print("\n[1/3] 准备数据...")

    data = prepare_dataset(n_subjects=n_subjects, epochs_per_subject=epochs_per_subject,
                           max_epochs_per_record=max_epochs_per_record, verbose=verbose,
                           label_mode=label_mode)
    X, y = data["X"], data["y"]
    if verbose:
        print(f"  数据来源: {data.get('source')}")
        print(f"  样本: {len(X)}  形状: {X.shape[1:]}  "
              f"train/val/test = {len(data['train_idx'])}/{len(data['val_idx'])}/{len(data['test_idx'])}")

    class_weights = compute_class_weights(y[data["train_idx"]], n_classes=n_classes)
    if verbose:
        print(f"  类别权重: {np.round(class_weights, 3)}")

    train_loader, val_loader = make_loaders(
        X, y, data["train_idx"], data["val_idx"], batch_size=batch_size
    )

    if verbose:
        print("\n[2/3] 构建模型并训练...")
    model = build_model(name, n_classes=n_classes)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  模型参数量: {n_params:,}  设备: {device}  epochs: {epochs}")

    result = train_torch_model(
        model, train_loader, val_loader, device=device, epochs=epochs,
        save_path=model_paths[name], verbose=verbose, class_weights=class_weights,
    )

    if verbose:
        print(f"\n[3/3] 最优验证准确率: {result['best_val_acc']:.4f}")
        print(f"  权重已保存: {result['save_path']}")

    save_val_accuracies({name: result["best_val_acc"]})
    return float(result["best_val_acc"])


def save_val_accuracies(accuracies, path=None):
    """保存各Agent验证集准确率（与已有记录合并，供权重回写使用）"""
    path = path or resolve_path("checkpoints/val_accuracies.json")
    ensure_dir(os.path.dirname(path))

    merged = {}
    if os.path.exists(path):
        try:
            from utils.io_utils import load_json
            merged = load_json(path) or {}
        except Exception:
            merged = {}
    merged.update({k: round(float(v), 6) for k, v in accuracies.items()})

    save_json(merged, path)
    return path


def load_val_accuracies(path=None):
    """读取已记录的验证集准确率"""
    path = path or resolve_path("checkpoints/val_accuracies.json")
    if not os.path.exists(path):
        return {}
    from utils.io_utils import load_json
    return load_json(path) or {}


def accuracies_to_weights(accuracies, min_weight=0.05, max_weight=0.60, floor=None):
    """
    按验证集准确率归一化得到投票权重（文档 4.2(1)：w_i 由 Acc_i 归一化）

    参数:
        accuracies: dict，{agent_name: accuracy}
        min_weight / max_weight: 权重裁剪边界
        floor: 基线准确率。默认 1/n_classes（随机猜测）。
            类别不平衡时建议传入"多数类基线"（例如 0.55），
            否则"总猜多数类"的退化模型也会拿到很高权重，融合反而被拖累。

    返回:
        dict: 归一化后的权重
    """
    acc = np.array([max(float(v), 0.0) for v in accuracies.values()], dtype=np.float64)
    # 用"超出基线的部分"做权重，避免随机水平/全猜多数类的模型拿到过高权重
    floor = 1.0 / max(N_CLASSES, 1) if floor is None else float(floor)
    margin = np.clip(acc - floor, 0.0, None)

    if margin.sum() <= 1e-8:        # 全部在随机水平附近 → 退化为均匀权重
        margin = np.ones_like(acc)

    weights = margin / margin.sum()
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.sum()
    return {k: float(w) for k, w in zip(accuracies.keys(), weights)}
