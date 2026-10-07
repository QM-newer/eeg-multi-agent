# evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, y_prob=None, average="weighted"):
    """
    计算分类评价指标

    参数:
        y_true: array, 真实标签
        y_pred: array, 预测标签
        y_prob: array, 预测概率 shape=(n_samples, n_classes)，可选
        average: str, 多分类平均方式，默认weighted

    返回:
        dict: 各项指标
    """
    metrics = {}

    # 基础分类指标
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # AUC（如果有概率）
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            metrics["auc"] = None

    # 混淆矩阵
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # 每类的指标
    metrics["per_class"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return metrics


def print_metrics(metrics, class_names=None):
    """
    美观打印评价指标
    """
    print("=" * 50)
    print("评价指标汇总")
    print("=" * 50)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1 值:              {metrics['f1']:.4f}")
    if metrics.get("auc"):
        print(f"AUC:                {metrics['auc']:.4f}")
    print()

    if class_names and "per_class" in metrics:
        print("各类别指标:")
        for i, name in enumerate(class_names):
            if str(i) in metrics["per_class"]:
                m = metrics["per_class"][str(i)]
                print(f"  {name:15s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}")
    print()

    if "confusion_matrix" in metrics:
        print("混淆矩阵:")
        for row in metrics["confusion_matrix"]:
            print(f"  {row}")
    print("=" * 50)
