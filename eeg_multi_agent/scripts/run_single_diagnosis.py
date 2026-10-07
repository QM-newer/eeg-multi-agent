# scripts/run_single_diagnosis.py
"""
单样本诊断演示脚本
用法：python scripts/run_single_diagnosis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from orchestrator.orchestrator import Orchestrator
from config import CLASS_NAMES


def main():
    print("=" * 60)
    print("多Agent脑电诊断系统 - 单样本诊断演示")
    print("=" * 60)

    # 初始化系统
    print("\n[1/3] 正在初始化多Agent系统...")
    system = Orchestrator(device="cpu")
    print(f"  特征Agent: {len(system.feature_agents)} 个")
    print(f"  诊断Agent: {len(system.diagnosis_agents)} 个")
    print(f"  验证Agent: {len(system.validation_agents)} 个")
    print("  系统初始化完成！")

    # 生成模拟EEG数据
    print("\n[2/3] 生成模拟EEG数据...")
    n_channels = 35
    n_times = 1000
    sfreq = 250
    duration = n_times / sfreq
    eeg_data = np.random.randn(n_channels, n_times) * 10  # 模拟μV级脑电信号
    print(f"  通道数: {n_channels}")
    print(f"  采样点数: {n_times}")
    print(f"  时长: {duration:.2f} 秒 (采样率 {sfreq} Hz)")

    # 运行诊断
    print("\n[3/3] 运行多Agent诊断...")
    result = system.diagnose(eeg_data)

    # 输出诊断报告
    print()
    print("=" * 60)
    print("诊 断 报 告")
    print("=" * 60)
    qr = result["quality_report"]
    print(f"最终诊断:  {CLASS_NAMES[result['final_label']]}")
    print(f"置信度:    {result['confidence']:.4f}")
    print(f"信号质量:  {result['quality_score']:.2f} "
          f"(SNR={qr['snr_db']:.1f}dB, 伪迹={qr['artifact_ratio']:.2%}, "
          f"等级={qr['quality_level']})")
    print(f"是否分歧:  {'是' if result['disagreement'] else '否'}")
    print(f"二次诊断:  {'已触发' if result['second_opinion_triggered'] else '未触发'}")

    print("\n动态权重（SNR感知调整后）:")
    for name, w in result["weights"].items():
        print(f"  {name:15s}: {w:.4f}")

    if result["arbitration"]:
        arb = result["arbitration"]
        print(f"\n仲裁信息: 空间域补充特征 {arb['spatial_dim']} 维 "
              f"(扩充后总特征 {arb['augmented_dim']} 维)")
        print("  特征可信度:")
        for name, c in arb["credibility"].items():
            print(f"    {name:15s}: {c:.4f}")
    print(f"一致性:    {result['consistency']['consistency_score']:.4f} "
          f"({result['consistency']['consistency_level']})")
    print(f"人工复核:  {'建议' if result['consistency']['need_human_review'] else '不需要'}")
    print()

    print("三类概率分布:")
    for name, prob in zip(CLASS_NAMES, result["final_prob"]):
        bar = "█" * int(prob * 30)
        print(f"  {name:15s}: {prob:.4f}  {bar}")
    print()

    print("各Agent诊断结果:")
    print("-" * 60)
    for r in result["agent_results"]:
        prob_str = "  ".join([f"{p:.3f}" for p in r["prob"]])
        print(f"  {r['agent_name']:15s} | {CLASS_NAMES[r['label']]:12s} | {prob_str}")
    print("-" * 60)
    print()

    print("可解释性分析:")
    print(f"  {result['explanation']}")
    print()
    print("=" * 60)
    print("演示完成！")


if __name__ == "__main__":
    main()
