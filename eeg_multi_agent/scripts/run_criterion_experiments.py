# scripts/run_criterion_experiments.py
"""
一键运行三个判据实验 + 标签源对比

执行顺序：
    ① 标签源对比（最优先，决定数据是否可信）
    ② 聚合方式 A/B（零成本，判聚合瓶颈）
    ③ 补特征 + 重训 LightGBM（补完已设计但未实现的 5 个特征）
    ④ 慢波子任务可行性（决定是否投事件级特征）

用法：
    python scripts/run_criterion_experiments.py
    python scripts/run_criterion_experiments.py --skip-retrain   # 跳过 LightGBM 重训
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import resolve_path
from utils.io_utils import ensure_dir, save_json


def run_experiment(name, script_path, args=""):
    """运行单个实验脚本"""
    import subprocess
    cmd = [sys.executable, script_path] + (args.split() if args else [])
    print(f"\n{'='*74}")
    print(f"  运行: {name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*74}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "✅ 成功" if result.returncode == 0 else f"❌ 退出码 {result.returncode}"
    print(f"\n  [{name}] {status}  耗时 {elapsed:.0f}s")
    return result.returncode == 0


def main(skip_retrain=False, n_per_class=60):
    base = os.path.dirname(os.path.abspath(__file__))
    results = {}

    # ========== ① 标签源对比 ==========
    ok = run_experiment(
        "① 标签源对比",
        os.path.join(base, "compare_label_sources.py"),
        "--split test"
    )
    results["label_source"] = "ok" if ok else "failed"

    # ========== ② 聚合方式 A/B ==========
    ok = run_experiment(
        "② 聚合方式 A/B (mean/max/p90)",
        os.path.join(base, "evaluate_subject_level.py"),
        "--split test --agg-methods mean,max,p90"
    )
    results["agg_ab"] = "ok" if ok else "failed"

    # ========== ③ 补特征 + 重训 LightGBM ==========
    if not skip_retrain:
        ok = run_experiment(
            "③ 补特征 + 重训 LightGBM",
            os.path.join(base, os.path.join(base, "..", "training", "train_lightgbm.py")),
        )
        results["retrain_lgbm"] = "ok" if ok else "failed"
    else:
        print("\n  [跳过] LightGBM 重训（--skip-retrain）")
        results["retrain_lgbm"] = "skipped"

    # ========== ④ 慢波子任务可行性 ==========
    ok = run_experiment(
        "④ 慢波子任务可行性",
        os.path.join(base, "test_slow_wave_subtask.py"),
        f"--n-per-class {n_per_class}"
    )
    results["slow_wave"] = "ok" if ok else "failed"

    # ========== 汇总 ==========
    print("\n" + "=" * 74)
    print("实验汇总")
    print("=" * 74)
    for name, status in results.items():
        print(f"  {name}: {status}")

    out = resolve_path("outputs/criterion_experiments_summary.json")
    ensure_dir(os.path.dirname(out))
    save_json(results, out)
    print(f"\n汇总已保存: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-retrain", action="store_true",
                    help="跳过 LightGBM 重训（特征已改但不想重训时使用）")
    ap.add_argument("--n-per-class", type=int, default=60,
                    help="慢波实验中 Normal/Abnormal 各抽取的记录数")
    a = ap.parse_args()
    main(skip_retrain=a.skip_retrain, n_per_class=a.n_per_class)
