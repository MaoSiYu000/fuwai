# -*- coding: utf-8 -*-
"""
补齐前后聚类对照实验（mode=8）。

目标：
- 在同样 mode=8 条件下，比较“基线数据”与“补齐数据”是否有实质提升。
- 输出结构化对照结果，便于写入文档。

输出：
- output/data_impute/compare/cluster_compare_summary.csv
- output/data_impute/compare/cluster_compare_metrics.csv
- output/data_impute/compare/imputed_student_term_modes_and_subtypes.csv
- output/data_impute/compare/imputed_soft_k_evaluation_mode.csv
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_COMPARE = ROOT / "output" / "data_impute" / "compare"
OUT_COMPARE.mkdir(parents=True, exist_ok=True)

FROZEN = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
IMPUTED = ROOT / "output" / "data_impute" / "01_features_imputed_safe.csv"

CLUSTER_MAIN = ROOT / "output" / "cluster" / "student_term_modes_and_subtypes.csv"
SOFT_EVAL = ROOT / "output" / "cluster" / "soft_k_evaluation_mode.csv"

BUILD_DIM = ROOT / "code" / "cluster" / "build_dimension_scores.py"
RUN_MODE = ROOT / "code" / "cluster" / "run_gmm_modes_and_subtypes.py"
RUN_STAB = ROOT / "code" / "cluster" / "evaluate_soft_clustering_stability.py"


def _run_py(path: Path) -> None:
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"脚本执行失败：{path}")


def _extract_metrics(tag: str, main_path: Path, soft_path: Path) -> pd.DataFrame:
    main = pd.read_csv(main_path, encoding="utf-8-sig", low_memory=False)
    soft = pd.read_csv(soft_path, encoding="utf-8-sig", low_memory=False)
    row8 = soft.loc[soft["k"] == 8]
    if row8.empty:
        raise ValueError("soft_k_evaluation_mode.csv 中未找到 k=8。")
    row8 = row8.iloc[0]

    mode_counts = main["mode_id"].value_counts(normalize=True)
    mode_min_pct = float(mode_counts.min()) if not mode_counts.empty else np.nan
    mode_max_pct = float(mode_counts.max()) if not mode_counts.empty else np.nan

    return pd.DataFrame(
        [
            {"scenario": tag, "metric": "mode_entropy_mean", "value": float(main["mode_entropy"].mean())},
            {"scenario": tag, "metric": "mode_pmax_mean", "value": float(main["mode_pmax"].mean())},
            {"scenario": tag, "metric": "mode_margin_mean", "value": float(main["mode_margin"].mean())},
            {"scenario": tag, "metric": "mode_size_min_pct", "value": mode_min_pct},
            {"scenario": tag, "metric": "mode_size_max_pct", "value": mode_max_pct},
            {"scenario": tag, "metric": "k8_ari_pair_mean", "value": float(row8["ari_pair_mean"])},
            {"scenario": tag, "metric": "k8_nmi_pair_mean", "value": float(row8["nmi_pair_mean"])},
            {"scenario": tag, "metric": "k8_prob_l1_pair_mean", "value": float(row8["prob_l1_pair_mean"])},
            {"scenario": tag, "metric": "k8_prob_jsd_pair_mean", "value": float(row8["prob_jsd_pair_mean"])},
            {"scenario": tag, "metric": "k8_bic_mean", "value": float(row8["bic_mean"])},
        ]
    )


def _build_summary(comp: pd.DataFrame) -> pd.DataFrame:
    piv = comp.pivot(index="metric", columns="scenario", values="value")
    rows = []
    for m in piv.index:
        b = float(piv.loc[m, "baseline"])
        i = float(piv.loc[m, "imputed"])
        delta = i - b

        # 指标方向（True: 越大越好；False: 越小越好）
        higher_better = m in {"mode_pmax_mean", "mode_margin_mean", "k8_ari_pair_mean", "k8_nmi_pair_mean"}
        if m in {"mode_entropy_mean", "k8_prob_l1_pair_mean", "k8_prob_jsd_pair_mean", "k8_bic_mean"}:
            higher_better = False
        if m in {"mode_size_min_pct"}:
            higher_better = True
        if m in {"mode_size_max_pct"}:
            higher_better = False

        improved = delta > 0 if higher_better else delta < 0
        rows.append(
            {
                "metric": m,
                "baseline": b,
                "imputed": i,
                "delta_imputed_minus_baseline": delta,
                "direction": "higher_better" if higher_better else "lower_better",
                "is_improved": int(improved),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not FROZEN.exists():
        raise FileNotFoundError(f"缺少文件：{FROZEN}")
    if not IMPUTED.exists():
        raise FileNotFoundError(f"缺少文件：{IMPUTED}")
    if not CLUSTER_MAIN.exists() or not SOFT_EVAL.exists():
        raise FileNotFoundError("请先完成 baseline 的聚类与软稳定性评估输出。")

    # 备份关键文件
    backup_frozen = OUT_COMPARE / "_backup_00_features_frozen.csv"
    backup_main = OUT_COMPARE / "_backup_student_term_modes_and_subtypes.csv"
    backup_soft = OUT_COMPARE / "_backup_soft_k_evaluation_mode.csv"
    shutil.copy2(FROZEN, backup_frozen)
    shutil.copy2(CLUSTER_MAIN, backup_main)
    shutil.copy2(SOFT_EVAL, backup_soft)

    baseline_metrics = _extract_metrics("baseline", CLUSTER_MAIN, SOFT_EVAL)

    try:
        # 切换到补齐输入 -> 重跑 mode=8 + 稳定性评估
        shutil.copy2(IMPUTED, FROZEN)
        _run_py(BUILD_DIM)
        _run_py(RUN_MODE)
        _run_py(RUN_STAB)

        # 保存补齐版关键输出
        shutil.copy2(CLUSTER_MAIN, OUT_COMPARE / "imputed_student_term_modes_and_subtypes.csv")
        shutil.copy2(SOFT_EVAL, OUT_COMPARE / "imputed_soft_k_evaluation_mode.csv")

        imputed_metrics = _extract_metrics("imputed", CLUSTER_MAIN, SOFT_EVAL)
        metrics = pd.concat([baseline_metrics, imputed_metrics], ignore_index=True)
        summary = _build_summary(metrics)

        metrics.to_csv(OUT_COMPARE / "cluster_compare_metrics.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUT_COMPARE / "cluster_compare_summary.csv", index=False, encoding="utf-8-sig")
        print(f"[OK] 对照明细：{OUT_COMPARE / 'cluster_compare_metrics.csv'}")
        print(f"[OK] 对照汇总：{OUT_COMPARE / 'cluster_compare_summary.csv'}")

    finally:
        # 恢复 baseline 状态
        shutil.copy2(backup_frozen, FROZEN)
        shutil.copy2(backup_main, CLUSTER_MAIN)
        shutil.copy2(backup_soft, SOFT_EVAL)
        print("[OK] 已恢复 baseline 文件状态。")


if __name__ == "__main__":
    main()

