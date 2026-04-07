# -*- coding: utf-8 -*-
"""
Step 02: 补齐影响评估（防止“补齐造成严重后果”）。

比较补齐前后每个数值特征的：
- 均值变化率
- 标准差变化率
- 中位数变化率
- P90 变化率

输出：
- output/data_impute/02_impute_impact_report.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_BEFORE = ROOT / "output" / "eda" / "features_student_term_draft.csv"
IN_AFTER = ROOT / "output" / "data_impute" / "01_features_imputed_safe.csv"
OUT_DIR = ROOT / "output" / "data_impute"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = OUT_DIR / "02_impute_impact_report.csv"

KEY_COLS = ["XH", "TERM_KEY"]
ALERT_THRESHOLD = 0.20  # 相对变化超 20% 给出告警


def _safe_rel_change(a: float, b: float) -> float:
    denom = abs(a) if abs(a) > 1e-12 else 1.0
    return float((b - a) / denom)


def main() -> None:
    if not IN_BEFORE.exists():
        raise FileNotFoundError(f"未找到输入文件：{IN_BEFORE}")
    if not IN_AFTER.exists():
        raise FileNotFoundError(f"未找到输入文件：{IN_AFTER}")

    before = pd.read_csv(IN_BEFORE, encoding="utf-8-sig", low_memory=False)
    after = pd.read_csv(IN_AFTER, encoding="utf-8-sig", low_memory=False)

    common_cols = [c for c in before.columns if c in after.columns and c not in KEY_COLS]
    rows = []
    for c in common_cols:
        b = pd.to_numeric(before[c], errors="coerce")
        a = pd.to_numeric(after[c], errors="coerce")
        if b.notna().sum() < 10 or a.notna().sum() < 10:
            continue

        b_mean, a_mean = float(b.mean(skipna=True)), float(a.mean(skipna=True))
        b_std, a_std = float(b.std(skipna=True, ddof=0)), float(a.std(skipna=True, ddof=0))
        b_med, a_med = float(b.median(skipna=True)), float(a.median(skipna=True))
        b_p90, a_p90 = float(b.quantile(0.90)), float(a.quantile(0.90))

        d_mean = _safe_rel_change(b_mean, a_mean)
        d_std = _safe_rel_change(b_std, a_std)
        d_med = _safe_rel_change(b_med, a_med)
        d_p90 = _safe_rel_change(b_p90, a_p90)

        max_abs_delta = max(abs(d_mean), abs(d_std), abs(d_med), abs(d_p90))
        rows.append(
            {
                "feature": c,
                "mean_rel_change": d_mean,
                "std_rel_change": d_std,
                "median_rel_change": d_med,
                "p90_rel_change": d_p90,
                "max_abs_rel_change": max_abs_delta,
                "alert_large_shift": int(max_abs_delta > ALERT_THRESHOLD),
            }
        )

    report = pd.DataFrame(rows).sort_values("max_abs_rel_change", ascending=False)
    report.to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")
    print(f"[OK] 补齐影响评估：{OUT_REPORT}")


if __name__ == "__main__":
    main()

