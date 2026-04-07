# -*- coding: utf-8 -*-
"""
Step 00: 缺失审计（只统计，不改数据）。

输入：
- output/eda/features_student_term_draft.csv

输出：
- output/data_impute/00_missing_audit.csv
- output/data_impute/00_feature_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "eda" / "features_student_term_draft.csv"
OUT_DIR = ROOT / "output" / "data_impute"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_AUDIT = OUT_DIR / "00_missing_audit.csv"
OUT_SUMMARY = OUT_DIR / "00_feature_summary.csv"

KEY_COLS = ["XH", "TERM_KEY"]


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{IN_PATH}")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    feat_cols = [c for c in df.columns if c not in KEY_COLS]

    rows = []
    for c in feat_cols:
        s = df[c]
        s_num = pd.to_numeric(s, errors="coerce")
        rows.append(
            {
                "feature": c,
                "dtype": str(s.dtype),
                "n_total": int(len(s)),
                "n_missing": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "n_numeric_parseable": int(s_num.notna().sum()),
                "numeric_parse_rate": float(s_num.notna().mean()),
                "n_unique_nonnull": int(s.dropna().nunique()),
            }
        )

    audit = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "n_cols_total": int(df.shape[1]),
                "n_feature_cols": int(len(feat_cols)),
                "avg_missing_rate": float(audit["missing_rate"].mean() if not audit.empty else 0.0),
                "feature_missing_gt_40pct": int((audit["missing_rate"] > 0.40).sum() if not audit.empty else 0),
                "feature_missing_gt_60pct": int((audit["missing_rate"] > 0.60).sum() if not audit.empty else 0),
            }
        ]
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"[OK] 缺失审计：{OUT_AUDIT}")
    print(f"[OK] 审计汇总：{OUT_SUMMARY}")


if __name__ == "__main__":
    main()

