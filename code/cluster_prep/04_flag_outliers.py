# -*- coding: utf-8 -*-
"""
Step 04：异常点标注（不删除，先标记）。

读取：
- output/cluster_prep/03_features_pruned.csv

做法：
- 对数值特征（不含 is_missing_*）按分位数阈值标注：
  - is_outlier_hi_<col>：> p99
  - is_outlier_lo_<col>：< p01
- 输出异常点占比汇总，帮助决定是否需要 HDBSCAN 或单独异常群体处理

输出：
- output/cluster_prep/04_features_outliers_flagged.csv
- output/cluster_prep/04_outlier_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "03_features_pruned.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "04_features_outliers_flagged.csv"
SUM_PATH = OUT_DIR / "04_outlier_summary.csv"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 03_drop_redundant.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in df.columns if c not in key_cols + miss_cols]

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    q01 = x.quantile(0.01)
    q99 = x.quantile(0.99)

    outlier_cols = []
    summary_rows = []
    for c in feat_cols:
        s = x[c]
        hi = (s > q99[c]).fillna(False).astype(int)
        lo = (s < q01[c]).fillna(False).astype(int)
        hi_name = f"is_outlier_hi_{c}"
        lo_name = f"is_outlier_lo_{c}"
        df[hi_name] = hi
        df[lo_name] = lo
        outlier_cols.extend([hi_name, lo_name])
        summary_rows.append(
            {
                "feature": c,
                "p01": float(q01[c]) if pd.notna(q01[c]) else None,
                "p99": float(q99[c]) if pd.notna(q99[c]) else None,
                "hi_rate": float(hi.mean()),
                "lo_rate": float(lo.mean()),
            }
        )

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(SUM_PATH, index=False, encoding="utf-8-sig")
    print(f"[完成] 异常点标注：{OUT_PATH}（新增{len(outlier_cols)}列）")


if __name__ == "__main__":
    main()

