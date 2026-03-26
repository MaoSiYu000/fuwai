# -*- coding: utf-8 -*-
"""
Step 02：稳健化与缩放（winsorize + robust scaler）。

读取：
- output/cluster_prep/01_features_imputed.csv

做法（保守默认）：
- 对数值特征（不含 is_missing_*）做 winsorize：截断到 [p01, p99]
- 对截断后的数值特征做 RobustScaler（中位数中心化 / IQR 缩放）
- 缺失指示列保持 0/1 不缩放

输出：
- output/cluster_prep/02_features_scaled.csv
- output/cluster_prep/02_scale_params.json（分位数与缩放参数）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "01_features_imputed.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "02_features_scaled.csv"
PARAMS_PATH = OUT_DIR / "02_scale_params.json"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 01_missing_indicators_and_impute.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]

    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in df.columns if c not in key_cols + miss_cols]

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # winsorize p01-p99
    q_low = x.quantile(0.01)
    q_high = x.quantile(0.99)
    x_clip = x.clip(lower=q_low, upper=q_high, axis=1)

    # RobustScaler： (x - median) / IQR
    med = x_clip.median()
    q1 = x_clip.quantile(0.25)
    q3 = x_clip.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)
    x_scaled = (x_clip - med) / iqr

    out = pd.concat([df[key_cols], x_scaled, df[miss_cols]], axis=1)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    params = {
        "winsorize": {"p_low": 0.01, "p_high": 0.99},
        "q_low": q_low.to_dict(),
        "q_high": q_high.to_dict(),
        "robust_scaler": {"median": med.to_dict(), "iqr": iqr.to_dict()},
        "note": "只对非is_missing_的数值特征做截尾与缩放；缺失指示列保持0/1。",
    }
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"[完成] winsorize+robust scaling：{OUT_PATH}（rows={len(out)} cols={len(out.columns)}）")


if __name__ == "__main__":
    main()

