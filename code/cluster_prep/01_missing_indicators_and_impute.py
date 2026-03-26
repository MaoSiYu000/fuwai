# -*- coding: utf-8 -*-
"""
Step 01：缺失处理（缺失指示 + 保守填补）。

读取：
- output/cluster_prep/00_features_frozen.csv

做法：
- 对除 XH、TERM_KEY 外的数值列：
  - 生成 is_missing_<col> 指示（1=缺失，0=非缺失）
  - 用该列中位数填补缺失（保守、稳定）

输出：
- output/cluster_prep/01_features_imputed.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "01_features_imputed.csv"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 00_freeze_input.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in key_cols]

    # 尝试把特征列转数值（无法转的会变 NaN）
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    miss_ind = {}
    for c in x.columns:
        miss_ind[f"is_missing_{c}"] = x[c].isna().astype(int)

    med = x.median(numeric_only=True)
    x_filled = x.fillna(med)

    out = pd.concat([df[key_cols], x_filled, pd.DataFrame(miss_ind)], axis=1)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    miss_rate = float(x.isna().mean().mean()) if len(x.columns) else 0.0
    print(f"[完成] 缺失指示+中位数填补：{OUT_PATH}（rows={len(out)} cols={len(out.columns)} avg_missing_rate={miss_rate:.3f}）")


if __name__ == "__main__":
    main()

