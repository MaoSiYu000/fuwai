# -*- coding: utf-8 -*-
"""
Step 02c：方差均衡（防止少数特征“支配”距离 / PCA）。

背景（通俗）：
- 即使做了 RobustScaler，有些特征在缩放后仍可能波动特别大
- 距离型聚类（KMeans/GMM）和 PCA 都会被“波动最大”的特征牵着走

它做什么：
- 读取 output/cluster_prep/02b_features_var_pruned.csv
- 对数值特征（排除 key 与 is_missing_*）再做一次“单位方差缩放”：
  x2 = x / std
  - 这样每个特征的波动尺度更接近，避免单一指标主导

说明：
- 这里不动 is_missing_*（仍保持 0/1）
- 这是“仅用于聚类输入空间”的工程处理；画像解释仍建议回看原始/冻结特征值

输出：
- output/cluster_prep/02c_features_std_equalized.csv
- output/cluster_prep/02c_std_params.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "02b_features_var_pruned.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "02c_features_std_equalized.csv"
PARAMS_PATH = OUT_DIR / "02c_std_params.csv"


STD_EPS = 1e-12


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 02b_drop_low_variance.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in df.columns if c not in key_cols + miss_cols]

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))

    std = x.std(axis=0, ddof=0)
    std_safe = std.where(std > STD_EPS, 1.0)
    x2 = x / std_safe

    out = pd.concat([df[key_cols], x2, df[miss_cols]], axis=1)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    params = (
        pd.DataFrame({"feature": std.index, "std": std.values, "std_safe": std_safe.values})
        .sort_values("std", ascending=False)
    )
    params.to_csv(PARAMS_PATH, index=False, encoding="utf-8-sig")

    print(f"[完成] 方差均衡：{OUT_PATH}；参数={PARAMS_PATH}")


if __name__ == "__main__":
    main()

