# -*- coding: utf-8 -*-
"""
Step 03：冗余特征处理（相关性阈值）。

读取：
- output/cluster_prep/02_features_scaled.csv

做法：
- 计算 Spearman 相关矩阵
- 找到 |corr| > 0.90 的特征对
- 默认规则：删除“更像重复统计量”的列（例如 *_median 在存在 *_mean 时删除 median）
  - 这是保守启发式，可后续根据解释需要调整

输出：
- output/cluster_prep/03_features_pruned.csv
- output/cluster_prep/03_dropped_features.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "02_features_scaled.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "03_features_pruned.csv"
DROP_PATH = OUT_DIR / "03_dropped_features.csv"


THRESH = 0.90


def _prefer_drop(a: str, b: str) -> str:
    # 简单启发式：有 mean 则优先删 median；否则删更长尾的“计数”类？这里先只做 mean/median 规则
    a_l = a.lower()
    b_l = b.lower()
    if a_l.endswith("_median") and b_l.endswith("_mean"):
        return a
    if b_l.endswith("_median") and a_l.endswith("_mean"):
        return b
    # 默认删 b（保持确定性）
    return b


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 02_robustify_scale.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in df.columns if c not in key_cols + miss_cols]

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman", min_periods=500)

    to_drop = set()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            if pd.isna(v):
                continue
            if abs(float(v)) > THRESH:
                a, b = cols[i], cols[j]
                drop = _prefer_drop(a, b)
                to_drop.add(drop)
                pairs.append({"a": a, "b": b, "corr": float(v), "drop": drop})

    pruned = df.drop(columns=sorted(to_drop), errors="ignore")
    pruned.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(DROP_PATH, index=False, encoding="utf-8-sig")

    print(f"[完成] 冗余特征删除：drop={len(to_drop)} 输出={OUT_PATH}")


if __name__ == "__main__":
    main()

