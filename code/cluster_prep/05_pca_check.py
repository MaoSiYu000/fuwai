# -*- coding: utf-8 -*-
"""
Step 05：PCA 检查（只用于诊断，不替代可解释特征）。

读取：
- output/cluster_prep/03_features_pruned.csv

做法：
- 对数值特征做 PCA，输出累计解释方差，用于判断是否被少数主成分支配

输出：
- output/cluster_prep/05_pca_explained_variance.csv
- output/cluster_prep/05_pca_top_loadings.csv（PC1/PC2 主要由哪些特征主导）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "03_features_pruned.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "05_pca_explained_variance.csv"
LOAD_PATH = OUT_DIR / "05_pca_top_loadings.csv"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 03_drop_redundant.py")

    from sklearn.decomposition import PCA

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in df.columns if c not in key_cols + miss_cols]

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    # 这里假设前面已完成填补与缩放；若未执行，也能跑但解释要谨慎
    x = x.fillna(x.median(numeric_only=True))

    pca = PCA(n_components=min(20, x.shape[1]), random_state=42)
    pca.fit(x)
    evr = pca.explained_variance_ratio_
    cum = evr.cumsum()
    out = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(evr))],
            "explained_variance_ratio": evr,
            "cum_explained_variance_ratio": cum,
        }
    )
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # 哪些特征主导 PC1/PC2（用 |loading| 排序，直观）
    comps = pca.components_  # (n_components, n_features)
    rows = []
    for k in [0, 1]:
        if k >= comps.shape[0]:
            continue
        pc_name = f"PC{k+1}"
        s = pd.Series(comps[k], index=feat_cols)
        top = s.abs().sort_values(ascending=False).head(20)
        for feat, abs_loading in top.items():
            rows.append(
                {
                    "component": pc_name,
                    "feature": feat,
                    "loading": float(s.loc[feat]),
                    "abs_loading": float(abs_loading),
                }
            )
    pd.DataFrame(rows).to_csv(LOAD_PATH, index=False, encoding="utf-8-sig")

    print(f"[完成] PCA检查：{OUT_PATH}")
    print(f"[完成] PCA主导来源：{LOAD_PATH}")


if __name__ == "__main__":
    main()

