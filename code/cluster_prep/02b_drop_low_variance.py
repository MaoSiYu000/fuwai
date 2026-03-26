# -*- coding: utf-8 -*-
"""
Step 02b：删除零方差 / 近零方差特征（缩放后）。

它做什么：
- 读取 output/cluster_prep/02_features_scaled.csv
- 计算每个数值特征的方差（排除键列与 is_missing_*）
- 删除方差过小（几乎恒定、没有区分度）的特征

为什么要做：
- “零方差/近零方差”特征对聚类没有贡献（所有人都一样）
- 但会干扰一些统计/可视化/降维诊断，让结果看起来更怪

输出：
- output/cluster_prep/02b_features_var_pruned.csv
- output/cluster_prep/02b_dropped_low_variance.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "02_features_scaled.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "02b_features_var_pruned.csv"
DROP_PATH = OUT_DIR / "02b_dropped_low_variance.csv"


# 对 RobustScaler 后的数据而言，“近零方差”一般非常小；这里取保守阈值
VAR_EPS = 1e-8


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 02_robustify_scale.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in df.columns if c not in key_cols + miss_cols]

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # 缩放后理论上已经没有缺失，但为了稳健，这里仍做一次中位数填补再算方差
    x = x.fillna(x.median(numeric_only=True))

    var = x.var(axis=0, ddof=0)
    to_drop = var[var <= VAR_EPS].index.tolist()

    pruned = df.drop(columns=to_drop, errors="ignore")
    pruned.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    dropped = (
        pd.DataFrame({"feature": var.index, "variance": var.values})
        .sort_values("variance", ascending=True)
        .assign(is_dropped=lambda d: d["feature"].isin(to_drop))
    )
    dropped.to_csv(DROP_PATH, index=False, encoding="utf-8-sig")

    print(
        "[完成] 低方差特征处理："
        f"drop={len(to_drop)} 输出={OUT_PATH}；清单={DROP_PATH}"
    )


if __name__ == "__main__":
    main()

