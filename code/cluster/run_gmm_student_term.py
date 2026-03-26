# -*- coding: utf-8 -*-
"""
GMM 软聚类（学生-学期粒度）：输出每个样本属于各簇的概率。

读取：
- output/cluster_prep/04_features_outliers_flagged.csv  （用于聚类的“处理后特征空间”）
- output/cluster_prep/00_features_frozen.csv            （用于输出画像时的“原始/可解释特征值”）

做法：
- 用 GaussianMixture 对样本做软聚类，K 在 4..8 网格搜索
- 用 BIC 选一个默认最优 K，同时保存每个 K 的结果与指标表
- 输出每个学生-学期：
  - cluster_map（最大概率所属类）
  - p_cluster_0..p_cluster_{K-1}
  - p_max / margin / entropy（边界样本“不确定性”）

输出：
- output/cluster/gmm_model_selection.csv
- output/cluster/student_term_gmm_probs.csv
- output/cluster/student_term_gmm_with_raw_features.csv
- output/cluster/cluster_profile_raw_means.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_FEAT = ROOT / "output" / "cluster_prep" / "04_features_outliers_flagged.csv"
IN_RAW = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SEL = OUT_DIR / "gmm_model_selection.csv"
OUT_PROB = OUT_DIR / "student_term_gmm_probs.csv"
OUT_JOIN = OUT_DIR / "student_term_gmm_with_raw_features.csv"
OUT_PROFILE = OUT_DIR / "cluster_profile_raw_means.csv"


K_LIST = list(range(4, 9))
RANDOM_STATE = 42


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def main() -> None:
    if not IN_FEAT.exists():
        raise FileNotFoundError(f"未找到输入：{IN_FEAT}，请先运行 code/cluster_prep/run_all.py")
    if not IN_RAW.exists():
        raise FileNotFoundError(f"未找到输入：{IN_RAW}，请先运行 code/cluster_prep/run_all.py")

    from sklearn.mixture import GaussianMixture

    df = pd.read_csv(IN_FEAT, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    if len(key_cols) < 2:
        raise ValueError("输入缺少 XH 或 TERM_KEY，无法作为学生-学期粒度输出。")

    miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    outlier_cols = [c for c in df.columns if c.startswith("is_outlier_")]
    feat_cols = [c for c in df.columns if c not in key_cols + outlier_cols]  # 保留 is_missing_ 参与聚类

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))

    selection_rows = []
    best = None

    for k in K_LIST:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=RANDOM_STATE,
            n_init=3,
            max_iter=500,
        )
        gmm.fit(x)
        bic = float(gmm.bic(x))
        aic = float(gmm.aic(x))
        ll = float(gmm.score(x) * len(x))  # total log-likelihood
        selection_rows.append({"k": k, "bic": bic, "aic": aic, "loglik": ll})
        if best is None or bic < best["bic"]:
            best = {"k": k, "bic": bic, "model": gmm}

    sel = pd.DataFrame(selection_rows).sort_values("bic", ascending=True)
    sel.to_csv(OUT_SEL, index=False, encoding="utf-8-sig")

    assert best is not None
    k = int(best["k"])
    gmm = best["model"]

    proba = gmm.predict_proba(x)  # (n, k)
    cluster_map = proba.argmax(axis=1).astype(int)

    # 不确定性指标
    p_sorted = np.sort(proba, axis=1)[:, ::-1]
    p_max = p_sorted[:, 0]
    p_second = p_sorted[:, 1] if k >= 2 else np.zeros_like(p_max)
    margin = p_max - p_second
    ent = np.array([_entropy(p) for p in proba], dtype=float)

    out = df[key_cols].copy()
    out["cluster_map"] = cluster_map
    for i in range(k):
        out[f"p_cluster_{i}"] = proba[:, i]
    out["p_max"] = p_max
    out["p_second"] = p_second
    out["margin"] = margin
    out["entropy"] = ent

    out.to_csv(OUT_PROB, index=False, encoding="utf-8-sig")

    raw = pd.read_csv(IN_RAW, encoding="utf-8-sig", low_memory=False)
    raw_key = [c for c in ["XH", "TERM_KEY"] if c in raw.columns]
    joined = out.merge(raw, on=raw_key, how="left", suffixes=("", "_raw"))
    joined.to_csv(OUT_JOIN, index=False, encoding="utf-8-sig")

    # 画像：用冻结的“原始特征”按簇求均值（便于解释/可视化）
    raw_feat_cols = [c for c in raw.columns if c not in raw_key]
    raw_x = raw[raw_feat_cols].apply(pd.to_numeric, errors="coerce")
    prof = (
        pd.concat([out[["cluster_map"]], raw_x], axis=1)
        .groupby("cluster_map", as_index=False)
        .agg(["mean", "median", "count"])
    )
    # 展平多级列名
    prof.columns = [
        (c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        if isinstance(c, tuple)
        else str(c)
        for c in prof.columns
    ]
    prof.to_csv(OUT_PROFILE, index=False, encoding="utf-8-sig")

    print(f"[完成] GMM 选型表：{OUT_SEL}")
    print(f"[完成] 概率输出：{OUT_PROB}（K={k}）")
    print(f"[完成] 概率+原始特征：{OUT_JOIN}")
    print(f"[完成] 簇画像（原始均值/中位数/计数）：{OUT_PROFILE}")


if __name__ == "__main__":
    main()

