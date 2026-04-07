# -*- coding: utf-8 -*-
"""
软聚类（GMM）K 值评估：同时看拟合、稳定性与不确定性。

输入：
- output/cluster/dimension_scores_student_term.csv

输出：
- output/cluster/soft_k_evaluation_mode.csv
- output/cluster/soft_k_pairwise_details_mode.csv

说明：
- 这是 mode 层的评估（在 dim_* 维度分数空间上）。
- 不仅看 BIC/AIC，还看多随机种子下：
  1) 硬标签一致性（ARI/NMI）
  2) 软概率一致性（mean_l1/jsd）
  3) 不确定性分布（pmax/entropy）
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parents[2]
IN_DIM = ROOT / "output" / "cluster" / "dimension_scores_student_term.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "soft_k_evaluation_mode.csv"
OUT_PAIRWISE = OUT_DIR / "soft_k_pairwise_details_mode.csv"

K_LIST = list(range(4, 13))
SEEDS = [11, 22, 33, 44, 55]
N_INIT = 3
MAX_ITER = 500


@dataclass
class RunResult:
    seed: int
    k: int
    gmm: GaussianMixture
    labels: np.ndarray
    probs: np.ndarray
    bic: float
    aic: float
    sil: float
    ch: float
    dbi: float
    pmax_mean: float
    pmax_p10: float
    pmax_p50: float
    pmax_p90: float
    ent_mean: float
    ent_norm_mean: float


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _fit_one(x: np.ndarray, k: int, seed: int) -> RunResult:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=seed,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )
    gmm.fit(x)
    labels = gmm.predict(x)
    probs = gmm.predict_proba(x)

    bic = float(gmm.bic(x))
    aic = float(gmm.aic(x))
    sil = float(silhouette_score(x, labels)) if k > 1 else np.nan
    ch = float(calinski_harabasz_score(x, labels)) if k > 1 else np.nan
    dbi = float(davies_bouldin_score(x, labels)) if k > 1 else np.nan

    pmax = probs.max(axis=1)
    ent = _entropy(probs)
    ent_norm = ent / np.log(k)

    return RunResult(
        seed=seed,
        k=k,
        gmm=gmm,
        labels=labels,
        probs=probs,
        bic=bic,
        aic=aic,
        sil=sil,
        ch=ch,
        dbi=dbi,
        pmax_mean=float(np.mean(pmax)),
        pmax_p10=float(np.quantile(pmax, 0.10)),
        pmax_p50=float(np.quantile(pmax, 0.50)),
        pmax_p90=float(np.quantile(pmax, 0.90)),
        ent_mean=float(np.mean(ent)),
        ent_norm_mean=float(np.mean(ent_norm)),
    )


def _greedy_match(cost: np.ndarray) -> np.ndarray:
    k = cost.shape[0]
    mapping = -np.ones(k, dtype=int)
    used_cols: set[int] = set()
    for i in range(k):
        best_j = None
        best_v = float("inf")
        for j in range(k):
            if j in used_cols:
                continue
            v = float(cost[i, j])
            if v < best_v:
                best_v = v
                best_j = j
        mapping[i] = int(best_j)
        used_cols.add(int(best_j))
    return mapping


def _align_probs_by_means(base_means: np.ndarray, other_means: np.ndarray, other_probs: np.ndarray) -> np.ndarray:
    # cost[i, j] = base component i 与 other component j 的距离
    diff = base_means[:, None, :] - other_means[None, :, :]
    cost = np.sqrt((diff * diff).sum(axis=2))
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = np.zeros(len(row_ind), dtype=int)
        mapping[row_ind] = col_ind
    except Exception:
        mapping = _greedy_match(cost)
    return other_probs[:, mapping]


def _jsd_mean(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.mean(jsd))


def main() -> None:
    if not IN_DIM.exists():
        raise FileNotFoundError(f"未找到输入：{IN_DIM}")

    df = pd.read_csv(IN_DIM, encoding="utf-8-sig", low_memory=False)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("dimension_scores_student_term.csv 未找到 dim_* 列。")

    x_df = df[dim_cols].apply(pd.to_numeric, errors="coerce")
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x = x_df.to_numpy()

    summary_rows: list[dict] = []
    pair_rows: list[dict] = []

    for k in K_LIST:
        runs = [_fit_one(x, k, s) for s in SEEDS]

        for r1, r2 in combinations(runs, 2):
            ari = float(adjusted_rand_score(r1.labels, r2.labels))
            nmi = float(normalized_mutual_info_score(r1.labels, r2.labels))
            p2_align = _align_probs_by_means(r1.gmm.means_, r2.gmm.means_, r2.probs)
            l1 = float(np.mean(np.abs(r1.probs - p2_align)))
            jsd = _jsd_mean(r1.probs, p2_align)
            pair_rows.append(
                {
                    "k": k,
                    "seed_a": r1.seed,
                    "seed_b": r2.seed,
                    "ari": ari,
                    "nmi": nmi,
                    "prob_mean_l1": l1,
                    "prob_mean_jsd": jsd,
                }
            )

        pair_k = [r for r in pair_rows if r["k"] == k]
        summary_rows.append(
            {
                "k": k,
                "bic_mean": float(np.mean([r.bic for r in runs])),
                "bic_std": float(np.std([r.bic for r in runs], ddof=0)),
                "aic_mean": float(np.mean([r.aic for r in runs])),
                "silhouette_mean": float(np.mean([r.sil for r in runs])),
                "ch_mean": float(np.mean([r.ch for r in runs])),
                "dbi_mean": float(np.mean([r.dbi for r in runs])),
                "ari_pair_mean": float(np.mean([r["ari"] for r in pair_k])),
                "ari_pair_min": float(np.min([r["ari"] for r in pair_k])),
                "nmi_pair_mean": float(np.mean([r["nmi"] for r in pair_k])),
                "prob_l1_pair_mean": float(np.mean([r["prob_mean_l1"] for r in pair_k])),
                "prob_jsd_pair_mean": float(np.mean([r["prob_mean_jsd"] for r in pair_k])),
                "pmax_mean": float(np.mean([r.pmax_mean for r in runs])),
                "pmax_p10_mean": float(np.mean([r.pmax_p10 for r in runs])),
                "pmax_p50_mean": float(np.mean([r.pmax_p50 for r in runs])),
                "pmax_p90_mean": float(np.mean([r.pmax_p90 for r in runs])),
                "entropy_mean": float(np.mean([r.ent_mean for r in runs])),
                "entropy_norm_mean": float(np.mean([r.ent_norm_mean for r in runs])),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("k")
    pairwise = pd.DataFrame(pair_rows).sort_values(["k", "seed_a", "seed_b"])

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8-sig")

    print(f"[OK] 软聚类K评估汇总: {OUT_SUMMARY}")
    print(f"[OK] 软聚类K评估明细: {OUT_PAIRWISE}")


if __name__ == "__main__":
    main()

