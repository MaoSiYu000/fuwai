# -*- coding: utf-8 -*-
"""
两层软聚类：模式层（mode）+ 模式内细分（subtype）。

你希望达到的效果：
- 先按“不同维度”分出清晰的 4+ 类模式（mode）
- 每个模式下面再细分子类（subtype），用于更细的解释与画像卡片
- 输出仍然是一张“最终结果表”，但包含 mode/subtype 以及对应概率

读取：
- output/cluster/dimension_scores_student_term.csv         （模式层输入：稳定维度分数）
- output/cluster_prep/00_features_frozen.csv               （子类输入：冻结原始特征，便于解释）

输出：
- output/cluster/student_term_modes_and_subtypes.csv
- output/cluster/mode_model_selection.csv
- output/cluster/subtype_model_selection.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_DIM = ROOT / "output" / "cluster" / "dimension_scores_student_term.csv"
IN_RAW = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MAIN = OUT_DIR / "student_term_modes_and_subtypes.csv"
OUT_MODE_SEL = OUT_DIR / "mode_model_selection.csv"
OUT_SUB_SEL = OUT_DIR / "subtype_model_selection.csv"


# 最终展示版口径：固定 mode=8（你确认“稳定性优先”后采用）
MODE_K_LIST = [8]
SUB_K_LIST = [2, 3, 4]
RANDOM_STATE = 42


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _fit_best_gmm(x: np.ndarray, k_list: list[int], n_init: int = 3) -> tuple[int, object, pd.DataFrame]:
    from sklearn.mixture import GaussianMixture

    rows = []
    best = None
    for k in k_list:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=RANDOM_STATE,
            n_init=n_init,
            max_iter=500,
        )
        gmm.fit(x)
        bic = float(gmm.bic(x))
        aic = float(gmm.aic(x))
        rows.append({"k": k, "bic": bic, "aic": aic})
        if best is None or bic < best["bic"]:
            best = {"k": k, "bic": bic, "model": gmm}
    assert best is not None
    return int(best["k"]), best["model"], pd.DataFrame(rows).sort_values("bic", ascending=True)


def main() -> None:
    if not IN_DIM.exists():
        raise FileNotFoundError(f"未找到输入：{IN_DIM}，请先运行 code/cluster/build_dimension_scores.py")
    if not IN_RAW.exists():
        raise FileNotFoundError(f"未找到输入：{IN_RAW}，请先运行 code/cluster_prep/run_all.py")

    dim = pd.read_csv(IN_DIM, encoding="utf-8-sig", low_memory=False)
    raw = pd.read_csv(IN_RAW, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in dim.columns and c in raw.columns]

    # 对齐（同一批学生-学期，按 key merge 保守对齐）
    base = dim.merge(raw, on=key_cols, how="inner", suffixes=("", "_raw"))

    dim_cols = [c for c in base.columns if c.startswith("dim_")]
    x_mode = base[dim_cols].apply(pd.to_numeric, errors="coerce")
    x_mode = x_mode.fillna(x_mode.median(numeric_only=True)).to_numpy()

    # 1) 模式层：按维度分数聚类（n_init=3，已通过稳定性评估，保持可解释解）
    mode_k, mode_model, mode_sel = _fit_best_gmm(x_mode, MODE_K_LIST, n_init=3)
    mode_sel.to_csv(OUT_MODE_SEL, index=False, encoding="utf-8-sig")

    p_mode = mode_model.predict_proba(x_mode)
    mode_id = p_mode.argmax(axis=1).astype(int)
    p_mode_sorted = np.sort(p_mode, axis=1)[:, ::-1]
    mode_pmax = p_mode_sorted[:, 0]
    mode_p2 = p_mode_sorted[:, 1]
    mode_margin = mode_pmax - mode_p2
    mode_ent = _entropy(p_mode)

    # 2) 子类层：在每个 mode 内，用原始特征再聚类
    raw_feat_cols = [c for c in raw.columns if c not in key_cols]
    # 仅用数值特征
    Xraw = base[raw_feat_cols].apply(pd.to_numeric, errors="coerce")
    Xraw = Xraw.fillna(Xraw.median(numeric_only=True))
    x_raw_all = Xraw.to_numpy()

    subtype_rows = []
    final_subtype = np.full(len(base), -1, dtype=int)
    final_sub_k = {}

    # 记录每个 mode 的子类选型
    sub_sel_rows = []

    for m in range(mode_k):
        idx = np.where(mode_id == m)[0]
        if len(idx) < 200:
            # 太小的模式先不细分，直接 subtype=0
            final_subtype[idx] = m * 100 + 0
            final_sub_k[m] = 1
            sub_sel_rows.append({"mode_id": m, "k": 1, "bic": None, "aic": None, "note": "too_small_skip"})
            continue

        x_sub = x_raw_all[idx]
        # 子类层也做一次简单标准化（避免某列支配）
        mu = np.nanmean(x_sub, axis=0)
        sd = np.nanstd(x_sub, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        x_sub_z = (np.nan_to_num(x_sub, nan=mu) - mu) / sd

        k_sub, sub_model, sub_sel = _fit_best_gmm(x_sub_z, SUB_K_LIST, n_init=10)
        sub_sel["mode_id"] = m
        sub_sel_rows.append({"mode_id": m, "k": k_sub, "bic": float(sub_sel.iloc[0]["bic"]), "aic": float(sub_sel.iloc[0]["aic"]), "note": "bic_best"})

        p_sub = sub_model.predict_proba(x_sub_z)
        sub_id_local = p_sub.argmax(axis=1).astype(int)
        final_subtype[idx] = m * 100 + sub_id_local
        final_sub_k[m] = k_sub

        # 保存子类概率（展开列名：p_subtype_<mode>_<j>）
        for j in range(k_sub):
            col = f"p_subtype_m{m}_{j}"
            if col not in base.columns:
                base[col] = np.nan
            base.loc[base.index[idx], col] = p_sub[:, j]

    pd.DataFrame(sub_sel_rows).to_csv(OUT_SUB_SEL, index=False, encoding="utf-8-sig")

    # 输出最终表
    out = base[key_cols + dim_cols].copy()
    out["mode_k"] = mode_k
    out["mode_id"] = mode_id
    for i in range(mode_k):
        out[f"p_mode_{i}"] = p_mode[:, i]
    out["mode_pmax"] = mode_pmax
    out["mode_margin"] = mode_margin
    out["mode_entropy"] = mode_ent

    out["subtype_id"] = final_subtype
    out["subtype_note"] = out["subtype_id"].map(lambda v: f"mode{int(v)//100}_sub{int(v)%100}" if v >= 0 else None)
    # 把子类概率列也带上（如果存在）
    sub_prob_cols = [c for c in base.columns if c.startswith("p_subtype_m")]
    if sub_prob_cols:
        out = pd.concat([out, base[sub_prob_cols]], axis=1)

    out.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")
    print(f"[完成] 两层软聚类输出：{OUT_MAIN}（mode_k={mode_k}）")
    print(f"[完成] mode 选型：{OUT_MODE_SEL}")
    print(f"[完成] subtype 选型：{OUT_SUB_SEL}")


if __name__ == "__main__":
    main()

