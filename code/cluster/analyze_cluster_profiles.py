# -*- coding: utf-8 -*-
"""
簇画像分析：输出“每个簇最显著的高/低指标”和簇命名草案（可视化/报告用）。

为什么要做这个：
- 聚类本身只给 cluster_id/概率，但你最终要的是“群体画像”和“模式命名”
- 所以我们需要把每个簇相对总体“高在哪/低在哪”量化出来，便于写解释

读取：
- output/cluster/student_term_gmm_with_raw_features.csv
  - 含：XH、TERM_KEY、cluster_map、p_cluster_*、以及冻结原始特征列（可解释）

输出：
- output/cluster/cluster_top_features.csv
  - 每个簇 TopN 指标（按 |z| 排序）
- output/cluster/cluster_name_draft.csv
  - 基于 Top 指标的命名草案（后续可人工调整）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster" / "student_term_gmm_with_raw_features.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TOP = OUT_DIR / "cluster_top_features.csv"
OUT_NAME = OUT_DIR / "cluster_name_draft.csv"

TOP_N = 10
STD_EPS = 1e-9


def _safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce")
    return float(v.mean()) if v.notna().any() else float("nan")


def _safe_std(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce")
    if not v.notna().any():
        return float("nan")
    return float(v.std(ddof=0))


def _label_from_feature_and_sign(feat: str, z: float) -> str | None:
    hi = z > 0
    f = feat.lower()
    # 学业
    if f in {"kccj_mean", "jdcj_mean", "by1_mean"}:
        return "学业更强" if hi else "学业更弱"
    if f == "kccj_fail_rate":
        return "挂科风险更高" if hi else "挂科风险更低"
    # 出勤/活动
    if f == "att_present_rate":
        return "出勤更好" if hi else "出勤更差"
    if f == "att_absent_rate":
        return "缺勤更多" if hi else "缺勤更少"
    if f == "att_event_cnt":
        return "活动/签到更活跃" if hi else "活动/签到更少"
    # 作业
    if f == "hw_submit_cnt":
        return "作业提交更多" if hi else "作业提交更少"
    if f == "hw_duration_median":
        return "作业耗时更长" if hi else "作业耗时更短"
    if f == "hw_ungraded_rate":
        return "未批阅比例更高" if hi else "未批阅比例更低"
    # 线上
    if f == "online_bfb":
        return "线上更活跃" if hi else "线上更不活跃"
    # 奖学金/竞赛/四六级
    if f in {"sch_amt_sum_term", "sch_amt_sum"}:
        return "奖学金金额更高" if hi else "奖学金金额更低"
    if f in {"sch_cnt", "sch_cnt_term"}:
        return "奖学金次数更多" if hi else "奖学金次数更少"
    if f in {"comp_cnt", "comp_cnt_term"}:
        return "竞赛更活跃" if hi else "竞赛更少"
    if f == "cet_score_max":
        return "四六级更强" if hi else "四六级更弱"
    # 体测
    if f in {"fit3_zf_mean", "fit3_fhl_mean", "fit_sjcj_mean", "fit_hscj_mean"}:
        return "体能更强" if hi else "体能更弱"
    return None


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 code/cluster/run_gmm_student_term.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    if "cluster_map" not in df.columns:
        raise ValueError("输入缺少 cluster_map")

    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]
    prob_cols = [c for c in df.columns if c.startswith("p_cluster_")]
    other_cols = {"cluster_map", "p_max", "p_second", "margin", "entropy"}
    feat_cols = [c for c in df.columns if c not in set(key_cols) | set(prob_cols) | other_cols]

    # 只保留“可解释数值特征”
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    overall_mean = X.mean(axis=0, skipna=True)
    overall_std = X.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    overall_std = overall_std.fillna(0.0)
    overall_std_safe = overall_std.where(overall_std > STD_EPS, 1.0)

    # 簇规模
    cnt = df["cluster_map"].value_counts().sort_index()
    total = int(cnt.sum())

    rows = []
    name_rows = []
    for k, n_k in cnt.items():
        sub = df[df["cluster_map"] == k]
        Xk = sub[feat_cols].apply(pd.to_numeric, errors="coerce")
        mean_k = Xk.mean(axis=0, skipna=True)
        z = (mean_k - overall_mean) / overall_std_safe
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        top = z.abs().sort_values(ascending=False).head(TOP_N)
        top_feats = []
        for feat, absz in top.items():
            zz = float(z.loc[feat])
            direction = "高" if zz > 0 else "低"
            label = _label_from_feature_and_sign(feat, zz)
            rows.append(
                {
                    "cluster_id": int(k),
                    "cluster_size": int(n_k),
                    "cluster_pct": float(n_k) / total,
                    "feature": feat,
                    "z": zz,
                    "abs_z": float(absz),
                    "direction_vs_overall": direction,
                    "mean_cluster": float(mean_k.loc[feat]) if pd.notna(mean_k.loc[feat]) else None,
                    "mean_overall": float(overall_mean.loc[feat]) if pd.notna(overall_mean.loc[feat]) else None,
                    "std_overall": float(overall_std_safe.loc[feat]) if pd.notna(overall_std_safe.loc[feat]) else None,
                    "interpretation_hint": label,
                }
            )
            if label:
                top_feats.append(label)

        # 命名草案：取前几条“解释提示”，去重后拼接
        uniq = []
        for t in top_feats:
            if t not in uniq:
                uniq.append(t)
        if not uniq:
            draft = "混合型/需进一步解释"
        else:
            draft = "、".join(uniq[:3])
        name_rows.append(
            {
                "cluster_id": int(k),
                "cluster_size": int(n_k),
                "cluster_pct": float(n_k) / total,
                "name_draft": draft,
            }
        )

    out = pd.DataFrame(rows).sort_values(["cluster_id", "abs_z"], ascending=[True, False])
    out.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")
    pd.DataFrame(name_rows).sort_values("cluster_id").to_csv(OUT_NAME, index=False, encoding="utf-8-sig")

    print(f"[完成] 簇Top指标：{OUT_TOP}")
    print(f"[完成] 命名草案：{OUT_NAME}")


if __name__ == "__main__":
    main()

