# -*- coding: utf-8 -*-
"""
为“两层软聚类（mode + subtype）”生成解释材料：
- 每个 mode 的维度分数画像（均值/占比）
- 每个 (mode, subtype) 的 Top 指标差异表（相对该 mode 的其他样本）
- 每个 (mode, subtype) 的命名草案

读取：
- output/cluster/student_term_modes_and_subtypes.csv   （含 mode_id/subtype_id + dim_*）
- output/cluster_prep/00_features_frozen.csv           （冻结原始特征，用于解释）

输出：
- output/cluster/mode_profile_dims.csv
- output/cluster/subtype_top_features.csv
- output/cluster/subtype_name_draft.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_MS = ROOT / "output" / "cluster" / "student_term_modes_and_subtypes.csv"
IN_RAW = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODE = OUT_DIR / "mode_profile_dims.csv"
OUT_TOP = OUT_DIR / "subtype_top_features.csv"
OUT_NAME = OUT_DIR / "subtype_name_draft.csv"

TOP_N = 10
STD_EPS = 1e-9


def _label_hint(feat: str, z: float) -> str | None:
    """给一个通俗标签提示（可用于自动命名），不追求全覆盖。"""
    hi = z > 0
    f = feat.lower()
    if f in {"kccj_mean", "jdcj_mean", "by1_mean"}:
        return "学业更强" if hi else "学业更弱"
    if f == "kccj_fail_rate":
        return "挂科风险更高" if hi else "挂科风险更低"
    if f == "att_present_rate":
        return "出勤更好" if hi else "出勤更差"
    if f == "att_absent_rate":
        return "缺勤更多" if hi else "缺勤更少"
    if f == "att_event_cnt":
        return "参与更活跃" if hi else "参与更少"
    if f == "hw_submit_cnt":
        return "作业提交更多" if hi else "作业提交更少"
    if f == "hw_ungraded_rate":
        return "未批阅比例更高" if hi else "未批阅比例更低"
    if f == "hw_duration_median":
        return "作业耗时更长" if hi else "作业耗时更短"
    if f == "online_bfb":
        return "线上更活跃" if hi else "线上更不活跃"
    if f in {"sch_amt_sum_term", "sch_amt_sum"}:
        return "奖学金金额更高" if hi else "奖学金金额更低"
    if f in {"comp_cnt_term", "comp_cnt"}:
        return "竞赛更活跃" if hi else "竞赛更少"
    if f == "cet_score_max":
        return "四六级更强" if hi else "四六级更弱"
    if f in {"fit3_zf_mean", "fit3_fhl_mean"}:
        return "体能更强" if hi else "体能更弱"
    return None


def main() -> None:
    if not IN_MS.exists():
        raise FileNotFoundError(f"未找到输入：{IN_MS}")
    if not IN_RAW.exists():
        raise FileNotFoundError(f"未找到输入：{IN_RAW}")

    ms = pd.read_csv(IN_MS, encoding="utf-8-sig", low_memory=False)
    raw = pd.read_csv(IN_RAW, encoding="utf-8-sig", low_memory=False)
    key = [c for c in ["XH", "TERM_KEY"] if c in ms.columns and c in raw.columns]

    df = ms.merge(raw, on=key, how="left")
    if "mode_id" not in df.columns or "subtype_id" not in df.columns:
        raise ValueError("输入缺少 mode_id/subtype_id")

    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    feat_cols = [c for c in raw.columns if c not in key]

    # 1) mode 维度画像
    mode_cnt = df["mode_id"].value_counts().sort_index()
    total = int(mode_cnt.sum())
    mode_rows = []
    for m, n in mode_cnt.items():
        g = df[df["mode_id"] == m]
        row = {"mode_id": int(m), "mode_size": int(n), "mode_pct": float(n) / total}
        for c in dim_cols:
            row[f"{c}_mean"] = float(pd.to_numeric(g[c], errors="coerce").mean())
        mode_rows.append(row)
    pd.DataFrame(mode_rows).to_csv(OUT_MODE, index=False, encoding="utf-8-sig")

    # 2) subtype Top 指标差异（相对该 mode 的其他样本）
    rows = []
    name_rows = []
    for m, g_m in df.groupby("mode_id"):
        g_m = g_m.copy()
        Xm = g_m[feat_cols].apply(pd.to_numeric, errors="coerce")
        overall_mean = Xm.mean(axis=0, skipna=True)
        overall_std = Xm.std(axis=0, ddof=0, skipna=True).fillna(0.0)
        overall_std_safe = overall_std.where(overall_std > STD_EPS, 1.0)

        for st, g_st in g_m.groupby("subtype_id"):
            Xst = g_st[feat_cols].apply(pd.to_numeric, errors="coerce")
            mean_st = Xst.mean(axis=0, skipna=True)
            z = ((mean_st - overall_mean) / overall_std_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            top = z.abs().sort_values(ascending=False).head(TOP_N)
            hints = []
            for feat, absz in top.items():
                zz = float(z.loc[feat])
                hint = _label_hint(feat, zz)
                if hint:
                    hints.append(hint)
                rows.append(
                    {
                        "mode_id": int(m),
                        "subtype_id": int(st),
                        "subtype_size": int(len(g_st)),
                        "feature": feat,
                        "z_vs_mode": zz,
                        "abs_z_vs_mode": float(absz),
                        "direction_vs_mode": "高" if zz > 0 else "低",
                        "mean_subtype": float(mean_st.loc[feat]) if pd.notna(mean_st.loc[feat]) else None,
                        "mean_mode": float(overall_mean.loc[feat]) if pd.notna(overall_mean.loc[feat]) else None,
                        "std_mode": float(overall_std_safe.loc[feat]) if pd.notna(overall_std_safe.loc[feat]) else None,
                        "interpretation_hint": hint,
                    }
                )

            uniq = []
            for t in hints:
                if t not in uniq:
                    uniq.append(t)
            draft = "、".join(uniq[:3]) if uniq else "混合型/需进一步解释"
            name_rows.append(
                {
                    "mode_id": int(m),
                    "subtype_id": int(st),
                    "subtype_size": int(len(g_st)),
                    "name_draft": draft,
                }
            )

    pd.DataFrame(rows).sort_values(["mode_id", "subtype_id", "abs_z_vs_mode"], ascending=[True, True, False]).to_csv(
        OUT_TOP, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(name_rows).sort_values(["mode_id", "subtype_id"]).to_csv(OUT_NAME, index=False, encoding="utf-8-sig")

    print(f"[完成] mode 维度画像：{OUT_MODE}")
    print(f"[完成] subtype Top指标：{OUT_TOP}")
    print(f"[完成] subtype 命名草案：{OUT_NAME}")


if __name__ == "__main__":
    main()

