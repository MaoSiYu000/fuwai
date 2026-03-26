# -*- coding: utf-8 -*-
"""
群体画像汇总表：按群体维度（现阶段：性别/民族/政治面貌/学院/专业）聚合簇占比与关键指标。

说明：
- 由于当前 `data/data1/学生基本信息_pre.csv` 不包含“班级/学院名称层级代码”等更细字段，
  这里先输出可用的群体维度：XB、MZMC、ZZMMMC、XSM、ZYM。
- 若后续拿到班级/学院更细字段，只需要在 GROUP_KEYS 里追加字段即可。

读取：
- output/cluster/student_term_gmm_with_raw_features.csv  （个体：含 cluster_map + 原始特征）
- data/data1/学生基本信息_pre.csv                       （群体维度字段）
- data/data1/学生签到记录_pre.csv / 学生作业提交记录_pre.csv （用于补充班级字段）

输出：
- output/cluster/group_profile_by_<group>.csv
  - 每个 group + TERM_KEY：各簇占比（cluster_0_pct...）+ 若干关键指标均值
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_INDIV = ROOT / "output" / "cluster" / "student_term_gmm_with_raw_features.csv"
IN_BASIC = ROOT / "data" / "data1" / "学生基本信息_pre.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)


GROUP_KEYS = {
    "sex": ["XB"],
    "ethnicity": ["MZMC"],
    "politics": ["ZZMMMC"],
    "college": ["XSM"],
    "major": ["ZYM"],
}

# 选一些最常用、最像“画像指标”的列做均值汇总（来自冻结原始特征表）
KEY_METRICS = [
    "kccj_mean",
    "kccj_fail_rate",
    "jdcj_mean",
    "by1_mean",
    "att_present_rate",
    "att_absent_rate",
    "att_event_cnt",
    "hw_submit_cnt",
    "hw_ungraded_rate",
    "hw_duration_median",
    "online_bfb",
    "sch_amt_sum_term",
    "comp_cnt_term",
    "cet_score_max",
    "fit3_zf_mean",
]

CLASS_SOURCE_CANDIDATES = [
    ROOT / "data" / "data1" / "学生签到记录_pre.csv",
    ROOT / "data" / "data1" / "学生作业提交记录_pre.csv",
]


def _build_class_map() -> pd.DataFrame:
    """
    从可用表里抽取 XH->班级 的映射。
    优先字段：
    - XH
    - LOGIN_NAME / CREATER_LOGIN_NAME（本项目中大多可直接当学号）
    - CLASS_ID + CLASS_NAME
    - CLASSID + CLASSNAME
    - BJ（班级名）
    """
    rows = []
    for p in CLASS_SOURCE_CANDIDATES:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        # 先找学号键
        if "XH" in df.columns:
            xh = df["XH"].astype(str)
        elif "LOGIN_NAME" in df.columns:
            xh = df["LOGIN_NAME"].astype(str)
        elif "CREATER_LOGIN_NAME" in df.columns:
            xh = df["CREATER_LOGIN_NAME"].astype(str)
        else:
            continue

        # 统一字段
        class_id = None
        class_name = None
        if "CLASS_ID" in df.columns:
            class_id = df["CLASS_ID"].astype(str)
        elif "CLASSID" in df.columns:
            class_id = df["CLASSID"].astype(str)

        if "CLASS_NAME" in df.columns:
            class_name = df["CLASS_NAME"].astype(str)
        elif "CLASSNAME" in df.columns:
            class_name = df["CLASSNAME"].astype(str)
        elif "BJ" in df.columns:
            class_name = df["BJ"].astype(str)

        if class_id is None and class_name is None:
            continue

        part = pd.DataFrame({"XH": xh})
        part["CLASS_ID"] = class_id if class_id is not None else ""
        part["CLASS_NAME"] = class_name if class_name is not None else ""
        part = part[(part["CLASS_ID"] != "") | (part["CLASS_NAME"] != "")]
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["XH", "CLASS_ID", "CLASS_NAME"])

    all_map = pd.concat(rows, axis=0, ignore_index=True)
    # 每个 XH 取出现频次最高的一条（保守）
    all_map["key"] = all_map["CLASS_ID"].astype(str) + "||" + all_map["CLASS_NAME"].astype(str)
    top = (
        all_map.groupby(["XH", "key"], as_index=False)
        .size()
        .sort_values(["XH", "size"], ascending=[True, False])
        .drop_duplicates(subset=["XH"], keep="first")
    )
    out = top.copy()
    out[["CLASS_ID", "CLASS_NAME"]] = out["key"].str.split("||", n=1, expand=True)
    return out[["XH", "CLASS_ID", "CLASS_NAME"]]


def _pivot_cluster_pct(g: pd.DataFrame) -> pd.DataFrame:
    pct = (
        g.groupby(["cluster_map"]).size().rename("cnt").reset_index()
    )
    pct["pct"] = pct["cnt"] / float(pct["cnt"].sum())
    wide = pct.pivot_table(index=None, columns="cluster_map", values="pct", fill_value=0.0)
    wide.columns = [f"cluster_{int(c)}_pct" for c in wide.columns]
    return wide.reset_index(drop=True)


def build_one(out_path: Path, group_cols: list[str], indiv_path: Path | None = None) -> None:
    src = indiv_path or IN_INDIV
    indiv = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    basic = pd.read_csv(IN_BASIC, encoding="utf-8-sig", low_memory=False)

    # 合并群体字段
    keep_basic = ["XH"] + [c for c in group_cols if c in basic.columns]
    merged = indiv.merge(basic[keep_basic], on="XH", how="left")

    # 准备数值指标
    for c in KEY_METRICS:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    out_rows = []
    groupby_cols = group_cols + ["TERM_KEY"]
    for keys, g in merged.groupby(groupby_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(groupby_cols, keys)}

        # 簇占比
        pct_wide = _pivot_cluster_pct(g)
        for c in pct_wide.columns:
            row[c] = float(pct_wide.iloc[0][c])

        # 指标均值
        for c in KEY_METRICS:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean(skipna=True)) if g[c].notna().any() else None

        row["n_records"] = int(len(g))
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[完成] 群体画像：{out_path}（rows={len(out)}）")


def main() -> None:
    if not IN_INDIV.exists():
        raise FileNotFoundError(f"未找到输入：{IN_INDIV}，请先运行 code/cluster/run_gmm_student_term.py")
    if not IN_BASIC.exists():
        raise FileNotFoundError(f"未找到输入：{IN_BASIC}")

    for name, cols in GROUP_KEYS.items():
        out_path = OUT_DIR / f"group_profile_by_{name}.csv"
        build_one(out_path, cols)

    # 额外：班级画像（若可从其他表补齐班级字段）
    cls_map = _build_class_map()
    if not cls_map.empty:
        indiv = pd.read_csv(IN_INDIV, encoding="utf-8-sig", low_memory=False)
        merged = indiv.merge(cls_map, on="XH", how="left")
        tmp = OUT_DIR / "_tmp_class_merge_for_group.csv"
        merged.to_csv(tmp, index=False, encoding="utf-8-sig")

        try:
            build_one(OUT_DIR / "group_profile_by_class.csv", ["CLASS_ID", "CLASS_NAME"], indiv_path=tmp)
        finally:
            if tmp.exists():
                tmp.unlink()
    else:
        print("[提示] 未找到可用班级映射字段，跳过 group_profile_by_class.csv")


if __name__ == "__main__":
    main()

