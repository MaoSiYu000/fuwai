# -*- coding: utf-8 -*-
"""
生成给前端同学的所有数据文件，输出到 front/data/。

输出文件：
  front/data/student_profiles.csv       - 学生个体画像（主表，可按学号查询）
  front/data/group_profile_by_class.csv  - 班级群体画像
  front/data/group_profile_by_major.csv  - 专业群体画像
  front/data/group_profile_by_college.csv- 学院群体画像
  front/data/mode_definitions.json       - 8个模式的名称/说明/维度画像
  front/data/subtype_definitions.json    - 32个子类的名称/说明
  front/data/dim_score_formulas.json     - 6个维度分数的计算公式说明

读取：
  output/cluster/student_term_modes_and_subtypes.csv  （聚类主结果）
  output/cluster_prep/00_features_frozen.csv          （冻结原始特征，用于原始指标展示）
  data/data1/学生基本信息_pre.csv                     （性别/学院/专业）
  data/data1/学生签到记录_pre.csv                     （班级信息来源）
  data/data1/学生作业提交记录_pre.csv                 （班级信息来源）
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "front" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CLUSTER  = ROOT / "output" / "cluster"  / "student_term_modes_and_subtypes.csv"
IN_FROZEN   = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
IN_BASIC    = ROOT / "data"   / "data1" / "学生基本信息_pre.csv"
IN_ATTN     = ROOT / "data"   / "data1" / "学生签到记录_pre.csv"
IN_HW       = ROOT / "data"   / "data1" / "学生作业提交记录_pre.csv"

# ──────────────────────────────────────────────
# 参照表：模式名称（来自 4当前模式说明.md）
# ──────────────────────────────────────────────
MODE_NAMES = {
    0: "参与稳定-学业偏弱型",
    1: "高风险波动型",
    2: "学业优势-参与积极型",
    3: "发展成就突出型",
    4: "学业中上-参与偏低型",
    5: "线上低活跃型",
    6: "学业薄弱-参与偏低型",
    7: "主流均衡型",
}

MODE_DESCRIPTIONS = {
    0: "出勤与参与较稳定，但学业整体偏弱，发展类投入（竞赛/奖学金）偏低。",
    1: "学业表现最差的高风险群体，多项指标波动明显，学业管理压力最大。",
    2: "学业突出且参与积极，各维度均衡偏高，是综合表现最好的主要群体之一。",
    3: "发展成就（奖学金/竞赛/四六级）维度远高于其他模式，是竞争力最强的少数群体。",
    4: "学业中等偏上，但课外参与与活动投入偏低，倾向专注学业的保守型。",
    5: "线上学习活跃度显著偏低（最大特征），学业和参与也偏低。",
    6: "学业偏弱且课外参与偏低，整体投入度不足，需要关注与干预。",
    7: "各维度接近整体平均，是规模最大、最典型的均衡中间群体。",
}

SUBTYPE_NAMES = {
    # mode 0
    0:   "基础稳定但发展成就偏低",
    1:   "参与活跃但学业压力较大",
    2:   "缺勤偏多且学业偏弱",
    3:   "同模式中的提升型",
    # mode 1
    100: "低发展投入型",
    101: "学业风险最突出的尖峰小群",
    102: "学业相对回升但体能短板突出",
    103: "作业投入提升、学业相对改善型",
    # mode 2
    200: "发展成就更强（奖学金/竞赛高）",
    201: "该模式主流稳态人群",
    202: "学业/体能相对回落的小群",
    203: "高挑战型（挂科风险上升但发展活跃）",
    # mode 3
    300: "学业与奖学金都较高的高成就小群",
    301: "语言/学业压力型（四六级与学业偏弱）",
    302: "稳态发展型（学业较强、发展中等）",
    303: "竞赛和体能更活跃，但学业分化",
    # mode 4
    400: "稳态保守型（挂科低但发展投入偏低）",
    401: "高表现少数群（学业+发展双高）",
    402: "发展投入较高但学业风险上升",
    403: "该模式中的学业薄弱小群",
    # mode 5
    500: "线上低活跃但学业相对维持",
    501: "线上+参与双低，且语言能力偏弱",
    502: "参与回升、发展活跃，但学业波动",
    503: "小规模高发展活跃群",
    # mode 6
    600: "薄弱稳态型（发展投入低、语言偏弱）",
    601: "综合短板型（学业与体能偏弱）",
    602: "该模式主流人群（规模最大）",
    603: "逆势提升型（学业与发展相对更好）",
    # mode 7
    700: "参与偏低、语言偏弱的小众群",
    701: "体能/参与更活跃的均衡提升群",
    702: "主流均衡基线群（人数最多）",
    703: "发展成就更高的均衡进阶群",
}

DIM_COLS = [
    "dim_academic",
    "dim_attendance_engagement",
    "dim_homework_behavior",
    "dim_online_learning",
    "dim_fitness",
    "dim_development",
]

DIM_NAMES_CN = {
    "dim_academic":               "学业",
    "dim_attendance_engagement":  "出勤参与",
    "dim_homework_behavior":      "作业行为",
    "dim_online_learning":        "线上学习",
    "dim_fitness":                "体能",
    "dim_development":            "发展成就",
}

# 用于个体画像和群体画像的关键原始指标
KEY_METRICS = [
    "kccj_mean",          # 课程成绩均值
    "kccj_fail_rate",     # 挂科率
    "jdcj_mean",          # 绩点均值
    "by1_mean",           # 百分制成绩均值
    "att_present_rate",   # 出勤率
    "att_absent_rate",    # 缺勤率
    "att_event_cnt",      # 参与活动数
    "hw_submit_cnt",      # 作业提交数
    "hw_ungraded_rate",   # 作业未批阅率
    "hw_duration_median", # 作业耗时中位数(秒)
    "online_bfb",         # 线上学习完成比例
    "sch_amt_sum_term",   # 当学期奖学金金额
    "comp_cnt_term",      # 当学期竞赛次数
    "cet_score_max",      # 四六级最高分
    "fit3_zf_mean",       # 体测综合分均值
]

METRIC_NAMES_CN = {
    "kccj_mean":          "课程成绩均值",
    "kccj_fail_rate":     "挂科率",
    "jdcj_mean":          "绩点均值",
    "by1_mean":           "百分制成绩均值",
    "att_present_rate":   "出勤率",
    "att_absent_rate":    "缺勤率",
    "att_event_cnt":      "参与活动数",
    "hw_submit_cnt":      "作业提交数",
    "hw_ungraded_rate":   "作业未批阅率",
    "hw_duration_median": "作业耗时中位数(秒)",
    "online_bfb":         "线上学习完成比例",
    "sch_amt_sum_term":   "当学期奖学金金额(元)",
    "comp_cnt_term":      "当学期竞赛次数",
    "cet_score_max":      "四六级最高分",
    "fit3_zf_mean":       "体测综合分均值",
}

# 维度 → 关键指标（用于生成依据文本）
# 每项：(字段名, 中文名, 格式串, 是否正向=True)
_DIM_TO_METRICS: dict[str, list] = {
    "dim_academic": [
        ("kccj_mean",       "课程成绩均值",   "{:.1f}分", True),
        ("kccj_fail_rate",  "挂科率",         "{:.1%}",   False),
    ],
    "dim_attendance_engagement": [
        ("att_present_rate", "出勤率",        "{:.1%}",   True),
        ("att_event_cnt",    "参与活动",      "{:.0f}次", True),
    ],
    "dim_homework_behavior": [
        ("hw_submit_cnt",    "作业提交",      "{:.0f}次", True),
        ("hw_ungraded_rate", "未批阅率",      "{:.1%}",   False),
    ],
    "dim_online_learning": [
        ("online_bfb",       "线上学习完成比例", "{:.1f}", True),
    ],
    "dim_fitness": [
        ("fit3_zf_mean",     "体测综合分",    "{:.1f}分", True),
    ],
    "dim_development": [
        ("sch_amt_sum_term", "奖学金",        "{:.0f}元", True),
        ("comp_cnt_term",    "竞赛次数",      "{:.0f}次", True),
        ("cet_score_max",    "四六级最高分",  "{:.0f}分", True),
    ],
}

_NUMBERED = ["①", "②", "③"]


def _score_level(score: float) -> str:
    a = abs(score)
    word = "高" if score > 0 else "低"
    if a >= 1.5:  return f"显著偏{word}"
    if a >= 0.5:  return f"明显偏{word}"
    if a >= 0.2:  return f"偏{word}"
    return "接近均值"


def _gen_mode_evidence(row: "pd.Series") -> str:
    """
    基于维度分数和原始指标，生成规则驱动的模式归属依据文本。
    例：①学业明显偏低（课程成绩均值68.2分，挂科率18%）；②出勤参与偏低（出勤率61%）。
    无需 LLM，可直接展示，也可作为 LLM prompt 的上下文。
    """
    dim_scores = {}
    for d in DIM_COLS:
        v = row.get(d)
        if pd.notna(v):
            dim_scores[d] = float(v)

    if not dim_scores:
        return "（数据不足，无法生成依据）"

    # 按绝对值降序，取最突出的最多 3 个维度
    sorted_dims = sorted(dim_scores.items(), key=lambda x: abs(x[1]), reverse=True)

    # 若最大绝对值 < 0.15，认为整体均衡，直接说明
    if abs(sorted_dims[0][1]) < 0.15:
        return "各维度均接近整体平均水平，整体行为表现均衡稳定。"

    parts = []
    for dim, score in sorted_dims:
        if len(parts) >= 3:
            break
        if abs(score) < 0.10:
            continue  # 太接近均值，不做依据

        level = _score_level(score)
        dim_name = DIM_NAMES_CN.get(dim, dim)

        metric_strs = []
        for metric, mname, fmt, _ in _DIM_TO_METRICS.get(dim, []):
            val = row.get(metric)
            if pd.notna(val):
                try:
                    metric_strs.append(f"{mname}{fmt.format(float(val))}")
                except Exception:
                    pass

        detail = "，".join(metric_strs[:2])
        if detail:
            parts.append(f"{dim_name}{level}（{detail}）")
        else:
            parts.append(f"{dim_name}{level}")

    if not parts:
        return "各维度均接近整体平均水平，整体行为表现均衡稳定。"

    body = "；".join(f"{_NUMBERED[i]}{p}" for i, p in enumerate(parts))
    return body + "。"


# ──────────────────────────────────────────────
# 辅助：提取班级映射
# ──────────────────────────────────────────────
def _build_class_map() -> pd.DataFrame:
    rows = []
    for path, xh_col, id_col, name_col in [
        (IN_ATTN, "LOGIN_NAME",          "CLASSID",   "CLASSNAME"),
        (IN_HW,   "CREATER_LOGIN_NAME",  "CLASS_ID",  "CLASS_NAME"),
    ]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        if xh_col not in df.columns:
            continue
        if id_col not in df.columns and name_col not in df.columns:
            continue
        part = pd.DataFrame({"XH": df[xh_col].astype(str)})
        if id_col in df.columns:
            part["CLASS_ID"] = df[id_col].astype(str)
        if name_col in df.columns:
            part["CLASS_NAME"] = df[name_col].astype(str)
        part = part.dropna(subset=["XH"])
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["XH", "CLASS_ID", "CLASS_NAME"])

    all_map = pd.concat(rows, ignore_index=True)
    # 每个 XH 取出现次数最多的班级名
    all_map["CLASS_NAME"] = all_map.get("CLASS_NAME", pd.Series(dtype=str)).fillna("")
    all_map["CLASS_ID"]   = all_map.get("CLASS_ID",   pd.Series(dtype=str)).fillna("")
    key_col = all_map["CLASS_NAME"].str.strip()
    tmp = all_map.copy()
    tmp["_key"] = key_col
    top = (
        tmp[tmp["_key"] != ""]
        .groupby(["XH", "_key"], as_index=False)
        .size()
        .sort_values(["XH", "size"], ascending=[True, False])
        .drop_duplicates(subset=["XH"], keep="first")
        .rename(columns={"_key": "CLASS_NAME"})
    )
    return top[["XH", "CLASS_NAME"]].copy()


# ──────────────────────────────────────────────
# 1. student_profiles.csv
# ──────────────────────────────────────────────
def build_student_profiles(cluster: pd.DataFrame, frozen: pd.DataFrame,
                           basic: pd.DataFrame, class_map: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["XH", "TERM_KEY"]

    # 合并原始指标
    avail_metrics = [c for c in KEY_METRICS if c in frozen.columns]
    df = cluster.merge(frozen[key_cols + avail_metrics], on=key_cols, how="left")

    # 合并基本信息（性别/学院/专业）
    basic_cols = [c for c in ["XH", "XB", "XSM", "ZYM"] if c in basic.columns]
    df = df.merge(basic[basic_cols], on="XH", how="left")

    # 合并班级
    if not class_map.empty:
        df = df.merge(class_map, on="XH", how="left")
    else:
        df["CLASS_NAME"] = None

    # 加入模式名称 & 子类名称 & 依据文本
    df["mode_name"]     = df["mode_id"].map(MODE_NAMES)
    df["subtype_name"]  = df["subtype_id"].map(SUBTYPE_NAMES)
    df["mode_evidence"] = df.apply(_gen_mode_evidence, axis=1)

    # 保留并排序输出列
    p_mode_cols = [f"p_mode_{i}" for i in range(8) if f"p_mode_{i}" in df.columns]
    base_order = [
        "XH", "TERM_KEY",
        "XB", "XSM", "ZYM", "CLASS_NAME",
        "mode_id", "mode_name",
        "mode_pmax", "mode_margin", "mode_entropy",
    ] + p_mode_cols + [
        "subtype_id", "subtype_name", "subtype_note",
        "mode_evidence",
    ] + DIM_COLS + avail_metrics

    existing = [c for c in base_order if c in df.columns]
    df = df[existing].copy()

    # 数值列保留4位小数
    float_cols = [c for c in df.select_dtypes(include="float").columns]
    df[float_cols] = df[float_cols].round(4)

    return df


# ──────────────────────────────────────────────
# 2. 群体画像（按 group 字段聚合）
# ──────────────────────────────────────────────
def build_group_profile(profiles: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in profiles.columns:
        return pd.DataFrame()

    avail_metrics = [c for c in KEY_METRICS if c in profiles.columns]
    rows = []
    for (grp_val, term), g in profiles.groupby([group_col, "TERM_KEY"], dropna=False):
        if pd.isna(grp_val) or str(grp_val).strip() in ("", "nan", "None"):
            continue
        row: dict = {group_col: grp_val, "TERM_KEY": term, "n_records": len(g)}

        # 各模式占比
        mode_counts = g["mode_id"].value_counts()
        total = len(g)
        for m in range(8):
            row[f"mode_{m}_pct"] = round(mode_counts.get(m, 0) / total, 4)

        # 主要模式（占比最高）
        row["dominant_mode_id"]   = int(g["mode_id"].mode()[0]) if len(g) > 0 else None
        row["dominant_mode_name"] = MODE_NAMES.get(row["dominant_mode_id"], "")

        # 维度分均值
        for d in DIM_COLS:
            if d in g.columns:
                row[f"{d}_mean"] = round(float(g[d].mean(skipna=True)), 4)

        # 关键原始指标均值
        for m in avail_metrics:
            row[f"{m}_avg"] = round(float(g[m].mean(skipna=True)), 4) if g[m].notna().any() else None

        rows.append(row)

    return pd.DataFrame(rows).sort_values([group_col, "TERM_KEY"])


# ──────────────────────────────────────────────
# 3. JSON 参照文件
# ──────────────────────────────────────────────
def build_mode_definitions(profiles: pd.DataFrame) -> list:
    mode_prof = profiles.groupby("mode_id", as_index=False)[DIM_COLS].mean(numeric_only=True)
    mode_sizes = profiles.groupby("mode_id", as_index=False)["XH"].count().rename(columns={"XH": "size"})
    total = len(profiles)

    result = []
    for m in sorted(MODE_NAMES.keys()):
        row: dict = {
            "mode_id":    m,
            "name":       MODE_NAMES[m],
            "description": MODE_DESCRIPTIONS[m],
        }
        # 规模
        sz_row = mode_sizes[mode_sizes["mode_id"] == m]
        sz = int(sz_row["size"].values[0]) if len(sz_row) else 0
        row["size"] = sz
        row["pct"]  = round(sz / total, 4) if total else 0

        # 维度均值
        prof_row = mode_prof[mode_prof["mode_id"] == m]
        row["dim_profile"] = {}
        for d in DIM_COLS:
            v = float(prof_row[d].values[0]) if len(prof_row) and d in prof_row.columns else 0.0
            row["dim_profile"][d] = round(v, 4)
            row["dim_profile"][f"{d}_name"] = DIM_NAMES_CN[d]

        result.append(row)
    return result


def build_subtype_definitions(profiles: pd.DataFrame) -> list:
    sub_sizes  = profiles.groupby(["mode_id", "subtype_id"], as_index=False)["XH"].count().rename(columns={"XH": "size"})
    mode_sizes = profiles.groupby("mode_id", as_index=False)["XH"].count().rename(columns={"XH": "mode_size"})
    merged = sub_sizes.merge(mode_sizes, on="mode_id")

    result = []
    for _, row in merged.iterrows():
        mid = int(row["mode_id"])
        sid = int(row["subtype_id"])
        result.append({
            "mode_id":      mid,
            "subtype_id":   sid,
            "mode_name":    MODE_NAMES.get(mid, ""),
            "name":         SUBTYPE_NAMES.get(sid, f"子类{sid}"),
            "size":         int(row["size"]),
            "pct_in_mode":  round(float(row["size"]) / float(row["mode_size"]), 4),
        })
    result.sort(key=lambda x: (x["mode_id"], x["subtype_id"]))
    return result


def build_dim_formulas() -> list:
    return [
        {
            "dim_id":      "dim_academic",
            "name_cn":     "学业",
            "description": "学业综合得分。正数表示高于全体平均，负数表示低于全体平均。越高代表成绩与绩点越好、挂科越少。",
            "components": [
                {"metric": "kccj_mean",      "name_cn": "课程成绩均值",   "weight": 0.40, "direction": "正向（越高越好）"},
                {"metric": "jdcj_mean",      "name_cn": "绩点均值",       "weight": 0.30, "direction": "正向（越高越好）"},
                {"metric": "by1_mean",       "name_cn": "百分制成绩均值", "weight": 0.30, "direction": "正向（越高越好）"},
                {"metric": "kccj_fail_rate", "name_cn": "挂科率",         "weight": 0.50, "direction": "负向（越高越差）"},
            ],
        },
        {
            "dim_id":      "dim_attendance_engagement",
            "name_cn":     "出勤参与",
            "description": "出勤与课外活动参与综合得分。正数表示出勤好、活动多；负数表示出勤不足、参与较少。",
            "components": [
                {"metric": "att_present_rate", "name_cn": "出勤率",     "weight": 0.50, "direction": "正向（越高越好）"},
                {"metric": "att_absent_rate",  "name_cn": "缺勤率",     "weight": 0.50, "direction": "负向（越高越差）"},
                {"metric": "att_event_cnt",    "name_cn": "参与活动数", "weight": 0.35, "direction": "正向（越多越好）"},
            ],
        },
        {
            "dim_id":      "dim_homework_behavior",
            "name_cn":     "作业行为",
            "description": "作业提交行为综合得分。正数表示提交积极、未批阅率低；负数表示提交偏少、未批阅率高。",
            "components": [
                {"metric": "hw_submit_cnt",      "name_cn": "作业提交数",     "weight": 0.55, "direction": "正向（越多越好）"},
                {"metric": "hw_ungraded_rate",   "name_cn": "作业未批阅率",   "weight": 0.45, "direction": "负向（越高越差）"},
                {"metric": "hw_duration_median", "name_cn": "作业耗时中位数", "weight": 0.10, "direction": "正向（行为差异参考）"},
            ],
        },
        {
            "dim_id":      "dim_online_learning",
            "name_cn":     "线上学习",
            "description": "线上学习平台活跃度得分。正数表示线上学习完成度高；负数表示线上学习低活跃。",
            "components": [
                {"metric": "online_bfb", "name_cn": "线上学习完成比例", "weight": 1.00, "direction": "正向（越高越好）"},
            ],
        },
        {
            "dim_id":      "dim_fitness",
            "name_cn":     "体能",
            "description": "体测成绩综合得分。正数表示体测表现优于平均；负数表示体测成绩偏低。",
            "components": [
                {"metric": "fit3_zf_mean",  "name_cn": "体测综合分均值", "weight": 0.60, "direction": "正向（越高越好）"},
                {"metric": "fit3_fhl_mean", "name_cn": "肺活量均值",     "weight": 0.40, "direction": "正向（越高越好）"},
            ],
        },
        {
            "dim_id":      "dim_development",
            "name_cn":     "发展成就",
            "description": "奖学金、竞赛、语言能力综合得分。正数表示发展投入与成就高于平均；负数表示该维度投入偏低。",
            "components": [
                {"metric": "sch_amt_sum_term", "name_cn": "当学期奖学金金额", "weight": 0.45, "direction": "正向（越高越好）"},
                {"metric": "comp_cnt_term",    "name_cn": "当学期竞赛次数",   "weight": 0.35, "direction": "正向（越多越好）"},
                {"metric": "cet_score_max",    "name_cn": "四六级最高分",     "weight": 0.20, "direction": "正向（越高越好）"},
            ],
        },
    ]


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main() -> None:
    print("读取聚类主结果...")
    cluster = pd.read_csv(IN_CLUSTER, encoding="utf-8-sig", low_memory=False)
    print(f"  → {len(cluster)} 行")

    print("读取冻结特征...")
    frozen = pd.read_csv(IN_FROZEN, encoding="utf-8-sig", low_memory=False)

    print("读取学生基本信息...")
    basic = pd.read_csv(IN_BASIC, encoding="utf-8-sig", low_memory=False)

    print("提取班级映射...")
    class_map = _build_class_map()
    print(f"  → 班级映射 {len(class_map)} 条")

    # ── 1. 个体画像 ──────────────────────────────
    print("\n生成 student_profiles.csv ...")
    profiles = build_student_profiles(cluster, frozen, basic, class_map)
    out_path = OUT_DIR / "student_profiles.csv"
    profiles.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  → {out_path} ({len(profiles)} 行, {len(profiles.columns)} 列)")

    # ── 2. 群体画像 ──────────────────────────────
    for col, fname in [
        ("CLASS_NAME", "group_profile_by_class.csv"),
        ("ZYM",        "group_profile_by_major.csv"),
        ("XSM",        "group_profile_by_college.csv"),
    ]:
        print(f"\n生成 {fname} ...")
        gp = build_group_profile(profiles, col)
        out_path = OUT_DIR / fname
        if gp.empty:
            print(f"  → 跳过（无 {col} 字段）")
        else:
            gp.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  → {out_path} ({len(gp)} 行)")

    # ── 3. mode_definitions.json ──────────────────
    print("\n生成 mode_definitions.json ...")
    mode_defs = build_mode_definitions(profiles)
    out_path = OUT_DIR / "mode_definitions.json"
    out_path.write_text(json.dumps(mode_defs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out_path}")

    # ── 4. subtype_definitions.json ───────────────
    print("\n生成 subtype_definitions.json ...")
    sub_defs = build_subtype_definitions(profiles)
    out_path = OUT_DIR / "subtype_definitions.json"
    out_path.write_text(json.dumps(sub_defs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out_path}")

    # ── 5. dim_score_formulas.json ────────────────
    print("\n生成 dim_score_formulas.json ...")
    formulas = build_dim_formulas()
    out_path = OUT_DIR / "dim_score_formulas.json"
    out_path.write_text(json.dumps(formulas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out_path}")

    print("\n[完成] 全部输出到：front/data/")


if __name__ == "__main__":
    main()
