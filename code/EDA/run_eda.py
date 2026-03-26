# -*- coding: utf-8 -*-
"""
EDA 主入口（聚类前的数据初步分析）。

它做什么：
1) 扫描 data/data1、data/data2、data/data3 下的 csv，生成表级概览（行列数、候选主键覆盖、关键列缺失率）
2) 生成“学生-学期”特征草稿表（先做五个稳定指标的可解释特征；部分表无学期键则先做学生级特征并合并）
3) 对特征草稿做值域/缺失/相关性统计，并做 KMeans 的 K 粗筛（K=4..8）与稳定性抽检

它产出什么（都在 output/eda/）：
- table_overview.csv
- features_student_term_draft.csv
- feature_ranges.csv
- corr_matrix.csv
- k_screening_metrics.csv
- k_stability.csv

运行方式：
在项目根目录执行：
  python code/EDA/run_eda.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DATA_DIRS = [DATA_ROOT / "data1", DATA_ROOT / "data2", DATA_ROOT / "data3"]
OUT_DIR = ROOT / "output" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_ONLY_RE = r"^\s*\d{1,2}:\d{2}:\d{2}\s*$"


def _read_csv_chunks(path: Path, usecols: Optional[List[str]] = None, chunksize: int = 200_000):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=usecols,
        chunksize=chunksize,
    )


def _guess_id_col(cols: List[str]) -> Optional[str]:
    # 优先用 XH（你们当前统一学生主键）
    for c in ["XH", "LOGIN_NAME", "CREATER_LOGIN_NAME", "XSBH", "SID", "KS_XH", "USERNUM", "cardId", "cardld"]:
        if c in cols:
            return c
    return None


def _missing_rate_series(s: pd.Series) -> float:
    s2 = s.astype(str)
    miss = s.isna() | (s2.str.strip() == "") | (s2.str.strip().str.lower() == "nan")
    return float(miss.mean()) if len(s) else math.nan


def table_overview() -> pd.DataFrame:
    rows = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.csv")):
            # 只读表头先判断
            try:
                head = pd.read_csv(p, encoding="utf-8-sig", nrows=0)
            except Exception:
                # 有些 csv 可能不是 utf-8-sig，兜底尝试 gbk
                head = pd.read_csv(p, encoding="gbk", nrows=0)
            cols = list(head.columns)
            id_col = _guess_id_col(cols)

            # 对大表做 chunk 统计，避免一次性读入
            n_rows = 0
            miss_id = 0
            uniq_sample = set()
            uniq_cap = 200_000  # 防止集合爆内存：只做“上限采样去重”
            key_cols = [c for c in ["XH", "LOGIN_NAME", "CREATER_LOGIN_NAME", "BFB", "KCCJ", "JDCJ", "SJCJ", "HSCJ", "ACTIVE_STATE", "SCORE", "FULLMARKS"] if c in cols]
            usecols = sorted(set(([id_col] if id_col else []) + key_cols))

            enc = "utf-8-sig"
            try:
                it = _read_csv_chunks(p, usecols=usecols)
            except Exception:
                enc = "gbk"
                it = pd.read_csv(p, encoding=enc, low_memory=False, usecols=usecols, chunksize=200_000)

            miss_rates: Dict[str, float] = {}
            miss_sum: Dict[str, int] = {c: 0 for c in usecols}
            total_sum = 0

            for chunk in it:
                n = len(chunk)
                n_rows += n
                total_sum += n
                for c in usecols:
                    s = chunk[c] if c in chunk.columns else pd.Series([pd.NA] * n)
                    s2 = s.astype(str)
                    miss = s.isna() | (s2.str.strip() == "") | (s2.str.strip().str.lower() == "nan")
                    miss_sum[c] += int(miss.sum())

                if id_col and id_col in chunk.columns:
                    s = chunk[id_col].astype(str).str.strip()
                    s = s[s != ""]
                    miss_id += int((chunk[id_col].isna() | (chunk[id_col].astype(str).str.strip() == "")).sum())
                    if len(uniq_sample) < uniq_cap:
                        uniq_sample.update(s.head(uniq_cap - len(uniq_sample)).tolist())

            for c in usecols:
                miss_rates[c] = (miss_sum[c] / total_sum) if total_sum else math.nan

            rows.append(
                {
                    "data_dir": d.name,
                    "file": str(p.relative_to(ROOT)),
                    "n_rows": n_rows,
                    "n_cols": len(cols),
                    "id_col_guess": id_col or "",
                    "id_missing_rate": (miss_id / n_rows) if (id_col and n_rows) else math.nan,
                    "id_unique_sample_cap": uniq_cap if id_col else 0,
                    "id_unique_sampled": len(uniq_sample) if id_col else 0,
                    "encoding_used": enc,
                    "key_cols_checked": ",".join(usecols),
                    "missing_rates_json": str({c: round(miss_rates[c], 6) for c in usecols}),
                }
            )

    df = pd.DataFrame(rows)
    out = OUT_DIR / "table_overview.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def _term_key_from_grade(zx: pd.Series) -> pd.Series:
    """
    成绩表 ZXJXJHH 常见形态：2020-2021-1 / 2020-2021-2 等。
    返回 TERM_KEY：原样保留（便于与你们其他口径对齐）。
    """
    s = zx.astype(str).str.strip()
    s = s.replace({"nan": pd.NA, "": pd.NA})
    return s


def _term_key_from_xq_cn(xq: pd.Series) -> pd.Series:
    """
    体测表 XQ 常见形态：2020-2021学年第2学期（也可能包含空格/不同编码）。
    映射为与成绩表一致的 TERM_KEY：YYYY-YYYY-1/2
    """
    s = xq.astype(str).str.strip()
    s = s.replace({"nan": pd.NA, "": pd.NA})

    # 先把常见中文字符统一
    s2 = (
        s.str.replace("学年", "学年", regex=False)
        .str.replace("学期", "学期", regex=False)
        .str.replace("第", "第", regex=False)
    )

    # 提取：起始年-结束年 与 学期号
    # 兼容：2020-2021学年第2学期 / 2020-2021 学年第 2 学期
    m = s2.str.extract(r"(?P<y1>\d{4})\s*-\s*(?P<y2>\d{4}).*?第\s*(?P<t>[12])\s*学期")
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    ok = m["y1"].notna() & m["y2"].notna() & m["t"].notna()
    out.loc[ok] = m.loc[ok, "y1"] + "-" + m.loc[ok, "y2"] + "-" + m.loc[ok, "t"]
    return out


def _term_key_from_tcnf_year(tcnf: pd.Series) -> pd.Series:
    """
    data3 体测数据pre.csv 使用年度字段 TCNF（如 2020.0/2021.0/2022.0）。
    保守映射策略：将年度 y 映射为 y-(y+1)-2（第二学期）。

    理由（保守、避免扩散）：
    - 你们 data1 的体能考核 XQ 主要集中在“学年第2学期”，与该映射更一致；
    - 不把年度体测同时填到两个学期，避免人为制造“全学期都有体测”的假象。
    """
    y = pd.to_numeric(tcnf, errors="coerce")
    y_int = y.dropna().astype(int)
    out = pd.Series(pd.NA, index=tcnf.index, dtype="object")
    ok = y.notna()
    # 仅当转换成 int 后仍合理时写入
    out.loc[ok] = y.loc[ok].astype(int).astype(str) + "-" + (y.loc[ok].astype(int) + 1).astype(str) + "-2"
    return out


def _term_key_from_year_month(y: pd.Series, m: pd.Series) -> pd.Series:
    """
    将 (year, month) 粗略映射到学期键 TERM_KEY（保守规则）：
    - 若 month >= 8：认为在当学年第一学期（y-(y+1)-1）
    - 否则：认为在当学年第二学期（y-(y+1)-2）
    用途：综合测评等表里只有“年+月”口径时的近似对齐。
    """
    yy = pd.to_numeric(y, errors="coerce")
    mm = pd.to_numeric(m, errors="coerce")
    out = pd.Series(pd.NA, index=y.index, dtype="object")
    ok = yy.notna() & mm.notna()
    y_int = yy.loc[ok].astype(int)
    term = np.where(mm.loc[ok] >= 8, 1, 2)
    out.loc[ok] = y_int.astype(str) + "-" + (y_int + 1).astype(str) + "-" + pd.Series(term, index=y_int.index).astype(str)
    return out


def _term_key_from_date_series(dt: pd.Series) -> pd.Series:
    """
    将日期时间列映射到学期键（保守规则）：
    - 取年份 year
    - 若月份 >= 8：TERM_KEY=year-(year+1)-1
    - 否则：TERM_KEY=year-(year+1)-2
    """
    d = pd.to_datetime(dt, errors="coerce")
    out = pd.Series(pd.NA, index=dt.index, dtype="object")
    ok = d.notna()
    y = d.loc[ok].dt.year.astype(int)
    term = np.where(d.loc[ok].dt.month >= 8, 1, 2)
    out.loc[ok] = y.astype(str) + "-" + (y + 1).astype(str) + "-" + pd.Series(term, index=y.index).astype(str)
    return out


def build_features_student_term_draft() -> pd.DataFrame:
    """
    生成学生-学期特征草稿表（五个稳定指标优先落地）。
    说明：部分表缺少学期键或时间轴口径不统一，暂以“学生级汇总特征”拼接到每个学生-学期行上，
    作为 EDA/算法选择阶段的近似输入；正式建模阶段可进一步做按时间对齐。
    """
    d1 = DATA_ROOT / "data1"
    grade_path = d1 / "学生成绩_pre.csv"
    hw_path = d1 / "学生作业提交记录_pre.csv"
    att_path = d1 / "学生签到记录_pre.csv"
    online_path = d1 / "线上学习（综合表现）_pre.csv"
    fit_path = d1 / "学生体能考核_pre.csv"

    if not grade_path.exists():
        raise FileNotFoundError(f"未找到 {grade_path}（需要先把清洗结果放到 data/data1）")

    # 1) 学业表现（学生-学期）
    usecols = [c for c in ["XH", "ZXJXJHH", "KCCJ", "JDCJ", "BY1"] if c]
    grade = pd.read_csv(grade_path, encoding="utf-8-sig", low_memory=False)
    for c in ["KCCJ", "JDCJ", "BY1"]:
        if c in grade.columns:
            grade[c] = pd.to_numeric(grade[c], errors="coerce")
    grade["TERM_KEY"] = _term_key_from_grade(grade["ZXJXJHH"]) if "ZXJXJHH" in grade.columns else pd.NA
    grade = grade.dropna(subset=["XH", "TERM_KEY"])

    def _fail_rate(s: pd.Series, thr: float = 60.0) -> float:
        s2 = pd.to_numeric(s, errors="coerce")
        valid = s2.notna()
        if valid.sum() == 0:
            return math.nan
        return float((s2[valid] < thr).mean())

    g_agg = (
        grade.groupby(["XH", "TERM_KEY"], as_index=False)
        .agg(
            grade_course_cnt=("KCCJ", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum()) if "KCCJ" in grade.columns else 0),
            kccj_mean=("KCCJ", "mean") if "KCCJ" in grade.columns else ("XH", "size"),
            kccj_median=("KCCJ", "median") if "KCCJ" in grade.columns else ("XH", "size"),
            kccj_fail_rate=("KCCJ", _fail_rate) if "KCCJ" in grade.columns else ("XH", "size"),
            jdcj_mean=("JDCJ", "mean") if "JDCJ" in grade.columns else ("XH", "size"),
            by1_mean=("BY1", "mean") if "BY1" in grade.columns else ("XH", "size"),
        )
    )

    # 2) 线上学习（学生级：BFB）
    online_feat = None
    if online_path.exists():
        online = pd.read_csv(online_path, encoding="utf-8-sig", low_memory=False)
        if "LOGIN_NAME" in online.columns:
            online = online.rename(columns={"LOGIN_NAME": "XH"})
        if "BFB" in online.columns:
            online["BFB"] = pd.to_numeric(online["BFB"], errors="coerce")
            online_feat = online.groupby("XH", as_index=False).agg(online_bfb=("BFB", "mean"))

    # 3) 体测（学生-学期或学生级：优先按 XQ）
    fit_feat_term = None
    if fit_path.exists():
        fit = pd.read_csv(fit_path, encoding="utf-8-sig", low_memory=False)
        for c in ["SJCJ", "HSCJ"]:
            if c in fit.columns:
                fit[c] = pd.to_numeric(fit[c], errors="coerce")
        if "XQ" in fit.columns:
            fit = fit.dropna(subset=["XH", "XQ"])
            # 将体测 XQ（中文学期）映射到成绩表同款 TERM_KEY
            fit["TERM_KEY"] = _term_key_from_xq_cn(fit["XQ"])
            fit = fit.dropna(subset=["TERM_KEY"])
            fit_feat_term = fit.groupby(["XH", "TERM_KEY"], as_index=False).agg(
                fit_sjcj_mean=("SJCJ", "mean") if "SJCJ" in fit.columns else ("XH", "size"),
                fit_hscj_mean=("HSCJ", "mean") if "HSCJ" in fit.columns else ("XH", "size"),
            )

    # 3b) 体测数据（data3，年度 TCNF → TERM_KEY=学年第2学期，保守映射）
    fit3_feat_term = None
    fit3_path = DATA_ROOT / "data3" / "体测数据pre.csv"
    if fit3_path.exists():
        fit3 = pd.read_csv(fit3_path, encoding="utf-8-sig", low_memory=False)
        if "XH" in fit3.columns and "TCNF" in fit3.columns:
            fit3["TERM_KEY"] = _term_key_from_tcnf_year(fit3["TCNF"])
            fit3 = fit3.dropna(subset=["XH", "TERM_KEY"])
            # 数值字段尽量转数值（缺失保留 NA）
            for c in ["ZF", "FHL", "WS", "LDTY", "ZWTQQ", "BB", "YQ", "YWQZ", "YTXS"]:
                if c in fit3.columns:
                    fit3[c] = pd.to_numeric(fit3[c], errors="coerce")
            # BMI 形态可能是 "166/53"（身高/体重），这里先不强转，后续若需要再拆
            fit3_feat_term = fit3.groupby(["XH", "TERM_KEY"], as_index=False).agg(
                fit3_zf_mean=("ZF", "mean") if "ZF" in fit3.columns else ("XH", "size"),
                fit3_fhl_mean=("FHL", "mean") if "FHL" in fit3.columns else ("XH", "size"),
                fit3_ws_mean=("WS", "mean") if "WS" in fit3.columns else ("XH", "size"),
                fit3_ldty_mean=("LDTY", "mean") if "LDTY" in fit3.columns else ("XH", "size"),
                fit3_zwtqq_mean=("ZWTQQ", "mean") if "ZWTQQ" in fit3.columns else ("XH", "size"),
            )

    # 3c) 综合测评（data2，本科生综合测评.csv）：
    # 两条路：
    # - 若能可靠对齐到学期：做 student-term 特征
    # - 若“学期级关键字段几乎全空/不可用”：降级为 student-level 特征（按 XH 聚合）
    zc_feat_term = None
    zc_feat_student = None
    zc_path = DATA_ROOT / "data2" / "本科生综合测评.csv"
    if zc_path.exists():
        zc = pd.read_csv(zc_path, encoding="utf-8-sig", low_memory=False)
        if "XH" in zc.columns:
            # 尝试用 PDXN/PDXQ（若为空则用 CPXN/CPXQ）
            y_col = "PDXN" if "PDXN" in zc.columns and zc["PDXN"].notna().any() else ("CPXN" if "CPXN" in zc.columns else None)
            m_col = "PDXQ" if "PDXQ" in zc.columns and zc["PDXQ"].notna().any() else ("CPXQ" if "CPXQ" in zc.columns else None)
            if y_col and m_col:
                zc["TERM_KEY"] = _term_key_from_year_month(zc[y_col], zc[m_col])
                zc = zc.dropna(subset=["XH", "TERM_KEY"])
                for c in ["ZF", "BJPM", "ZYPM", "BJPMBL", "ZYPMBL", "BJRS", "ZYRS"]:
                    if c in zc.columns:
                        zc[c] = pd.to_numeric(zc[c], errors="coerce")
                # 经验规则：若 ZF/ZYPM 学期级几乎全空，则不做学期级，降级为学生级
                zf_ok = ("ZF" in zc.columns) and (zc["ZF"].notna().mean() > 0.05)
                zypm_ok = ("ZYPM" in zc.columns) and (zc["ZYPM"].notna().mean() > 0.05)
                if zf_ok or zypm_ok:
                    zc_feat_term = zc.groupby(["XH", "TERM_KEY"], as_index=False).agg(
                        zc_zf_mean=("ZF", "mean") if "ZF" in zc.columns else ("XH", "size"),
                        zc_bjpm_mean=("BJPM", "mean") if "BJPM" in zc.columns else ("XH", "size"),
                        zc_zypm_mean=("ZYPM", "mean") if "ZYPM" in zc.columns else ("XH", "size"),
                    )

                # 学生级（总体）特征：更稳健
                agg_dict = {
                    "zc_records": ("XH", "size"),
                }
                if "BJPM" in zc.columns and zc["BJPM"].notna().any():
                    agg_dict["zc_bjpm_best"] = ("BJPM", "min")
                    agg_dict["zc_bjpm_mean"] = ("BJPM", "mean")

                # ZYPM 在当前数据中可能全空：只有当非空比例足够时才纳入
                if "ZYPM" in zc.columns and (zc["ZYPM"].notna().mean() > 0.05):
                    agg_dict["zc_zypm_best"] = ("ZYPM", "min")
                    agg_dict["zc_zypm_mean"] = ("ZYPM", "mean")

                zc_feat_student = zc.groupby("XH", as_index=False).agg(**agg_dict)

    # 3d) 奖学金（data2，奖学金获奖.csv）：学年 PDXN（年度）保守映射到第二学期
    sch_feat = None
    sch_path = DATA_ROOT / "data2" / "奖学金获奖.csv"
    if sch_path.exists():
        sch = pd.read_csv(sch_path, encoding="utf-8-sig", low_memory=False)
        if "XSBH" in sch.columns and "XH" not in sch.columns:
            sch = sch.rename(columns={"XSBH": "XH"})
        if "PDXN" in sch.columns:
            sch["TERM_KEY"] = _term_key_from_tcnf_year(sch["PDXN"])
        else:
            sch["TERM_KEY"] = pd.NA
        for c in ["FFJE"]:
            if c in sch.columns:
                sch[c] = pd.to_numeric(sch[c], errors="coerce")
        if "XH" in sch.columns:
            # 既输出学生级，也输出学生-学期（若 TERM_KEY 能构造）
            sch_student = sch.groupby("XH", as_index=False).agg(
                sch_cnt=("XH", "size"),
                sch_amt_sum=("FFJE", "sum") if "FFJE" in sch.columns else ("XH", "size"),
            )
            sch_feat = ("student", sch_student)
            if "TERM_KEY" in sch.columns and sch["TERM_KEY"].notna().any():
                sch_term = sch.dropna(subset=["TERM_KEY"]).groupby(["XH", "TERM_KEY"], as_index=False).agg(
                    sch_cnt_term=("XH", "size"),
                    sch_amt_sum_term=("FFJE", "sum") if "FFJE" in sch.columns else ("XH", "size"),
                )
                sch_feat = ("both", (sch_student, sch_term))

    # 3e) 四六级（data3，四六级成绩pre.csv）：学生级特征（保守合并到所有学期）
    cet_feat = None
    cet_path = DATA_ROOT / "data3" / "四六级成绩pre.csv"
    if cet_path.exists():
        cet = pd.read_csv(cet_path, encoding="utf-8-sig", low_memory=False)
        if "KS_XH" in cet.columns and "XH" not in cet.columns:
            cet = cet.rename(columns={"KS_XH": "XH"})
        if "KS_CJ" in cet.columns:
            cet["KS_CJ"] = pd.to_numeric(cet["KS_CJ"], errors="coerce")
        if "KS_YYJB" in cet.columns:
            cet["KS_YYJB"] = pd.to_numeric(cet["KS_YYJB"], errors="coerce")
        if "XH" in cet.columns:
            cet_feat = cet.groupby("XH", as_index=False).agg(
                cet_score_max=("KS_CJ", "max") if "KS_CJ" in cet.columns else ("XH", "size"),
                cet_level_max=("KS_YYJB", "max") if "KS_YYJB" in cet.columns else ("XH", "size"),
            )

    # 3f) 竞赛（data1，学科竞赛_pre.csv）：按获奖时间近似映射 TERM_KEY（学生级与学生-学期都输出）
    comp_feat = None
    comp_path = DATA_ROOT / "data1" / "学科竞赛_pre.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path, encoding="utf-8-sig", low_memory=False)
        if "XHHGH" in comp.columns and "XH" not in comp.columns:
            comp = comp.rename(columns={"XHHGH": "XH"})
        if "HJSJ_PARSED" in comp.columns:
            comp["TERM_KEY"] = _term_key_from_date_series(comp["HJSJ_PARSED"])
        else:
            comp["TERM_KEY"] = pd.NA
        if "XH" in comp.columns:
            comp_student = comp.groupby("XH", as_index=False).agg(comp_cnt=("XH", "size"))
            comp_feat = ("student", comp_student)
            if comp["TERM_KEY"].notna().any():
                comp_term = comp.dropna(subset=["TERM_KEY"]).groupby(["XH", "TERM_KEY"], as_index=False).agg(comp_cnt_term=("XH", "size"))
                comp_feat = ("both", (comp_student, comp_term))

    # 3g) 学籍异动（data1，学籍异动_pre.csv）：按生效日期近似映射 TERM_KEY（事件表 → 学期聚合）
    yd_feat_term = None
    yd_path = DATA_ROOT / "data1" / "学籍异动_pre.csv"
    if yd_path.exists():
        yd = pd.read_csv(yd_path, encoding="utf-8-sig", low_memory=False)
        if "BY1_parsed" in yd.columns:
            yd["TERM_KEY"] = _term_key_from_date_series(yd["BY1_parsed"])
            yd = yd.dropna(subset=["XH", "TERM_KEY"])
            yd["YDLBDM"] = pd.to_numeric(yd["YDLBDM"], errors="coerce")
            yd_feat_term = yd.groupby(["XH", "TERM_KEY"], as_index=False).agg(
                yd_cnt=("XH", "size"),
                yd_type_nunique=("YDLBDM", "nunique"),
            )

    # 4) 作业（学生级汇总：提交次数、未批阅占比、作答耗时中位数）
    hw_feat = None
    if hw_path.exists():
        hw = pd.read_csv(hw_path, encoding="utf-8-sig", low_memory=False)
        if "CREATER_LOGIN_NAME" in hw.columns:
            hw = hw.rename(columns={"CREATER_LOGIN_NAME": "XH"})
        if "IS_UNGRADED" in hw.columns:
            hw["IS_UNGRADED"] = pd.to_numeric(hw["IS_UNGRADED"], errors="coerce")
        # 修正作业耗时（口径2：假设跨小时）：
        # 如果原始时间只有“时分秒”，且 ANSWER < CREATER，则视为跨 1 小时（+3600 秒）。
        # 背景：该表时间字段几乎全部显示为 00:MM:SS，说明并非真实日期时间；用 +3600 更符合常识。
        dur = None
        if "ANSWER_DURATION_SEC" in hw.columns:
            dur = pd.to_numeric(hw["ANSWER_DURATION_SEC"], errors="coerce")
        if "CREATER_TIME" in hw.columns and "ANSWER_TIME" in hw.columns:
            ct_s = hw["CREATER_TIME"].astype(str)
            at_s = hw["ANSWER_TIME"].astype(str)
            mask_time_only = ct_s.str.match(TIME_ONLY_RE, na=False) & at_s.str.match(TIME_ONLY_RE, na=False)
            if mask_time_only.any():
                # 只对 time-only 行重新计算并修正
                ct_td = pd.to_timedelta(ct_s.where(mask_time_only), errors="coerce")
                at_td = pd.to_timedelta(at_s.where(mask_time_only), errors="coerce")
                delta = (at_td - ct_td).dt.total_seconds()
                # 跨小时修正：负值 + 3600
                delta_fixed = delta.where(delta.isna() | (delta >= 0), delta + 3600)
                if dur is None:
                    dur = pd.Series([pd.NA] * len(hw))
                # 写回修正后的耗时（仅覆盖 time-only 的行）
                dur = dur.copy()
                dur.loc[mask_time_only] = delta_fixed.loc[mask_time_only]

        if dur is not None:
            hw["ANSWER_DURATION_SEC_FIXED"] = pd.to_numeric(dur, errors="coerce")

        hw_feat = hw.groupby("XH", as_index=False).agg(
            hw_submit_cnt=("XH", "size"),
            hw_ungraded_rate=("IS_UNGRADED", "mean") if "IS_UNGRADED" in hw.columns else ("XH", "size"),
            hw_duration_median=("ANSWER_DURATION_SEC_FIXED", "median")
            if "ANSWER_DURATION_SEC_FIXED" in hw.columns
            else (("ANSWER_DURATION_SEC", "median") if "ANSWER_DURATION_SEC" in hw.columns else ("XH", "size")),
        )

    # 5) 签到（学生级汇总：出勤/缺勤/异常状态占比）
    att_feat = None
    if att_path.exists():
        att = pd.read_csv(att_path, encoding="utf-8-sig", low_memory=False)
        if "LOGIN_NAME" in att.columns:
            att = att.rename(columns={"LOGIN_NAME": "XH"})
        if "ACTIVE_STATE" in att.columns:
            st = pd.to_numeric(att["ACTIVE_STATE"], errors="coerce")
            att["_is_present"] = (st == 1).astype(float)  # 已签
            att["_is_absent"] = (st == 5).astype(float)   # 缺勤（按你们字典）
            att["_is_abnormal"] = (~st.isin([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]) & st.notna()).astype(float)
            att_feat = att.groupby("XH", as_index=False).agg(
                att_event_cnt=("XH", "size"),
                att_present_rate=("_is_present", "mean"),
                att_absent_rate=("_is_absent", "mean"),
                att_abnormal_rate=("_is_abnormal", "mean"),
            )

    # 合并：以成绩学生-学期为主表
    feat = g_agg.copy()
    if fit_feat_term is not None:
        feat = feat.merge(fit_feat_term, on=["XH", "TERM_KEY"], how="left")
    if fit3_feat_term is not None:
        feat = feat.merge(fit3_feat_term, on=["XH", "TERM_KEY"], how="left")
    if zc_feat_term is not None:
        feat = feat.merge(zc_feat_term, on=["XH", "TERM_KEY"], how="left")
    if yd_feat_term is not None:
        feat = feat.merge(yd_feat_term, on=["XH", "TERM_KEY"], how="left")
    if online_feat is not None:
        feat = feat.merge(online_feat, on="XH", how="left")
    if hw_feat is not None:
        feat = feat.merge(hw_feat, on="XH", how="left")
    if att_feat is not None:
        feat = feat.merge(att_feat, on="XH", how="left")
    if zc_feat_student is not None:
        feat = feat.merge(zc_feat_student, on="XH", how="left")
    # 学生级特征保守合并（对齐到每个学生-学期）
    if sch_feat is not None:
        if sch_feat[0] == "student":
            feat = feat.merge(sch_feat[1], on="XH", how="left")
        else:
            sch_student, sch_term = sch_feat[1]
            feat = feat.merge(sch_student, on="XH", how="left")
            feat = feat.merge(sch_term, on=["XH", "TERM_KEY"], how="left")
    if cet_feat is not None:
        feat = feat.merge(cet_feat, on="XH", how="left")
    if comp_feat is not None:
        if comp_feat[0] == "student":
            feat = feat.merge(comp_feat[1], on="XH", how="left")
        else:
            comp_student, comp_term = comp_feat[1]
            feat = feat.merge(comp_student, on="XH", how="left")
            feat = feat.merge(comp_term, on=["XH", "TERM_KEY"], how="left")

    out = OUT_DIR / "features_student_term_draft.csv"
    feat.to_csv(out, index=False, encoding="utf-8-sig")
    return feat


def feature_ranges(feat: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in feat.columns if c not in ["XH", "TERM_KEY"]]
    rows = []
    for c in numeric_cols:
        s = pd.to_numeric(feat[c], errors="coerce")
        rows.append(
            {
                "feature": c,
                "missing_rate": float(s.isna().mean()),
                "zero_rate": float((s.fillna(0) == 0).mean()),
                "min": float(s.min()) if s.notna().any() else math.nan,
                "p50": float(s.quantile(0.5)) if s.notna().any() else math.nan,
                "mean": float(s.mean()) if s.notna().any() else math.nan,
                "p95": float(s.quantile(0.95)) if s.notna().any() else math.nan,
                "p99": float(s.quantile(0.99)) if s.notna().any() else math.nan,
                "max": float(s.max()) if s.notna().any() else math.nan,
            }
        )
    df = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    out = OUT_DIR / "feature_ranges.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def corr_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    x = feat.drop(columns=["XH", "TERM_KEY"], errors="ignore")
    x = x.apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman", min_periods=500)
    out = OUT_DIR / "corr_matrix.csv"
    corr.to_csv(out, encoding="utf-8-sig")
    return corr


def kmeans_screening(feat: pd.DataFrame, k_min: int = 4, k_max: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    K 粗筛（EDA用，不是最终建模）。
    - 先做简单缺失填补（中位数）+ 标准化（z-score）
    - 计算 silhouette / CH / DBI
    - 稳定性：同一K不同随机种子下的 ARI（抽检）
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
        adjusted_rand_score,
    )

    x = feat.drop(columns=["XH", "TERM_KEY"], errors="ignore").copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    # 仅保留“缺失不太严重”的列（EDA 阶段简单阈值）
    keep_cols = [c for c in x.columns if float(x[c].isna().mean()) <= 0.6]
    x = x[keep_cols]
    # 中位数填补
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    # 标准化
    mu = x.mean()
    sd = x.std(ddof=0).replace(0, 1)
    xz = (x - mu) / sd

    metrics_rows = []
    stab_rows = []

    seeds = [0, 7, 21, 42, 100]
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(xz)
        # silhouette 对样本量大也会慢，但你们学生-学期行数通常可控；若太慢可后续改成抽样
        sil = silhouette_score(xz, labels, metric="euclidean")
        ch = calinski_harabasz_score(xz, labels)
        dbi = davies_bouldin_score(xz, labels)
        metrics_rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": dbi})

        # 稳定性：同K不同seed两两 ARI（抽检：只算与基准seed=42的）
        base = labels
        for seed in seeds:
            km2 = KMeans(n_clusters=k, n_init=20, random_state=seed)
            lab2 = km2.fit_predict(xz)
            ari = adjusted_rand_score(base, lab2)
            stab_rows.append({"k": k, "seed": seed, "ari_vs_seed42": ari})

    metrics_df = pd.DataFrame(metrics_rows)
    stab_df = pd.DataFrame(stab_rows)
    metrics_df.to_csv(OUT_DIR / "k_screening_metrics.csv", index=False, encoding="utf-8-sig")
    stab_df.to_csv(OUT_DIR / "k_stability.csv", index=False, encoding="utf-8-sig")
    return metrics_df, stab_df


def main() -> None:
    print("== EDA 1/4 表级概览 ==")
    table_overview()

    print("== EDA 2/4 学生-学期特征草稿 ==")
    feat = build_features_student_term_draft()

    print("== EDA 3/4 特征值域与相关性 ==")
    feature_ranges(feat)
    corr_matrix(feat)

    print("== EDA 4/4 KMeans K粗筛与稳定性 ==")
    kmeans_screening(feat, 4, 8)

    print("\nEDA 完成，产出目录：output/eda/")


if __name__ == "__main__":
    main()

