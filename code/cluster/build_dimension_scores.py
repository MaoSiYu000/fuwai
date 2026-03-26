# -*- coding: utf-8 -*-
"""
构造“稳定维度分数”（用于模式层 mode 聚类）。

目标（你提出的诉求）：
- 模式要“分得清楚”，而且尽量来自不同维度（不是全都被某一两个指标支配）
- 每个模式下面还能再细分

做法：
- 读取 output/cluster_prep/00_features_frozen.csv（冻结原始特征，便于解释）
- 将现有特征按“数据最稳且当前已有”的维度汇总成少量维度分数：
  - academic：学业强弱（成绩/绩点/挂科）
  - attendance_engagement：出勤/活动参与
  - homework_behavior：作业提交/未批阅/耗时（行为维度）
  - online_learning：线上学习综合表现
  - fitness：体能
  - development：发展/成就（奖学金/竞赛/四六级）
- 每个维度分数内部先对各指标做标准化，再做加权求和（权重取简单可解释的 +1/-1）
- 输出：output/cluster/dimension_scores_student_term.csv

注意：
- 这些维度分数只用于“模式层（mode）聚类输入”，不替代原始特征画像。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster_prep" / "00_features_frozen.csv"
OUT_DIR = ROOT / "output" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dimension_scores_student_term.csv"


EPS = 1e-9


def _z(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    mu = v.mean(skipna=True)
    sd = v.std(ddof=0, skipna=True)
    if not np.isfinite(sd) or sd < EPS:
        sd = 1.0
    return (v.fillna(mu) - mu) / sd


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 code/cluster_prep/run_all.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in df.columns]

    # 现有稳定特征（来自冻结表）
    # 学业：分数高=更好；挂科率高=更差（负号）
    academic = (
        0.40 * _z(df.get("kccj_mean"))
        + 0.30 * _z(df.get("jdcj_mean"))
        + 0.30 * _z(df.get("by1_mean"))
        - 0.50 * _z(df.get("kccj_fail_rate"))
    )

    # 出勤/参与：出勤率高/事件多更好；缺勤率高更差
    attendance = (
        0.50 * _z(df.get("att_present_rate"))
        - 0.50 * _z(df.get("att_absent_rate"))
        + 0.35 * _z(df.get("att_event_cnt"))
    )

    # 作业行为：提交多更积极；未批阅比例高更差；耗时长/短是行为差异（这里先弱化权重）
    homework = (
        0.55 * _z(df.get("hw_submit_cnt"))
        - 0.45 * _z(df.get("hw_ungraded_rate"))
        + 0.10 * _z(df.get("hw_duration_median"))
    )

    # 线上学习
    online = 1.00 * _z(df.get("online_bfb"))

    # 体能：以综合分与肺活量为主
    fitness = 0.60 * _z(df.get("fit3_zf_mean")) + 0.40 * _z(df.get("fit3_fhl_mean"))

    # 发展/成就：奖学金金额、竞赛次数、四六级
    development = (
        0.45 * _z(df.get("sch_amt_sum_term"))
        + 0.35 * _z(df.get("comp_cnt_term"))
        + 0.20 * _z(df.get("cet_score_max"))
    )

    out = df[key_cols].copy()
    out["dim_academic"] = academic
    out["dim_attendance_engagement"] = attendance
    out["dim_homework_behavior"] = homework
    out["dim_online_learning"] = online
    out["dim_fitness"] = fitness
    out["dim_development"] = development

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[完成] 维度分数表：{OUT_PATH}（rows={len(out)} cols={len(out.columns)}）")


if __name__ == "__main__":
    main()

