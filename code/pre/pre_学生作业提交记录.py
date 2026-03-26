 # -*- coding: utf-8 -*-
"""
学生作业提交记录 预处理（严格版）
- 删除：缺 CREATER_LOGIN_NAME 或 COURSE_ID；SCORE > FULLMARKS。
- 标记：STATUS=3（待批阅）且 SCORE=0 记为未批阅 IS_UNGRADED=1。
- 规范时间：尝试解析 CREATER_TIME / ANSWER_TIME，计算简单耗时（秒），便于后续分析。
- 输出到 output/pre/学生作业提交记录_pre.csv，并逐类打印统计。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学生作业提交记录.csv"
TARGET = "学生作业提交记录_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    n_before = len(df)

    # 计数器
    del_missing_key = 0
    del_score_gt_full = 0
    mark_ungraded = 0
    parsed_time_ok = 0
    parsed_time_fail = 0

    # 1) 缺关键字段：CREATER_LOGIN_NAME, COURSE_ID
    key_cols = [c for c in ["CREATER_LOGIN_NAME", "COURSE_ID"] if c in df.columns]
    if key_cols:
        mask = df[key_cols[0]].isna() | (df[key_cols[0]].astype(str).str.strip() == "")
        if len(key_cols) > 1:
            mask |= df[key_cols[1]].isna() | (df[key_cols[1]].astype(str).str.strip() == "")
        del_missing_key = int(mask.sum())
        df = df[~mask].copy()

    # 2) SCORE > FULLMARKS 删除
    if "SCORE" in df.columns and "FULLMARKS" in df.columns:
        sc = pd.to_numeric(df["SCORE"], errors="coerce")
        fm = pd.to_numeric(df["FULLMARKS"], errors="coerce")
        bad = sc.notna() & fm.notna() & (sc > fm)
        del_score_gt_full = int(bad.sum())
        df = df[~bad].copy()

    # 3) 标记未批阅记录：STATUS=3 & SCORE=0 → IS_UNGRADED=1
    df["IS_UNGRADED"] = 0
    if "STATUS" in df.columns and "SCORE" in df.columns:
        status = pd.to_numeric(df["STATUS"], errors="coerce")
        score = pd.to_numeric(df["SCORE"], errors="coerce")
        mask_ungraded = (status == 3) & (score == 0)
        mark_ungraded = int(mask_ungraded.sum())
        df.loc[mask_ungraded, "IS_UNGRADED"] = 1

    # 4) 时间解析与耗时（秒）计算：CREATER_TIME / ANSWER_TIME
    if "CREATER_TIME" in df.columns and "ANSWER_TIME" in df.columns:
        # 这里原始有 00:25:20 这种纯时间，也有完整日期时间，统一用 to_datetime 尝试
        ct = pd.to_datetime(df["CREATER_TIME"], errors="coerce")
        at = pd.to_datetime(df["ANSWER_TIME"], errors="coerce")
        parsed_time_ok = int((ct.notna() & at.notna()).sum())
        parsed_time_fail = int(len(df) - parsed_time_ok)

        df["CREATER_TIME_parsed"] = ct
        df["ANSWER_TIME_parsed"] = at
        # 简单耗时：只在两者都非空时计算
        delta = (at - ct).dt.total_seconds()
        df["ANSWER_DURATION_SEC"] = delta

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学生作业提交记录 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:           {n_before}")
    print(f"  删除(缺主键):         {del_missing_key}")
    print(f"  删除(分数>满分):      {del_score_gt_full}")
    print(f"  标记未批阅(IS_UNGRADED=1): {mark_ungraded}")
    if "CREATER_TIME" in df.columns and "ANSWER_TIME" in df.columns:
        print(f"  解析时间成功(两端都有): {parsed_time_ok}")
        print(f"  解析时间失败:         {parsed_time_fail}")
    print(f"  清洗后行数:           {n_after}")
    print(f"  输出文件:             {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
