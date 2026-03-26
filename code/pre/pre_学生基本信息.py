 # -*- coding: utf-8 -*-
"""
学生基本信息 预处理（严格版）
- 删除：缺学号 XH 的记录。
- 填补：ZZMMMC 政治面貌空白用众数填补。
- 规范：按学号去重；CSRQ 出生日期解析为日期并抽取出生年份（用于后续年龄段分析）。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学生基本信息.csv"
TARGET = "学生基本信息_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")
    n_before = len(df)

    # 计数器
    del_missing_xh = 0
    del_dup_xh = 0
    fill_zzmm = 0
    parsed_csrq_ok = 0
    parsed_csrq_fail = 0

    # 1) 删除缺学号
    mask_miss = df["XH"].isna() | (df["XH"].astype(str).str.strip() == "")
    del_missing_xh = int(mask_miss.sum())
    df = df[~mask_miss].copy()

    # 2) 政治面貌空白用众数填补
    if "ZZMMMC" in df.columns:
        mode_val = df["ZZMMMC"].mode()
        if len(mode_val) and pd.notna(mode_val.iloc[0]):
            fill_mask = df["ZZMMMC"].isna() | (df["ZZMMMC"].astype(str).str.strip() == "")
            fill_zzmm = int(fill_mask.sum())
            df.loc[fill_mask, "ZZMMMC"] = mode_val.iloc[0]

    # 3) 按学号去重，保留第一条
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["XH"], keep="first")
    del_dup_xh = before_dedup - len(df)

    # 4) CSRQ 出生日期解析为日期，并抽取出生年份
    if "CSRQ" in df.columns:
        # CSRQ 通常为 YYYYMMDD，如 20021029
        cs = df["CSRQ"].astype(str).str.strip()
        # 只保留长度为 8 且全数字的作为“可解析候选”
        valid_mask = cs.str.len() == 8 & cs.str.isdigit()
        # 非候选直接视为解析失败
        parsed = pd.to_datetime(cs.where(valid_mask, pd.NA), format="%Y%m%d", errors="coerce")
        parsed_csrq_ok = int(parsed.notna().sum())
        parsed_csrq_fail = int(len(parsed) - parsed_csrq_ok)
        df["CSRQ_parsed"] = parsed
        df["BIRTH_YEAR"] = df["CSRQ_parsed"].dt.year

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学生基本信息 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:          {n_before}")
    print(f"  删除(缺学号):        {del_missing_xh}")
    print(f"  删除(学号重复):      {del_dup_xh}")
    print(f"  填补(ZZMMMC 众数):   {fill_zzmm}")
    if "CSRQ" in df.columns:
        print(f"  CSRQ 解析成功:       {parsed_csrq_ok}")
        print(f"  CSRQ 解析失败:       {parsed_csrq_fail}")
    print(f"  清洗后行数:          {n_after}")
    print(f"  输出文件:            {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
