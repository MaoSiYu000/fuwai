 # -*- coding: utf-8 -*-
"""
学生成绩 预处理（严格按数据集说明.md 中的规则）
- 有效字段：XH、KCH、KCSXDM、KCCJ、DJCJ、JDCJ、BY1 等。
- 删除：缺 XH/KCH；KCCJ/BY1/JDCJ 超出合理范围；BY1 与 JDCJ 同时空或为负的行。
- DJCJ：允许为空或为负，只记录统计信息，不作为删行条件（视为“质量很差的辅助字段”）。
- 输出到 output/pre/学生成绩_pre.csv，并打印各类删除/修改统计。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学生成绩.csv"
TARGET = "学生成绩_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    n_before = len(df)

    # 统计用计数器
    del_missing_key = 0
    del_kccj_range = 0
    del_by1_range = 0
    del_jdcj_range = 0
    del_by1_jdcj_invalid_pair = 0

    # 缺学号或课程号
    before = len(df)
    df = df.dropna(subset=["XH", "KCH"], how="any")
    del_missing_key = before - len(df)

    # 数值列范围校验：KCCJ、BY1、JDCJ
    def drop_out_of_range(col: str, low: float, high: float) -> int:
        if col not in df.columns:
            return 0
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s.notna() & (~s.between(low, high))
        n_bad = bad.sum()
        if n_bad:
            df.drop(df.index[bad], inplace=True)
        return int(n_bad)

    del_kccj_range = drop_out_of_range("KCCJ", 0, 100)
    del_by1_range = drop_out_of_range("BY1", 0, 100)
    del_jdcj_range = drop_out_of_range("JDCJ", 0, 5)

    # BY1 与 JDCJ 同时空或为负：删除
    by1 = (
        pd.to_numeric(df["BY1"], errors="coerce")
        if "BY1" in df.columns
        else pd.Series(index=df.index, dtype=float)
    )
    jdcj = (
        pd.to_numeric(df["JDCJ"], errors="coerce")
        if "JDCJ" in df.columns
        else pd.Series(index=df.index, dtype=float)
    )
    invalid_pair = (by1.isna() | (by1 < 0)) & (jdcj.isna() | (jdcj < 0))
    del_by1_jdcj_invalid_pair = int(invalid_pair.sum())
    if del_by1_jdcj_invalid_pair:
        df = df.loc[~invalid_pair].copy()

    # 统计 DJCJ 的“无效情况”（不删，只统计）
    djcj = pd.to_numeric(df.get("DJCJ"), errors="coerce")
    djcj_invalid = djcj.isna() | (djcj < 0)
    n_djcj_invalid = int(djcj_invalid.sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学生成绩 预处理完成")
    print("=" * 50)
    print(f"  清洗前行数:           {n_before}")
    print(f"  删除(缺学号/课程号):  {del_missing_key}")
    print(f"  删除(KCCJ 越界):      {del_kccj_range}")
    print(f"  删除(BY1 越界):       {del_by1_range}")
    print(f"  删除(JDCJ 越界):      {del_jdcj_range}")
    print(f"  删除(BY1+JDCJ 皆空/负): {del_by1_jdcj_invalid_pair}")
    print(f"  DJCJ 无效(空/负, 保留): {n_djcj_invalid}")
    print(f"  清洗后行数:           {n_after}")
    print(f"  总删除行数:           {n_before - n_after}")
    print(f"  输出文件:             {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
