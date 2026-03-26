 # -*- coding: utf-8 -*-
"""
学生选课信息 预处理（严格版）
- 删除：缺 XH 或 KCH 的记录。
- 规范/填补：
  - KCLB 课程类别：优先按同一课程号 KCH 的众数填补；若整门课全为空，再用 1（基础课）填补。
  - XKRQ 选课日期：解析为日期；统计解析成功/失败的数量（失败暂不删行）。
- 日志：逐类打印删除/填补/解析统计。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学生选课信息.csv"
TARGET = "学生选课信息_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    n_before = len(df)

    # 计数器
    del_missing_key = 0
    fill_kclb_by_mode = 0
    fill_kclb_by_default = 0
    parsed_xkrq_ok = 0
    parsed_xkrq_fail = 0

    # 1) 删除缺学号或课程号
    before = len(df)
    df = df.dropna(subset=["XH", "KCH"], how="any")
    del_missing_key = before - len(df)

    # 2) KCLB 课程类别填补：先按 KCH 众数，再默认 1
    if "KCLB" in df.columns:
        # 先转为数值，便于统计众数
        df["KCLB"] = pd.to_numeric(df["KCLB"], errors="coerce")
        # 统计每门课的众数（排除 NaN）
        kclb_mode_by_kch = (
            df.dropna(subset=["KCLB"])
            .groupby("KCH")["KCLB"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
        )

        # 逐行填补
        mask_na = df["KCLB"].isna()
        for idx in df[mask_na].index:
            kch = df.at[idx, "KCH"]
            mode_val = kclb_mode_by_kch.get(kch, pd.NA)
            if pd.notna(mode_val):
                df.at[idx, "KCLB"] = mode_val
                fill_kclb_by_mode += 1
            else:
                df.at[idx, "KCLB"] = 1  # 默认基础课
                fill_kclb_by_default += 1

    # 3) XKRQ 选课日期解析
    if "XKRQ" in df.columns:
        parsed = pd.to_datetime(df["XKRQ"], errors="coerce")
        parsed_xkrq_ok = int(parsed.notna().sum())
        parsed_xkrq_fail = int(parsed.isna().sum())
        df["XKRQ_parsed"] = parsed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学生选课信息 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:             {n_before}")
    print(f"  删除(缺学号/课程号):    {del_missing_key}")
    print(f"  填补(KCLB 众数):        {fill_kclb_by_mode}")
    print(f"  填补(KCLB 默认=1):      {fill_kclb_by_default}")
    print(f"  解析XKRQ 成功:          {parsed_xkrq_ok}")
    print(f"  解析XKRQ 失败(置为 NaT): {parsed_xkrq_fail}")
    print(f"  清洗后行数:             {n_after}")
    print(f"  输出文件:               {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
