 # -*- coding: utf-8 -*-
"""
学生体能考核 预处理（严格版）
- 删除：缺 XH；SJCJ 或 HSCJ 超出 0–100 范围。
- 保留：CFBFS 即使为 0 也不删，只标为辅助字段。
- 规范：可为后续按学期做体质与学业关联分析提供干净的体测成绩。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学生体能考核.csv"
TARGET = "学生体能考核_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")
    n_before = len(df)

    # 计数器
    del_missing_xh = 0
    del_sjcj_range = 0
    del_hscj_range = 0

    # 1) 删除缺学号
    mask = df["XH"].isna() | (df["XH"].astype(str).str.strip() == "")
    del_missing_xh = int(mask.sum())
    df = df[~mask].copy()

    # 2) SJCJ、HSCJ 范围：0–100，超出删
    for col, counter_name in [("SJCJ", "del_sjcj_range"), ("HSCJ", "del_hscj_range")]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            bad = s.notna() & (~s.between(0, 100))
            n_bad = int(bad.sum())
            if col == "SJCJ":
                del_sjcj_range = n_bad
            else:
                del_hscj_range = n_bad
            if n_bad:
                df = df[~bad].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学生体能考核 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:     {n_before}")
    print(f"  删除(缺学号):   {del_missing_xh}")
    print(f"  删除(SJCJ 越界): {del_sjcj_range}")
    print(f"  删除(HSCJ 越界): {del_hscj_range}")
    print(f"  清洗后行数:     {n_after}")
    print(f"  输出文件:       {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
