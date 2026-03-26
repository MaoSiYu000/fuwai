 # -*- coding: utf-8 -*-
"""
线上学习（综合表现） 预处理（严格版）
- 删除：缺 LOGIN_NAME；BFB 超出 0–100；ROLEID 非 1/3/7。
- 保留：院系/专业/班级字段原样，用于后续画像。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "线上学习（综合表现）.csv"
TARGET = "线上学习（综合表现）_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")
    n_before = len(df)

    # 计数器
    del_missing_login = 0
    del_bfb_out_of_range = 0
    del_invalid_role = 0

    # 1) 删除缺 LOGIN_NAME
    mask = df["LOGIN_NAME"].isna() | (df["LOGIN_NAME"].astype(str).str.strip() == "")
    del_missing_login = int(mask.sum())
    df = df[~mask].copy()

    # 2) BFB 0–100
    if "BFB" in df.columns:
        bfb = pd.to_numeric(df["BFB"], errors="coerce")
        bad = bfb.notna() & (~bfb.between(0, 100))
        del_bfb_out_of_range = int(bad.sum())
        df = df[~bad].copy()

    # 3) ROLEID 合法性：允许 1(老师)、3(学生)、7(管理员)，其他删除
    if "ROLEID" in df.columns:
        role = pd.to_numeric(df["ROLEID"], errors="coerce")
        valid_roles = {1, 3, 7}
        mask_role = role.isna() | role.isin(valid_roles)
        del_invalid_role = int((~mask_role).sum())
        df = df[mask_role].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("线上学习（综合表现） 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:      {n_before}")
    print(f"  删除(缺账号):    {del_missing_login}")
    print(f"  删除(BFB 越界):  {del_bfb_out_of_range}")
    print(f"  删除(ROLEID 非法): {del_invalid_role}")
    print(f"  清洗后行数:      {n_after}")
    print(f"  输出文件:        {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
