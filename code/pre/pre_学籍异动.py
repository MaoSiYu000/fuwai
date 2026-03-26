 # -*- coding: utf-8 -*-
"""
学籍异动 预处理（严格版）
- 删除：缺学号 XH；YDLBDM/YDYYDM 超出合理代码范围的记录。
- 解析：SPRQ、BY1 解析为日期，新增 *_parsed 列，统计解析成功/失败数量。
- 保留：院系/专业/年级等信息不因空白删除，后续在画像和建模阶段再利用。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学籍异动.csv"
TARGET = "学籍异动_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")
    n_before = len(df)

    # 计数器
    del_missing_xh = 0
    del_ydlb_invalid = 0
    del_ydyy_invalid = 0
    parsed_sprq_ok = 0
    parsed_sprq_fail = 0
    parsed_by1_ok = 0
    parsed_by1_fail = 0

    # 1) 删除缺学号
    if "XH" in df.columns:
        mask = df["XH"].isna() | (df["XH"].astype(str).str.strip() == "")
        del_missing_xh = int(mask.sum())
        df = df[~mask].copy()

    # 2) 合法性检查：异动类别/原因代码
    #   YDLBDM：异动类别 1,2,3,4,5,11,12,13,14,15,16,17,18（题面中的所有代码）
    if "YDLBDM" in df.columns:
        ydlb = pd.to_numeric(df["YDLBDM"], errors="coerce")
        valid_ydlb = {1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18}
        mask_valid_ydlb = ydlb.isna() | ydlb.isin(valid_ydlb)
        del_ydlb_invalid = int((~mask_valid_ydlb).sum())
        df = df[mask_valid_ydlb].copy()

    #   YDYYDM：异动原因 1–31（题面给出的完整原因代码）
    if "YDYYDM" in df.columns:
        ydyy = pd.to_numeric(df["YDYYDM"], errors="coerce")
        mask_valid_ydyy = ydyy.isna() | ydyy.between(1, 31)
        del_ydyy_invalid = int((~mask_valid_ydyy).sum())
        df = df[mask_valid_ydyy].copy()

    # 3) SPRQ、BY1 解析为日期（新增列保留原列）
    for col, ok_counter, fail_counter in [
        ("SPRQ", "parsed_sprq_ok", "parsed_sprq_fail"),
        ("BY1", "parsed_by1_ok", "parsed_by1_fail"),
    ]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            ok = int(parsed.notna().sum())
            fail = int(len(parsed) - ok)
            if col == "SPRQ":
                parsed_sprq_ok, parsed_sprq_fail = ok, fail
            else:
                parsed_by1_ok, parsed_by1_fail = ok, fail
            df[col + "_parsed"] = parsed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学籍异动 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:          {n_before}")
    print(f"  删除(缺学号):        {del_missing_xh}")
    print(f"  删除(YDLBDM 非法):   {del_ydlb_invalid}")
    print(f"  删除(YDYYDM 非法):   {del_ydyy_invalid}")
    print(f"  SPRQ 解析成功:       {parsed_sprq_ok}")
    print(f"  SPRQ 解析失败:       {parsed_sprq_fail}")
    print(f"  BY1(生效日期)解析成功: {parsed_by1_ok}")
    print(f"  BY1(生效日期)解析失败: {parsed_by1_fail}")
    print(f"  清洗后行数:          {n_after}")
    print(f"  输出文件:            {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
