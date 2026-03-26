 # -*- coding: utf-8 -*-
"""
学生签到记录 预处理（严格版）
- 删除：缺 LOGIN_NAME；ACTIVE_STATE 不在合法集合；ROLE 非 1/3。
- 规范：ATTEND_TIME 视为毫秒时间戳，解析为 ATTEND_DATETIME；字符串 "null" 统一当成缺失。
- 保留：XYID/XYMC/MAJORID/MAJORNAME/CLASSID/CLASSNAME 等院系/专业/班级信息，不因空白删除，后续通过学号跨表填补。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学生签到记录.csv"
TARGET = "学生签到记录_pre.csv"
STU_INFO_PRE = "学生基本信息_pre.csv"

VALID_STATE = {0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12}
VALID_ROLE = {1.0, 3.0, 1, 3}


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    n_before = len(df)

    # 计数器
    del_missing_login = 0
    del_invalid_state = 0
    del_invalid_role = 0
    parsed_attend_ok = 0
    parsed_attend_fail = 0
    fill_xy_from_basic = 0
    fill_major_from_basic = 0

    # 1) 删除缺 LOGIN_NAME
    mask = df["LOGIN_NAME"].isna() | (df["LOGIN_NAME"].astype(str).str.strip() == "")
    del_missing_login = int(mask.sum())
    df = df[~mask].copy()

    # 2) ACTIVE_STATE 限定
    if "ACTIVE_STATE" in df.columns:
        df["ACTIVE_STATE"] = pd.to_numeric(df["ACTIVE_STATE"], errors="coerce")
        mask_valid_state = df["ACTIVE_STATE"].isna() | df["ACTIVE_STATE"].isin(VALID_STATE)
        del_invalid_state = int((~mask_valid_state).sum())
        df = df[mask_valid_state].copy()

    # 3) ROLE 限定 1 或 3（允许空，按“学生”为主，可不强制填补）
    if "ROLE" in df.columns:
        df["ROLE"] = pd.to_numeric(df["ROLE"], errors="coerce")
        mask_valid_role = df["ROLE"].isna() | df["ROLE"].isin(VALID_ROLE)
        del_invalid_role = int((~mask_valid_role).sum())
        df = df[mask_valid_role].copy()

    # 4) 统一将 XYMC 为 "null" 的视为缺失（不删行，只规范）
    if "XYMC" in df.columns:
        xy_str = df["XYMC"].astype(str)
        df["XYMC"] = xy_str.where(~xy_str.str.strip().eq('"null"'), "")

    # 5) 利用 学生基本信息_pre.csv 填补院系/专业名称（按 LOGIN_NAME = XH）
    info_path = OUT_DIR.parent / "学生基本信息_pre.csv"
    if info_path.exists():
        info_df = pd.read_csv(info_path, encoding="utf-8-sig")
        # 只需要 XH, XSM, ZYM
        info_df = info_df[["XH", "XSM", "ZYM"]]
        info_df = info_df.drop_duplicates(subset=["XH"], keep="first")
        # 以 LOGIN_NAME 为键左连接
        merged = df.merge(info_df, how="left", left_on="LOGIN_NAME", right_on="XH", suffixes=("", "_basic"))
        # 对于 XYMC 为空且 XSM 有值，用 XSM 填补
        if "XYMC" in merged.columns:
            xy_na = (merged["XYMC"].astype(str).str.strip() == "") & merged["XSM"].notna()
            fill_xy_from_basic = int(xy_na.sum())
            merged.loc[xy_na, "XYMC"] = merged.loc[xy_na, "XSM"]
        # 对于 MAJORNAME 为空且 ZYM 有值，用 ZYM 填补
        if "MAJORNAME" in merged.columns:
            mj_na = (merged["MAJORNAME"].astype(str).str.strip() == "") & merged["ZYM"].notna()
            fill_major_from_basic = int(mj_na.sum())
            merged.loc[mj_na, "MAJORNAME"] = merged.loc[mj_na, "ZYM"]
        # 丢掉辅助列 XH_basic
        merged = merged.drop(columns=["XH"], errors="ignore")
        df = merged

    # 6) ATTEND_TIME 毫秒时间戳转 datetime（新增一列便于后续分析，保留原列）
    if "ATTEND_TIME" in df.columns:
        ms = pd.to_numeric(df["ATTEND_TIME"], errors="coerce")
        attend_dt = pd.to_datetime(ms, unit="ms", errors="coerce")
        parsed_attend_ok = int(attend_dt.notna().sum())
        parsed_attend_fail = int(attend_dt.isna().sum())
        df["ATTEND_DATETIME"] = attend_dt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学生签到记录 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:          {n_before}")
    print(f"  删除(缺账号):        {del_missing_login}")
    print(f"  删除(状态非法):      {del_invalid_state}")
    print(f"  删除(角色非法):      {del_invalid_role}")
    print(f"  ATTEND_TIME 解析成功: {parsed_attend_ok}")
    print(f"  ATTEND_TIME 解析失败: {parsed_attend_fail}")
    print(f"  填补XYMC(来自基本信息): {fill_xy_from_basic}")
    print(f"  填补MAJORNAME(来自基本信息): {fill_major_from_basic}")
    print(f"  清洗后行数:         {n_after}")
    print(f"  输出文件:           {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
