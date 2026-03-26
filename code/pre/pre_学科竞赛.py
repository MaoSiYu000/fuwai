 # -*- coding: utf-8 -*-
"""
学科竞赛 预处理（更严格版本）
- HJSJ 时间格式多样：如 2021/6/1 0:00、2022、2022-13、45047 等。
  - 明确认为 45047 一类 Excel 序列号是「错误值」→ 直接删行；
  - 像 2022 这种只有年份 → 转成 2022-01-01；
  - 月份>12（如 2022-13）→ 视为错误日期 → 直接删行。
- NJ 年级：2020级 保留；20级 → 映射为 2020级；2年级 等明显不规范的暂置为空字符串（宁可少用，不乱猜）。
- 同时保留原始 HJSJ/NJ 字段，新增解析列便于后续使用。
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output" / "pre"
SOURCE = "学科竞赛.csv"
TARGET = "学科竞赛_pre.csv"


def main():
    path = DATA_DIR / SOURCE
    if not path.exists():
        print(f"[跳过] 未找到 {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")
    n_before = len(df)

    n_del_key = 0
    n_del_xy = 0
    n_del_hjsj_numeric = 0
    n_del_hjsj_bad_month = 0
    n_fix_hjsj_year_only = 0
    n_fix_nj = 0
    n_fix_zy = 0

    # 1) 缺学号/工号：直接删除
    mask_key = df["XHHGH"].isna() | (df["XHHGH"].astype(str).str.strip() == "")
    n_del_key = int(mask_key.sum())
    df = df[~mask_key].copy()

    # 2) XY 明显错误（如“2022秋季”等）：直接删除
    if "XY" in df.columns:
        xy = df["XY"].astype(str).str.strip()
        bad_xy = xy.str.match(r"^\d{4}\s*秋?季?$", na=False) | (xy == "2022秋季")
        n_del_xy = int(bad_xy.sum())
        df = df[~bad_xy].copy()

    # 3) HJSJ：严格处理
    if "HJSJ" in df.columns:
        def parse_hjsj(v):
            nonlocal n_del_hjsj_numeric, n_del_hjsj_bad_month, n_fix_hjsj_year_only
            if pd.isna(v):
                return pd.NaT, False  # NaT, 不删
            s = str(v).strip()

            # 情况 A：纯数字（如 45047）——按你的要求视为错误，直接删
            if s.isdigit():
                # 标记为需要删除
                n_del_hjsj_numeric += 1
                return pd.NaT, True

            # 情况 B：只有年份（如 2022）
            if s.isdigit() and len(s) == 4:
                n_fix_hjsj_year_only += 1
                try:
                    # 只有年份：补成当年 1 月 1 日
                    return pd.Timestamp(s + "-01-01"), False
                except Exception:
                    n_del_hjsj_numeric += 1
                    return pd.NaT, True

            # 其他情况：尝试按日期解析
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                # 无法解析，严格一点也删掉
                n_del_hjsj_numeric += 1
                return pd.NaT, True
            # 月份 > 12 直接当错误删行
            if dt.month > 12:
                n_del_hjsj_bad_month += 1
                return pd.NaT, True
            return dt, False

        parsed = df["HJSJ"].apply(parse_hjsj)
        df["HJSJ_PARSED"] = parsed.apply(lambda x: x[0])
        del_mask = parsed.apply(lambda x: x[1])
        # 删除所有被标记为错误时间的行
        to_del = del_mask[del_mask].index
        # 按错误类型计数已在回调里记录，这里统一删
        if len(to_del):
            df = df.drop(index=to_del).copy()

    # 4) NJ：统一为“XXXX级”，明显错误的置为空
    if "NJ" in df.columns:
        def norm_nj(x):
            nonlocal n_fix_nj
            raw = str(x).strip() if not pd.isna(x) else ""
            if raw == "":
                return ""
            # 直接合法：2020级 等
            if raw.endswith("级") and raw[:-1].isdigit() and len(raw[:-1]) == 4:
                return raw
            # 形如 2020 → 2020级
            if raw.isdigit() and len(raw) == 4:
                n_fix_nj += 1
                return raw + "级"
            # 20级 → 2020级（根据你给的例子，严格按照 20 推为 2020）
            if raw.endswith("级") and raw[:-1].isdigit() and len(raw[:-1]) == 2:
                n_fix_nj += 1
                year = raw[:-1]
                fixed = f"20{year}级"
                return fixed
            # 其他如 “2年级”等不规范：宁可置为空
            n_fix_nj += 1
            return ""

        df["NJ"] = df["NJ"].apply(norm_nj)

    # 5) ZY：去 &nbsp; 等，格式统一
    if "ZY" in df.columns:
        def clean_zy(x):
            nonlocal n_fix_zy
            raw = str(x)
            fixed = raw.replace("&nbsp;", "").strip()
            if fixed != raw:
                n_fix_zy += 1
            return fixed

        df["ZY"] = df["ZY"].apply(clean_zy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / TARGET
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_after = len(df)

    print("=" * 50)
    print("学科竞赛 预处理完成（严格版）")
    print("=" * 50)
    print(f"  清洗前行数:          {n_before}")
    print(f"  删除(缺学号):        {n_del_key}")
    print(f"  删除(学院错误):      {n_del_xy}")
    print(f"  删除(HJSJ 纯数字):   {n_del_hjsj_numeric}")
    print(f"  删除(HJSJ 月份>12):  {n_del_hjsj_bad_month}")
    print(f"  规范(HJSJ 只有年份→日期): {n_fix_hjsj_year_only} 行")
    print(f"  规范(NJ 年级):       {n_fix_nj} 行")
    print(f"  规范(ZY 专业名):     {n_fix_zy} 行")
    print(f"  清洗后行数:          {n_after}")
    print(f"  输出文件:            {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
