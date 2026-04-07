# -*- coding: utf-8 -*-
"""
Step 01b: 白名单补齐（V2）。

思路：
- 仅对白名单特征做补齐，避免“全量补齐”扰动过大；
- 白名单默认规则（可调）：
  1) 0 < missing_rate <= 0.40
  2) numeric_parse_rate >= 0.95
  3) n_unique_nonnull >= 20
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "eda" / "features_student_term_draft.csv"
IN_AUDIT = ROOT / "output" / "data_impute" / "00_missing_audit.csv"
OUT_DIR = ROOT / "output" / "data_impute"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURES = OUT_DIR / "01b_features_imputed_whitelist.csv"
OUT_WHITELIST = OUT_DIR / "01b_impute_whitelist.csv"
OUT_LOG = OUT_DIR / "01b_impute_log.csv"

KEY_COLS = ["XH", "TERM_KEY"]

MISS_MAX = 0.40
PARSE_MIN = 0.95
UNIQUE_MIN = 20


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{IN_PATH}")
    if not IN_AUDIT.exists():
        raise FileNotFoundError(f"未找到审计文件：{IN_AUDIT}，请先运行 00_missing_audit.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    audit = pd.read_csv(IN_AUDIT, encoding="utf-8-sig", low_memory=False)

    whitelist = audit[
        (audit["missing_rate"] > 0)
        & (audit["missing_rate"] <= MISS_MAX)
        & (audit["numeric_parse_rate"] >= PARSE_MIN)
        & (audit["n_unique_nonnull"] >= UNIQUE_MIN)
    ].copy()
    wl_cols = set(whitelist["feature"].tolist())
    whitelist.to_csv(OUT_WHITELIST, index=False, encoding="utf-8-sig")

    out = df.copy()
    feat_cols = [c for c in out.columns if c not in KEY_COLS]
    log_rows: list[dict] = []

    for c in feat_cols:
        s_raw = out[c]
        s_num = pd.to_numeric(s_raw, errors="coerce")
        missing_mask = s_num.isna()
        parse_rate = float(s_num.notna().mean())
        miss_rate = float(missing_mask.mean())

        # 所有列都保留缺失指示
        out[f"is_missing_{c}"] = missing_mask.astype(int)

        if c not in wl_cols:
            out[c] = s_raw
            log_rows.append(
                {
                    "feature": c,
                    "in_whitelist": 0,
                    "missing_rate": miss_rate,
                    "numeric_parse_rate": parse_rate,
                    "imputed_total": 0,
                    "from_xh_median": 0,
                    "from_term_median": 0,
                    "from_global_median": 0,
                    "note": "skip_not_in_whitelist",
                }
            )
            continue

        work = s_num.copy()
        before_missing = int(work.isna().sum())

        # 1) XH 历史中位数
        if "XH" in out.columns:
            xh_med = out.assign(_v=work).groupby("XH", dropna=False)["_v"].transform("median")
            use1 = work.isna() & xh_med.notna()
            work.loc[use1] = xh_med.loc[use1]
        else:
            use1 = pd.Series(False, index=work.index)

        # 2) TERM_KEY 同期中位数
        if "TERM_KEY" in out.columns:
            term_med = out.assign(_v=work).groupby("TERM_KEY", dropna=False)["_v"].transform("median")
            use2 = work.isna() & term_med.notna()
            work.loc[use2] = term_med.loc[use2]
        else:
            use2 = pd.Series(False, index=work.index)

        # 3) 全局中位数兜底
        gmed = float(np.nanmedian(work.to_numpy(dtype=float))) if work.notna().any() else 0.0
        use3 = work.isna()
        work.loc[use3] = gmed

        out[c] = work
        after_missing = int(work.isna().sum())
        log_rows.append(
            {
                "feature": c,
                "in_whitelist": 1,
                "missing_rate": miss_rate,
                "numeric_parse_rate": parse_rate,
                "imputed_total": int(before_missing - after_missing),
                "from_xh_median": int(use1.sum()),
                "from_term_median": int(use2.sum()),
                "from_global_median": int(use3.sum()),
                "note": "ok",
            }
        )

    out.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")
    pd.DataFrame(log_rows).to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

    print(f"[OK] 白名单表：{OUT_WHITELIST}")
    print(f"[OK] 白名单补齐结果：{OUT_FEATURES}")
    print(f"[OK] 白名单补齐日志：{OUT_LOG}")


if __name__ == "__main__":
    main()

