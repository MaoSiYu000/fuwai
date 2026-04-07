# -*- coding: utf-8 -*-
"""
Step 01: 稳健补齐（分层 + 保守阈值）。

策略：
1) 仅处理数值特征；
2) 每个特征先生成缺失标记 is_missing_*；
3) 分层补齐顺序：
   - 学生历史中位数（按 XH）
   - 同学期中位数（按 TERM_KEY）
   - 全局中位数
4) 风险控制：
   - missing_rate > 0.60 的列标记为 high_missing_feature=1（提醒谨慎使用）
   - 输出每列补齐来源占比，便于回溯
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "eda" / "features_student_term_draft.csv"
OUT_DIR = ROOT / "output" / "data_impute"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURES = OUT_DIR / "01_features_imputed_safe.csv"
OUT_LOG = OUT_DIR / "01_impute_log.csv"

KEY_COLS = ["XH", "TERM_KEY"]
HIGH_MISSING_THRESHOLD = 0.60


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{IN_PATH}")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    out = df.copy()

    feat_cols = [c for c in out.columns if c not in KEY_COLS]
    log_rows: list[dict] = []

    for c in feat_cols:
        s_raw = out[c]
        s_num = pd.to_numeric(s_raw, errors="coerce")
        missing_mask = s_num.isna()
        miss_rate = float(missing_mask.mean())

        # 缺失指示列
        miss_col = f"is_missing_{c}"
        out[miss_col] = missing_mask.astype(int)

        # 只处理可数值化的特征
        parse_rate = float(s_num.notna().mean())
        if parse_rate < 0.5:
            out[c] = s_raw
            log_rows.append(
                {
                    "feature": c,
                    "missing_rate": miss_rate,
                    "numeric_parse_rate": parse_rate,
                    "imputed_total": 0,
                    "from_xh_median": 0,
                    "from_term_median": 0,
                    "from_global_median": 0,
                    "high_missing_feature": int(miss_rate > HIGH_MISSING_THRESHOLD),
                    "note": "skip_non_numeric_feature",
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

        # 3) 全局中位数（兜底）
        gmed = float(np.nanmedian(work.to_numpy(dtype=float))) if work.notna().any() else 0.0
        use3 = work.isna()
        work.loc[use3] = gmed

        out[c] = work
        after_missing = int(work.isna().sum())

        log_rows.append(
            {
                "feature": c,
                "missing_rate": miss_rate,
                "numeric_parse_rate": parse_rate,
                "imputed_total": int(before_missing - after_missing),
                "from_xh_median": int(use1.sum()),
                "from_term_median": int(use2.sum()),
                "from_global_median": int(use3.sum()),
                "high_missing_feature": int(miss_rate > HIGH_MISSING_THRESHOLD),
                "note": "ok",
            }
        )

    pd.DataFrame(log_rows).sort_values(["high_missing_feature", "missing_rate"], ascending=[False, False]).to_csv(
        OUT_LOG, index=False, encoding="utf-8-sig"
    )
    out.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

    print(f"[OK] 稳健补齐后特征：{OUT_FEATURES}")
    print(f"[OK] 补齐日志：{OUT_LOG}")


if __name__ == "__main__":
    main()

