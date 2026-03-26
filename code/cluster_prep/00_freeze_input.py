# -*- coding: utf-8 -*-
"""
Step 00：冻结输入特征表（复制一份，便于回溯）。

读取：
- output/eda/features_student_term_draft.csv

输出：
- output/cluster_prep/00_features_frozen.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "eda" / "features_student_term_draft.csv"
OUT_DIR = ROOT / "output" / "cluster_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "00_features_frozen.csv"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 EDA：python code/EDA/run_eda.py")
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[完成] 冻结输入：{OUT_PATH}（rows={len(df)} cols={len(df.columns)}）")


if __name__ == "__main__":
    main()

