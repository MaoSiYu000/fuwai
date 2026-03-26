# -*- coding: utf-8 -*-
"""
聚类前准备：一键执行（模块化脚本串联）。

它做什么：
- 按顺序运行 code/cluster_prep/ 下的 00~05 脚本
- 输入：output/eda/features_student_term_draft.csv
- 输出：output/cluster_prep/ 下的逐步产物（不会覆盖原文件）

运行方式：
  python code/cluster_prep/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "00_freeze_input.py",
    "01_missing_indicators_and_impute.py",
    "02_robustify_scale.py",
    "02b_drop_low_variance.py",
    "02c_equalize_variance.py",
    "03_drop_redundant.py",
    "04_flag_outliers.py",
    "05_pca_check.py",
]


def main() -> None:
    this_dir = Path(__file__).resolve().parent
    for name in SCRIPTS:
        p = this_dir / name
        if not p.exists():
            print(f"[跳过] 未找到 {p}")
            continue
        print(f"\n>> 运行 {name}")
        ret = subprocess.run([sys.executable, str(p)], cwd=str(this_dir))
        if ret.returncode != 0:
            raise SystemExit(f"[失败] {name} 返回码 {ret.returncode}")

    print("\n全部聚类前准备步骤已完成。请查看 output/cluster_prep/。")


if __name__ == "__main__":
    main()

