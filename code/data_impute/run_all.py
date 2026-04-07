# -*- coding: utf-8 -*-
"""
数据补齐模块一键运行入口。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "00_missing_audit.py",
    "01_safe_impute.py",
    "02_impute_impact_report.py",
]


def main() -> None:
    for s in SCRIPTS:
        p = THIS_DIR / s
        print(f"\n[RUN] {s}")
        result = subprocess.run([sys.executable, str(p)], cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("\n[OK] data_impute 全流程执行完成。")


if __name__ == "__main__":
    main()

