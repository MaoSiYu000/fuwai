# -*- coding: utf-8 -*-
"""
把 data/data2、data/data3 下的所有 xlsx 转成 csv，并删除原 xlsx。

设计目标：
- 不依赖具体表名，只按扩展名扫描
- 每个 xlsx 只取第一张 sheet（与你们表头读取逻辑一致）
- 输出 csv 使用 utf-8-sig，确保中文表头可读
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA2_DIR = DATA_DIR / "data2"
DATA3_DIR = DATA_DIR / "data3"

targets = [DATA2_DIR, DATA3_DIR]


def convert_one_xlsx(xlsx_path: Path) -> tuple[bool, str]:
    csv_path = xlsx_path.with_suffix(".csv")
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        return False, f"读取/写入失败: {type(e).__name__}: {e}"

    # 简单校验：csv 是否存在
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False, "csv 未生成或为空，未删除 xlsx"

    # 删除原 xlsx
    try:
        os.remove(xlsx_path)
    except Exception as e:
        return False, f"csv 生成成功但删除 xlsx 失败: {type(e).__name__}: {e}"

    return True, "ok"


def main() -> None:
    xlsx_files: list[Path] = []
    for base in targets:
        if not base.exists():
            continue
        xlsx_files.extend(sorted(base.rglob("*.xlsx")))

    if not xlsx_files:
        print("未在 data2/data3 发现任何 .xlsx")
        return

    print(f"共发现 .xlsx：{len(xlsx_files)} 个")
    ok = 0
    fail = 0
    errors: list[str] = []

    for i, p in enumerate(xlsx_files, start=1):
        rel = p.relative_to(ROOT)
        print(f"[{i}/{len(xlsx_files)}] 转换：{rel}")
        success, msg = convert_one_xlsx(p)
        if success:
            ok += 1
            print("  -> 成功，已删除 xlsx")
        else:
            fail += 1
            print(f"  -> 失败：{msg}")
            errors.append(f"{rel}: {msg}")

    print(f"完成：成功 {ok}，失败 {fail}")

    out_summary = ROOT / "output" / "xlsx_to_csv_data2_data3_summary.txt"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"total={len(xlsx_files)} ok={ok} fail={fail}",
            "",
            *errors,
        ]
    )
    out_summary.write_text(text, encoding="utf-8-sig")
    print(f"错误/摘要：{out_summary}")


if __name__ == "__main__":
    main()

