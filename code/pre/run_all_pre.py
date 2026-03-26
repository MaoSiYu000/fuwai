# -*- coding: utf-8 -*-
"""
按顺序执行所有预处理脚本，输出到 output/pre/，并在控制台打印各表清洗前后对比。
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "pre_学生基本信息.py",
    "build_id_map.py",
    "pre_学生成绩.py",
    "pre_学生选课信息.py",
    "pre_学生作业提交记录.py",
    "pre_学生签到记录.py",
    "pre_学生体能考核.py",
    "pre_学科竞赛.py",
    "pre_学籍异动.py",
    "pre_线上学习综合表现.py",
]


def main():
    pre_dir = Path(__file__).resolve().parent
    for name in SCRIPTS:
        path = pre_dir / name
        if not path.exists():
            print(f"[跳过] 未找到 {path}")
            continue
        print(f"\n>> 运行 {name}")
        ret = subprocess.run([sys.executable, str(path)], cwd=str(pre_dir))
        if ret.returncode != 0:
            print(f"[警告] {name} 返回码 {ret.returncode}")
    print("\n全部预处理脚本已执行完毕。请查看上方各表「清洗前后对比」并更新 工作流程.md 中的结果列。")


if __name__ == "__main__":
    main()
