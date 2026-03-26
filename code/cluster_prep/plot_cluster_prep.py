# -*- coding: utf-8 -*-
"""
聚类前准备可视化（用于查看处理效果）。

它做什么：
- 读取 output/cluster_prep/ 的结果文件
- 生成一组 PNG 图片，输出到 output/cluster_prep/image/

输出图：
- outlier_rates_top.png：异常点标注的 hi/lo 比例 Top20
- pca_scree.png：PCA 解释方差曲线（来自 05_pca_explained_variance.csv）

运行方式：
  python code/cluster_prep/plot_cluster_prep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "cluster_prep"
IMG_DIR = OUT_DIR / "image"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _set_cn_font():
    """解决中文显示为方框问题（Windows 常见字体优先）。"""
    import matplotlib as mpl

    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"[图] {path}")


def plot_outlier_rates():
    p = OUT_DIR / "04_outlier_summary.csv"
    if not p.exists():
        return
    import matplotlib.pyplot as plt

    _set_cn_font()
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["max_rate"] = df[["hi_rate", "lo_rate"]].max(axis=1)
    top = df.sort_values("max_rate", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["max_rate"][::-1])
    ax.set_xlabel("max(hi_rate, lo_rate)")
    ax.set_title("cluster_prep: 异常点比例 Top20（p01/p99 标注）")
    _savefig(fig, IMG_DIR / "outlier_rates_top.png")


def plot_pca_scree():
    p = OUT_DIR / "05_pca_explained_variance.csv"
    if not p.exists():
        return
    import matplotlib.pyplot as plt

    _set_cn_font()
    df = pd.read_csv(p, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(df) + 1), df["cum_explained_variance_ratio"], marker="o")
    ax.set_xlabel("n_components")
    ax.set_ylabel("cum_explained_variance_ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("cluster_prep: PCA 累计解释方差（诊断用）")
    _savefig(fig, IMG_DIR / "pca_scree.png")


def main():
    plot_outlier_rates()
    plot_pca_scree()
    print("cluster_prep 可视化完成：output/cluster_prep/image/")


if __name__ == "__main__":
    main()

