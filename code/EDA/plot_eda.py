# -*- coding: utf-8 -*-
"""
EDA 可视化（用于快速看数据结构）。

它做什么：
- 读取 output/eda/ 的统计结果与特征草稿表
- 生成一组 PNG 图片，输出到 output/eda/image/

输出图（可能因数据缺失略有不同）：
- missing_rate_top.png：缺失率最高的特征 Top20
- corr_heatmap.png：相关性热力图（Spearman）
- k_screening.png：K=4..8 的 silhouette/DBI 指标曲线
- k_stability.png：不同随机种子下 ARI 稳定性
- pca_scatter.png：特征表 PCA 2D 散点（抽样）

运行方式：
  python code/EDA/plot_eda.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "output" / "eda"
IMG_DIR = EDA_DIR / "image"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _set_cn_font():
    """
    解决中文标题显示为方框的问题：
    - 优先选择 Windows 常见中文字体（微软雅黑/黑体/宋体）
    - 关闭 unicode_minus 的负号乱码
    """
    import matplotlib as mpl

    mpl.rcParams["axes.unicode_minus"] = False
    # 按优先级尝试（环境没有也不会报错）
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


def plot_missing_rates():
    path = EDA_DIR / "feature_ranges.csv"
    if not path.exists():
        return
    import matplotlib.pyplot as plt

    _set_cn_font()
    fr = pd.read_csv(path, encoding="utf-8-sig")
    fr = fr.sort_values("missing_rate", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fr["feature"][::-1], fr["missing_rate"][::-1])
    ax.set_xlabel("missing_rate")
    ax.set_title("EDA: 特征缺失率 Top20")
    _savefig(fig, IMG_DIR / "missing_rate_top.png")


def plot_corr_heatmap():
    path = EDA_DIR / "corr_matrix.csv"
    if not path.exists():
        return
    import matplotlib.pyplot as plt

    _set_cn_font()
    corr = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("EDA: Spearman 相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _savefig(fig, IMG_DIR / "corr_heatmap.png")


def plot_k_screening():
    km_path = EDA_DIR / "k_screening_metrics.csv"
    st_path = EDA_DIR / "k_stability.csv"
    if not km_path.exists() or not st_path.exists():
        return
    import matplotlib.pyplot as plt

    _set_cn_font()
    km = pd.read_csv(km_path, encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(km["k"], km["silhouette"], marker="o", label="silhouette")
    ax1.set_xlabel("k")
    ax1.set_ylabel("silhouette")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(km["k"], km["davies_bouldin"], marker="s", color="tab:red", label="DBI")
    ax2.set_ylabel("davies_bouldin (lower better)")
    ax1.set_title("EDA: K 粗筛指标（silhouette / DBI）")
    _savefig(fig, IMG_DIR / "k_screening.png")

    st = pd.read_csv(st_path, encoding="utf-8-sig")
    fig2, ax = plt.subplots(figsize=(9, 5))
    for k, g in st.groupby("k"):
        ax.plot(g["seed"], g["ari_vs_seed42"], marker="o", label=f"k={k}")
    ax.set_xlabel("seed")
    ax.set_ylabel("ARI vs seed42")
    ax.set_title("EDA: K 稳定性（不同随机种子）")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    _savefig(fig2, IMG_DIR / "k_stability.png")


def plot_pca_scatter(sample_n: int = 4000):
    feat_path = EDA_DIR / "features_student_term_draft.csv"
    if not feat_path.exists():
        return

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _set_cn_font()
    df = pd.read_csv(feat_path, encoding="utf-8-sig", low_memory=False)
    keep = [c for c in df.columns if c not in ["XH", "TERM_KEY"]]
    x = df[keep].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))

    if len(x) > sample_n:
        x = x.sample(sample_n, random_state=42)

    # 标准化后 PCA（只用于可视化诊断）
    mu = x.mean()
    sd = x.std(ddof=0).replace(0, 1)
    xz = (x - mu) / sd
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(xz.values)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.35)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title("EDA: PCA 2D 散点（抽样）")
    ax.grid(True, alpha=0.2)
    _savefig(fig, IMG_DIR / "pca_scatter.png")


def main():
    plot_missing_rates()
    plot_corr_heatmap()
    plot_k_screening()
    plot_pca_scatter()
    print("EDA 可视化完成：output/eda/image/")


if __name__ == "__main__":
    main()

