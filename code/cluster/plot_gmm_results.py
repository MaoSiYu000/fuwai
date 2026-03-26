# -*- coding: utf-8 -*-
"""
GMM 软聚类结果可视化（通俗效果图）。

读取：
- output/cluster/student_term_gmm_probs.csv

输出到：
- output/cluster/image/

输出图：
- gmm_pca2_scatter.png：二维投影散点（颜色=簇；透明度=置信度 p_max；基于“特征空间”PCA）
- gmm_pca2_regions.png：二维投影 + 每簇半透明“不规则覆盖区域”（2D KDE 等密度线）
- gmm_umap2_scatter.png：UMAP 2D 可视化散点（仅展示用，常更“成团”）
- gmm_tsne2_scatter.png：t-SNE 2D 可视化散点（仅展示用，作为对照）
- gmm_pca3_scatter.png：PCA 3D 可视化散点（仅展示用）
- gmm_umap3_scatter.png：UMAP 3D 可视化散点（仅展示用）
- gmm_cluster_size.png：各簇样本数/占比
- gmm_uncertainty.png：p_max、entropy 分布（包含“放大尾部”的视图，避免只看到一个柱）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "output" / "cluster" / "student_term_gmm_probs.csv"
FEAT_PATH = ROOT / "output" / "cluster_prep" / "03_features_pruned.csv"
OUT_DIR = ROOT / "output" / "cluster" / "image"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIS_SAMPLE = 8000  # 仅用于可视化抽样，避免过慢/过密


def _set_cn_font():
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


def plot_pca2_scatter(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _set_cn_font()

    # 你反馈：scatter 用原来的 tab10 更舒服；region 用更“正色”的调色板更好看
    # 所以这里准备两套颜色：
    # - scatter_cmap：tab10（用于 gmm_pca2_scatter.png）
    # - region_colors：高饱和度（用于 gmm_pca2_regions.png 的区域/轮廓/散点）
    scatter_cmap = plt.get_cmap("tab10")

    vivid_colors = [
        "#1F4FFF",  # vivid blue
        "#FF8A00",  # vivid orange
        "#00B050",  # vivid green
        "#FF2D2D",  # vivid red
        "#8E2BFF",  # vivid purple
        "#00C2C7",  # vivid cyan
        "#FF2DB2",  # vivid magenta
        "#7A7A7A",  # neutral gray
        "#B59B00",  # mustard
        "#5A2D0C",  # brown
    ]
    from matplotlib.colors import ListedColormap
    region_cmap = ListedColormap(vivid_colors, name="vivid10")

    # 关键修正：
    # 之前用“概率向量”做 PCA，概率几乎是 one-hot 时会天然落在近似一条线上，视觉上像“点很少且都挤在一条线”
    # 这里改为：用“聚类用的特征空间”（03_features_pruned）做 PCA 投影，然后用 cluster_map 上色、p_max 控制透明度
    if not FEAT_PATH.exists():
        raise FileNotFoundError(f"未找到特征输入：{FEAT_PATH}，请先运行 code/cluster_prep/run_all.py")

    feat_df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig", low_memory=False)
    key_cols = [c for c in ["XH", "TERM_KEY"] if c in feat_df.columns]
    miss_cols = [c for c in feat_df.columns if c.startswith("is_missing_")]
    feat_cols = [c for c in feat_df.columns if c not in key_cols]  # 缺失指示也参与投影
    x = feat_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).to_numpy()

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(x)

    cluster = df["cluster_map"].astype(int).to_numpy()
    pmax = df["p_max"].astype(float).to_numpy()
    alpha = 0.15 + 0.85 * np.clip(pmax, 0.0, 1.0)  # 置信度越高越不透明

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=cluster,
        cmap=scatter_cmap,
        s=10,
        alpha=alpha,
        linewidths=0,
    )
    ax.set_title("GMM 软聚类效果图（特征空间PCA；颜色=簇；透明度=p_max）")
    ax.set_xlabel("PCA1（基于聚类特征）")
    ax.set_ylabel("PCA2（基于聚类特征）")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cluster_map")
    _savefig(fig, OUT_DIR / "gmm_pca2_scatter.png")

    # 额外输出：用“半透明区域”表示每簇在2D里的覆盖范围
    # 改进：不再用椭圆，而用 2D KDE（核密度估计）画“不规则等密度线区域”，更接近论文常见画法
    # 说明：这是可视化近似（在 2D 投影平面上估计密度），不是模型在原始高维空间的严格等密度面
    from sklearn.neighbors import KernelDensity

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sc2 = ax2.scatter(
        xy[:, 0],
        xy[:, 1],
        c=cluster,
        cmap=region_cmap,
        s=8,
        alpha=0.22,
        linewidths=0,
    )

    # 网格范围（留出边界）
    x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
    y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
    pad_x = 0.05 * (x1 - x0 + 1e-9)
    pad_y = 0.05 * (y1 - y0 + 1e-9)
    xg = np.linspace(x0 - pad_x, x1 + pad_x, 240)
    yg = np.linspace(y0 - pad_y, y1 + pad_y, 240)
    xx, yy = np.meshgrid(xg, yg)
    grid = np.c_[xx.ravel(), yy.ravel()]

    def _kde_regions(points: np.ndarray, color):
        if points.shape[0] < 30:
            return
        # 带宽用数据尺度自适应（粗略经验：用两维 std 的中位数乘系数）
        bw = float(np.median(np.std(points, axis=0, ddof=0)) * 0.35)
        bw = max(bw, 0.15)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        kde.fit(points)
        z = np.exp(kde.score_samples(grid)).reshape(xx.shape)
        # 用密度分位数做等密度线（核心区 + 覆盖区）
        z_flat = z.ravel()
        q_core = float(np.quantile(z_flat, 0.92))
        q_cover = float(np.quantile(z_flat, 0.82))
        levels = [q_cover, q_core, z.max()]
        # 半透明填充：让重叠区域通过“混色”更明显
        ax2.contourf(xx, yy, z, levels=levels, colors=[color, color], alpha=0.14, antialiased=True)
        # 核心等密度线：更实一些，便于辨认边界
        ax2.contour(xx, yy, z, levels=[q_core], colors=[color], linewidths=1.4, alpha=0.85)

    k_list = sorted(np.unique(cluster).tolist())
    for k in k_list:
        pts = xy[cluster == k]
        color = vivid_colors[k % len(vivid_colors)]
        _kde_regions(pts, color=color)

    ax2.set_title("GMM 聚类覆盖区域（2D PCA；KDE不规则区域=簇范围）")
    ax2.set_xlabel("PCA1（基于聚类特征）")
    ax2.set_ylabel("PCA2（基于聚类特征）")
    cb2 = fig2.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("cluster_map")
    _savefig(fig2, OUT_DIR / "gmm_pca2_regions.png")

    # 同一份 2D 坐标也可复用做 UMAP/t-SNE，但为了更直观，这里基于原始特征空间再做 2D 可视化
    # 说明：这些降维图“好看”不代表模型更好，仅用于展示结构与重叠程度
    _plot_umap_tsne(df=df, feat_df=feat_df, key_cols=key_cols)


def _sample_for_vis(
    df_probs: pd.DataFrame, feat_df: pd.DataFrame, key_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """按行对齐抽样：返回 probs 子集和对应的特征矩阵。"""
    # 这里假设 df_probs 与 feat_df 行顺序一致（同一批学生-学期），若未来不一致应改为按 key merge
    n = min(len(df_probs), len(feat_df))
    dfp = df_probs.iloc[:n].copy()
    feat_cols = [c for c in feat_df.columns if c not in key_cols]
    x = feat_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).to_numpy()
    x = x[:n]

    if n > VIS_SAMPLE:
        rs = np.random.RandomState(42)
        idx = rs.choice(n, size=VIS_SAMPLE, replace=False)
        idx.sort()
        return dfp.iloc[idx].reset_index(drop=True), x[idx]
    return dfp.reset_index(drop=True), x


def _plot_umap_tsne(df: pd.DataFrame, feat_df: pd.DataFrame, key_cols: list[str]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    _set_cn_font()

    df_s, x = _sample_for_vis(df_probs=df, feat_df=feat_df, key_cols=key_cols)
    cluster = df_s["cluster_map"].astype(int).to_numpy()
    pmax = df_s["p_max"].astype(float).to_numpy()
    alpha = 0.15 + 0.85 * np.clip(pmax, 0.0, 1.0)

    # 先做一次 PCA 到 15 维，加速 t-SNE / UMAP（常规做法）
    x15 = PCA(n_components=min(15, x.shape[1]), random_state=42).fit_transform(x)

    # 1) t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    xy_t = tsne.fit_transform(x15)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xy_t[:, 0], xy_t[:, 1], c=cluster, cmap=plt.get_cmap("tab10"), s=9, alpha=alpha, linewidths=0)
    ax.set_title(f"t-SNE 2D 展示（抽样 n={len(df_s)}；颜色=簇；透明度=p_max）")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cluster_map")
    _savefig(fig, OUT_DIR / "gmm_tsne2_scatter.png")

    # 2) UMAP（可选依赖：umap-learn）
    try:
        import umap  # type: ignore
    except Exception:
        print("[跳过] 未安装 umap-learn，无法生成 gmm_umap2_scatter.png。可运行：pip install umap-learn")
        return

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )
    xy_u = reducer.fit_transform(x15)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sc2 = ax2.scatter(xy_u[:, 0], xy_u[:, 1], c=cluster, cmap=plt.get_cmap("tab10"), s=9, alpha=alpha, linewidths=0)
    ax2.set_title(f"UMAP 2D 展示（抽样 n={len(df_s)}；颜色=簇；透明度=p_max）")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    cb2 = fig2.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("cluster_map")
    _savefig(fig2, OUT_DIR / "gmm_umap2_scatter.png")

    # 3D PCA（直接在 x15 上做 PCA3）
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x3 = PCA(n_components=3, random_state=42).fit_transform(x15)
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    sc3 = ax3.scatter(
        x3[:, 0],
        x3[:, 1],
        x3[:, 2],
        c=cluster,
        cmap=plt.get_cmap("tab10"),
        s=7,
        alpha=alpha,
        linewidths=0,
    )
    ax3.set_title(f"PCA 3D 展示（抽样 n={len(df_s)}；颜色=簇；透明度=p_max）")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    cb3 = fig3.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.08)
    cb3.set_label("cluster_map")
    _savefig(fig3, OUT_DIR / "gmm_pca3_scatter.png")

    # 3D UMAP（同样依赖 umap-learn）
    reducer3 = umap.UMAP(
        n_components=3,
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )
    u3 = reducer3.fit_transform(x15)
    fig4 = plt.figure(figsize=(9, 7))
    ax4 = fig4.add_subplot(111, projection="3d")
    sc4 = ax4.scatter(
        u3[:, 0],
        u3[:, 1],
        u3[:, 2],
        c=cluster,
        cmap=plt.get_cmap("tab10"),
        s=7,
        alpha=alpha,
        linewidths=0,
    )
    ax4.set_title(f"UMAP 3D 展示（抽样 n={len(df_s)}；颜色=簇；透明度=p_max）")
    ax4.set_xlabel("UMAP1")
    ax4.set_ylabel("UMAP2")
    ax4.set_zlabel("UMAP3")
    cb4 = fig4.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.08)
    cb4.set_label("cluster_map")
    _savefig(fig4, OUT_DIR / "gmm_umap3_scatter.png")


def plot_cluster_size(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    _set_cn_font()

    cnt = df["cluster_map"].value_counts().sort_index()
    pct = cnt / cnt.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cnt.index.astype(str), cnt.values)
    for i, (k, v) in enumerate(cnt.items()):
        ax.text(i, v, f"{pct.loc[k]*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("GMM 各簇样本数（柱）与占比（文字）")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, OUT_DIR / "gmm_cluster_size.png")


def plot_uncertainty(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    _set_cn_font()

    pmax = df["p_max"].astype(float).to_numpy()
    ent = df["entropy"].astype(float).to_numpy()

    # 关键修正：
    # 如果绝大多数样本非常确定（p_max≈1、entropy≈0），普通直方图会只剩“一个大柱子”
    # 所以这里给“全局 + 放大尾部”两层视图
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].hist(pmax, bins=30, color="tab:blue", alpha=0.85)
    axes[0, 0].set_title("p_max 全局分布（越接近1越确定）")
    axes[0, 0].set_xlabel("p_max")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].hist(ent, bins=30, color="tab:orange", alpha=0.85)
    axes[0, 1].set_title("entropy 全局分布（越大越边界/混合）")
    axes[0, 1].set_xlabel("entropy")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].hist(pmax, bins=30, range=(0.95, 1.0), color="tab:blue", alpha=0.85)
    axes[1, 0].set_title("p_max 尾部放大（0.95~1.00）")
    axes[1, 0].set_xlabel("p_max")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].hist(ent, bins=30, range=(0.0, 0.1), color="tab:orange", alpha=0.85)
    axes[1, 1].set_title("entropy 尾部放大（0~0.10）")
    axes[1, 1].set_xlabel("entropy")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True, alpha=0.25)

    _savefig(fig, OUT_DIR / "gmm_uncertainty.png")


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"未找到输入：{IN_PATH}，请先运行 code/cluster/run_gmm_student_term.py")

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig", low_memory=False)
    if "cluster_map" not in df.columns or "p_max" not in df.columns:
        raise ValueError("输入缺少 cluster_map/p_max，可能不是预期的 GMM 概率输出表。")

    plot_pca2_scatter(df)
    plot_cluster_size(df)
    plot_uncertainty(df)
    print("GMM 可视化完成：output/cluster/image/")


if __name__ == "__main__":
    main()

