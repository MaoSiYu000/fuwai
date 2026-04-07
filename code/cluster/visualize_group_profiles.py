"""
群体画像可视化
输入：front/data/group_profile_by_college.csv
       front/data/group_profile_by_major.csv
       front/data/student_profiles.csv（仅用于整体分布）
输出：output/cluster/after/*.png（6 张图）
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# ── 路径 ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONT = os.path.join(BASE, "front", "data")
OUT   = os.path.join(BASE, "output", "cluster", "after")
os.makedirs(OUT, exist_ok=True)

# ── 中文字体（优先 SimHei，其次系统可用中文字体） ─────────────────────────
def _set_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Zen Hei",
                  "Noto Sans CJK SC", "DejaVu Sans"]
    for font in candidates:
        try:
            rcParams["font.family"] = font
            rcParams["axes.unicode_minus"] = False
            plt.figure()
            plt.title("测试")
            plt.close()
            return font
        except Exception:
            continue
    return None

_set_chinese_font()
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 120
rcParams["savefig.dpi"] = 150
rcParams["savefig.bbox"] = "tight"

# ── 模式定义 ─────────────────────────────────────────────────────────────────
MODE_NAMES = {
    0: "参与稳定-学业偏弱型",
    1: "高风险波动型",
    2: "学业优势-参与积极型",
    3: "发展成就突出型",
    4: "学业中上-参与偏低型",
    5: "线上低活跃型",
    6: "学业薄弱-参与偏低型",
    7: "主流均衡型",
}
MODE_COLS  = [f"mode_{i}_pct" for i in range(8)]
MODE_COLORS = [
    "#5B8DB8",  # 0 参与稳定-学业偏弱
    "#D65F5F",  # 1 高风险波动
    "#4CAF7D",  # 2 学业优势-参与积极
    "#F0A500",  # 3 发展成就突出
    "#8E7CC3",  # 4 学业中上-参与偏低
    "#4ABCCC",  # 5 线上低活跃
    "#C47A5A",  # 6 学业薄弱-参与偏低
    "#7CBB6E",  # 7 主流均衡
]
DIM_COLS   = ["dim_academic_mean", "dim_attendance_engagement_mean",
              "dim_homework_behavior_mean", "dim_online_learning_mean",
              "dim_fitness_mean", "dim_development_mean"]
DIM_LABELS = ["学业", "出勤参与", "作业行为", "线上学习", "体能", "发展成就"]
DIM_COLORS_POS = "#5B8DB8"
DIM_COLORS_NEG = "#D65F5F"


# ════════════════════════════════════════════════════════════════════════════
# 工具：按 n_records 加权聚合（跨学期 → 整体画像）
# ════════════════════════════════════════════════════════════════════════════
def weighted_agg(df, group_col):
    """对同一 group 下的多学期数据做 n_records 加权均值。"""
    numeric = [c for c in df.columns if c not in [group_col, "TERM_KEY"]
               and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for name, grp in df.groupby(group_col):
        w = grp["n_records"].fillna(1).values
        total = w.sum()
        row = {group_col: name, "n_records": total}
        for c in numeric:
            if c == "n_records":
                continue
            vals = grp[c].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() == 0:
                row[c] = np.nan
            else:
                row[c] = np.average(vals[mask], weights=w[mask])
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 图 1：整体 mode 分布总览（饼图 + 水平条形图）
# ════════════════════════════════════════════════════════════════════════════
def fig1_overall_mode_distribution(df_student):
    """
    df_student: student_profiles.csv（仅取 mode_id）
    """
    counts = df_student["mode_id"].value_counts().sort_index()
    total  = counts.sum()
    labels = [f"M{i} {MODE_NAMES[i]}" for i in counts.index]
    sizes  = counts.values
    colors = [MODE_COLORS[i] for i in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("各行为模式学生-学期记录分布（整体）", fontsize=14, y=1.01)

    # 左：饼图
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.2))
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              fontsize=8, ncol=2, frameon=False)
    ax.set_title("比例分布", fontsize=11)

    # 右：水平条形图
    ax = axes[1]
    y_pos = range(len(counts))
    bars = ax.barh(list(y_pos), sizes, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"M{i}" for i in counts.index], fontsize=9)
    ax.set_xlabel("学生-学期记录数", fontsize=9)
    ax.set_title("各模式记录数", fontsize=11)
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_width() + total * 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({v/total*100:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, sizes.max() * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUT, "01_overall_mode_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  [图1] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 图 2：各学院 mode 分布（堆叠横条）
# ════════════════════════════════════════════════════════════════════════════
def fig2_college_mode_stacked(df_college):
    agg = weighted_agg(df_college, "XSM")
    agg = agg.sort_values("n_records", ascending=True)

    # 保留有数据的列
    mode_cols = [c for c in MODE_COLS if c in agg.columns]
    # 填 NaN 为 0
    for c in mode_cols:
        agg[c] = agg[c].fillna(0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(agg) * 0.45)))
    fig.suptitle("各学院行为模式分布（跨学期加权）", fontsize=13)

    lefts = np.zeros(len(agg))
    for col, color in zip(mode_cols, MODE_COLORS):
        idx = int(col.split("_")[1])
        vals = agg[col].values * 100
        ax.barh(agg["XSM"].values, vals, left=lefts,
                color=color, edgecolor="white", height=0.7,
                label=f"M{idx} {MODE_NAMES[idx]}")
        # 标注 >8% 的块
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold")
        lefts += vals

    ax.set_xlabel("占比 (%)", fontsize=9)
    ax.set_xlim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, frameon=True,
              framealpha=0.85)

    # 右侧标注 n
    for i, row in enumerate(agg.itertuples()):
        ax.text(102, i, f"n={int(row.n_records):,}", va="center", fontsize=7.5,
                color="#555555")

    plt.tight_layout()
    path = os.path.join(OUT, "02_college_mode_stacked.png")
    plt.savefig(path)
    plt.close()
    print(f"  [图2] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 图 3：各学院 6 维度得分热图
# ════════════════════════════════════════════════════════════════════════════
def fig3_college_dim_heatmap(df_college):
    agg = weighted_agg(df_college, "XSM")
    agg = agg.sort_values("n_records", ascending=False)

    dim_cols = [c for c in DIM_COLS if c in agg.columns]
    if not dim_cols:
        print("  [图3] 跳过：缺少维度列")
        return

    mat = agg[dim_cols].values.astype(float)
    college_names = agg["XSM"].values

    fig, ax = plt.subplots(figsize=(10, max(5, len(college_names) * 0.42)))
    fig.suptitle("各学院行为维度得分热图（正=高于均值，负=低于均值）", fontsize=12)

    vabs = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(len(dim_cols)))
    ax.set_xticklabels(DIM_LABELS[:len(dim_cols)], fontsize=10)
    ax.set_yticks(range(len(college_names)))
    ax.set_yticklabels(college_names, fontsize=9)

    # 数值标注
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(v) > vabs * 0.55 else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    path = os.path.join(OUT, "03_college_dim_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  [图3] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 图 4：Top 25 专业 mode 分布（堆叠横条）
# ════════════════════════════════════════════════════════════════════════════
def fig4_major_mode_stacked(df_major, top_n=25):
    agg = weighted_agg(df_major, "ZYM")
    agg = agg.sort_values("n_records", ascending=False).head(top_n)
    agg = agg.sort_values("n_records", ascending=True)

    mode_cols = [c for c in MODE_COLS if c in agg.columns]
    for c in mode_cols:
        agg[c] = agg[c].fillna(0)

    fig, ax = plt.subplots(figsize=(12, max(7, top_n * 0.42)))
    fig.suptitle(f"Top {top_n} 专业行为模式分布（按人数排序）", fontsize=13)

    lefts = np.zeros(len(agg))
    for col, color in zip(mode_cols, MODE_COLORS):
        idx = int(col.split("_")[1])
        vals = agg[col].values * 100
        ax.barh(agg["ZYM"].values, vals, left=lefts,
                color=color, edgecolor="white", height=0.7,
                label=f"M{idx} {MODE_NAMES[idx]}")
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 10:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=6.5, color="white",
                        fontweight="bold")
        lefts += vals

    ax.set_xlabel("占比 (%)", fontsize=9)
    ax.set_xlim(0, 108)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, ncol=2, frameon=True, framealpha=0.85)
    for i, row in enumerate(agg.itertuples()):
        ax.text(105, i, f"n={int(row.n_records):,}", va="center", fontsize=7,
                color="#555555")

    plt.tight_layout()
    path = os.path.join(OUT, "04_major_mode_stacked.png")
    plt.savefig(path)
    plt.close()
    print(f"  [图4] {path}")


# ════════════════════════════════════════════════════════════════════���═══════
# 图 5：8 个 mode 的维度画像对比（2×4 子图）
# ════════════════════════════════════════════════════════════════════════════
def fig5_mode_dim_profiles():
    """
    直接用文档中的 mode 维度均值（来自 4当前模式说明.md），无需读外部文件。
    这样即使 output 路径不同也能稳定生成。
    """
    mode_dims = {
        0: [-0.544,  0.145, -0.065,  0.184,  0.030, -0.068],
        1: [-2.542, -0.803,  1.202, -0.100, -0.128, -0.088],
        2: [ 0.774,  0.246, -0.094,  0.316,  0.014, -0.006],
        3: [ 0.779,  0.103, -0.016, -0.264,  0.107,  0.814],
        4: [ 0.451, -0.429,  0.015, -0.087, -0.069, -0.034],
        5: [-0.793, -0.486,  0.019, -4.181,  0.016, -0.078],
        6: [-1.399, -0.428,  0.149, -0.253, -0.000, -0.048],
        7: [ 0.425,  0.182, -0.093,  0.247,  0.000,  0.012],
    }
    mode_counts = {0: 2421, 1: 633, 2: 4200, 3: 494, 4: 1654,
                   5: 591, 6: 2876, 7: 7145}
    total = sum(mode_counts.values())

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    fig.suptitle("各行为模式的 6 维度得分画像", fontsize=14, y=1.01)

    x = np.arange(len(DIM_LABELS))

    for idx, ax in enumerate(axes.flat):
        vals = mode_dims[idx]
        colors = [DIM_COLORS_POS if v >= 0 else DIM_COLORS_NEG for v in vals]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, fontsize=8, rotation=20, ha="right")
        ax.set_title(
            f"M{idx}  {MODE_NAMES[idx]}\n"
            f"n={mode_counts[idx]:,}  ({mode_counts[idx]/total*100:.1f}%)",
            fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        # 数值标注
        for bar, v in zip(bars, vals):
            ypos = v + (0.05 if v >= 0 else -0.05)
            va   = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:.2f}", ha="center", va=va, fontsize=7)
        ax.set_ylabel("维度得分", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT, "05_mode_dim_profiles.png")
    plt.savefig(path)
    plt.close()
    print(f"  [图5] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 图 6：各学院风险学生占比（mode 1 + mode 6）
# ════════════════════════════════════════════════════════════════════════════
def fig6_college_risk_bar(df_college):
    agg = weighted_agg(df_college, "XSM")
    agg = agg.sort_values("n_records", ascending=False)

    m1 = agg.get("mode_1_pct", pd.Series(np.zeros(len(agg)), index=agg.index)).fillna(0)
    m6 = agg.get("mode_6_pct", pd.Series(np.zeros(len(agg)), index=agg.index)).fillna(0)
    agg["risk_pct"] = (m1 + m6) * 100
    agg = agg.sort_values("risk_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.45)))
    fig.suptitle("各学院风险模式学生占比（M1 高风险波动 + M6 学业薄弱-参与偏低）", fontsize=12)

    # 分段堆叠：M6 在下，M1 在上
    m6_vals = m6.reindex(agg.index) * 100
    m1_vals = m1.reindex(agg.index) * 100

    bars6 = ax.barh(agg["XSM"].values, m6_vals.values, color="#C47A5A",
                    edgecolor="white", height=0.65, label="M6 学业薄弱-参与偏低型")
    bars1 = ax.barh(agg["XSM"].values, m1_vals.values, left=m6_vals.values,
                    color="#D65F5F", edgecolor="white", height=0.65,
                    label="M1 高风险波动型")

    # 合计标注
    for i, row in enumerate(agg.itertuples()):
        total = row.risk_pct
        ax.text(total + 0.5, i, f"{total:.1f}%", va="center", fontsize=8.5,
                color="#333333")

    ax.set_xlabel("占比 (%)", fontsize=9)
    ax.set_xlim(0, agg["risk_pct"].max() * 1.2 + 3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=True, framealpha=0.85, loc="lower right")

    # 右侧 n
    for i, row in enumerate(agg.itertuples()):
        ax.text(agg["risk_pct"].max() * 1.2 + 2, i,
                f"n={int(row.n_records):,}", va="center", fontsize=7.5,
                color="#666666")

    plt.tight_layout()
    path = os.path.join(OUT, "06_college_risk_bar.png")
    plt.savefig(path)
    plt.close()
    print(f"  [图6] {path}")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("读取数据...")
    df_college = pd.read_csv(os.path.join(FRONT, "group_profile_by_college.csv"),
                             encoding="utf-8-sig")
    df_major   = pd.read_csv(os.path.join(FRONT, "group_profile_by_major.csv"),
                             encoding="utf-8-sig")
    df_student = pd.read_csv(os.path.join(FRONT, "student_profiles.csv"),
                             encoding="utf-8-sig", usecols=["XH", "TERM_KEY", "mode_id"])

    print("生成图表...")
    fig1_overall_mode_distribution(df_student)
    fig2_college_mode_stacked(df_college)
    fig3_college_dim_heatmap(df_college)
    fig4_major_mode_stacked(df_major, top_n=25)
    fig5_mode_dim_profiles()
    fig6_college_risk_bar(df_college)

    print(f"\n全部完成，输出目录：{OUT}")


if __name__ == "__main__":
    main()
