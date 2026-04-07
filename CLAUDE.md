# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于多源数据的大学生行为分析与干预模型设计 — a data science competition project that mines multi-source university student behavioral data to identify behavioral patterns (modes/subtypes) via two-layer soft clustering.

**Language**: Python 3. No build system, no package manager files. Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Running the Pipeline

Each stage must be run sequentially. There is no master runner.

```bash
# Stage 1 – Preprocessing (9 table-specific cleaning scripts)
python code/pre/run_all_pre.py

# Stage 2 – EDA + feature engineering (outputs features_student_term_draft.csv)
python code/EDA/run_eda.py

# Stage 3 – Pre-clustering preparation (6-step modular pipeline)
python code/cluster_prep/run_all.py

# Stage 4 – Clustering
python code/cluster/build_dimension_scores.py
python code/cluster/run_gmm_modes_and_subtypes.py

# Experimental: data imputation (concluded not beneficial, kept as reference)
python code/data_impute/run_all.py
```

## Pipeline Architecture

```
data/{data1,data2,data3}/  (ignored by git)
         ↓
code/pre/          → output/pre/           Cleaned CSVs, ID mapping
         ↓
code/EDA/          → output/eda/           36 student-term features
         ↓
code/cluster_prep/ → output/cluster_prep/  Preprocessed feature matrix
    00_freeze_input.py       → freeze inputs for reproducibility
    01_missing_indicators…   → add missingness flags + impute
    02_robustify_scale.py    → winsorize + RobustScaler
    02b_drop_low_variance    → remove near-zero variance features
    02c_equalize_variance    → normalize variance across features
    03_drop_redundant.py     → remove highly correlated features
    04_flag_outliers.py      → flag outliers (never delete)
    05_pca_check.py          → PCA diagnostics
         ↓
code/cluster/      → output/cluster/       Final results
    build_dimension_scores.py         → 6D dimension score space
    run_gmm_modes_and_subtypes.py     → two-layer GMM (mode + subtype)
    analyze_mode_subtype_profiles.py  → profile naming
    build_group_profiles.py           → class/major aggregates
    evaluate_soft_clustering_stability.py → K stability metrics
```

## Key Design Decisions

**Primary granularity**: `(XH, TERM_KEY)` — student × academic term. The main output table is `output/cluster/student_term_modes_and_subtypes.csv` (~20,014 rows).

**Two-layer GMM clustering**:
- Mode layer: GMM in 6D dimension score space, K=8 (evaluated K=4..12; K=6–8 most stable)
- Subtype layer: GMM per mode in original feature space, K=2–4 subtypes per mode
- Output columns: `mode_id`, `p_mode_*`, `subtype_id`, `p_subtype_*`, `p_max`, `entropy`

**Data handling philosophy**: preserve information — flag outliers rather than delete, add missingness indicators rather than silently impute, use RobustScaler (median/IQR) over StandardScaler.

**Imputation experiments** (`code/data_impute/`) concluded that imputation worsens clustering stability (ARI drops, cluster imbalance increases). The baseline (no imputation) is the current standard.

**Random state**: `RANDOM_STATE = 42` is used throughout.

## Key Identifiers

- `XH`: student ID (primary key across all tables)
- `LOGIN_NAME`: online platform login (used in assignment/online-learning tables)
- `TERM_KEY`: constructed term identifier, e.g., `"2023-1"` = Fall 2023

## Documentation Files

| File | Purpose |
|------|---------|
| `00方案选择.md` | **Final technical approach** — the authoritative spec |
| `0任务梳理.md` | 8-stage competition requirements and success metrics |
| `0数据集.md` | Full dataset catalog for all 30+ source tables |
| `3聚类.md` | Clustering methodology, K evaluation rationale |
| `4当前模式说明.md` | Description of the current K=8 mode solution |
| `目前做的事情.md` | Work log (11 batches), tracks what has been completed |
| `output/README.md` | Guide to all output directories and files |
