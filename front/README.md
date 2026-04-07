# 前端数据说明文档

> 本目录由后端聚类模块生成，供前端网页开发直接使用。
> 数据来源：基于 20,014 条「学生-学期」记录的两层软聚类结果（GMM，K=8 模式）。

---

## 目录结构

```
front/
└── data/
    ├── student_profiles.csv          # 学生个体画像（主表，20,014 行）
    ├── group_profile_by_class.csv    # 班级群体画像（1,591 行）
    ├── group_profile_by_major.csv    # 专业群体画像（578 行）
    ├── group_profile_by_college.csv  # 学院群体画像（158 行）
    ├── mode_definitions.json         # 8 个模式的名称/说明/维度画像
    ├── subtype_definitions.json      # 32 个子类的名称/说明
    └── dim_score_formulas.json       # 6 个维度分数的计算公式说明
```

---

## 核心概念

### 两层聚类结构

系统对每个「学生-学期」做了两层软聚类（概率聚类，而非硬分组）：

- **模式层（mode）**：8 个大类，基于 6 个维度分数，反映学生整体行为模式。
- **子类层（subtype）**：每个模式下 4 个子类（共 32 个），进一步细化同模式内部差异。

**subtype 是确定的附加信息**，前端可以放心接入。subtype 在 student_profiles.csv 里只有 3 列（subtype_id、subtype_name、subtype_note），是附加信息，不影响 mode 层、6 个维度分、原始指标、群体画像等核心结构。

**前端开发建议**：
- 优先做好 mode 层 8 个大类（画像核心）
- subtype 作为锦上添花，建议只在"个人详情页"展开，群体页面不涉及
- 软聚类的关键字段是 `p_mode_0`~`p_mode_7` + `mode_pmax` + `mode_entropy`，可用于展示"归属置信度"（如 pmax < 0.6 时提示"该学期行为模式较混合"）

**注意**：mode 1 下 sub 101 只有 31 人、mode 3 下 sub 301 只有 34 人，这些极小子类展示时建议合并到更大子类或直接用 mode 级别描述。

### 6 个维度分数


| 字段名                         | 中文名  | 含义说明                 |
| --------------------------- | ---- | -------------------- |
| `dim_academic`              | 学业   | 成绩/绩点/挂科综合，正数=优于平均   |
| `dim_attendance_engagement` | 出勤参与 | 出勤率与课外活动参与，正数=积极     |
| `dim_homework_behavior`     | 作业行为 | 作业提交与完成情况，正数=积极      |
| `dim_online_learning`       | 线上学习 | 线上平台学习完成度，正数=高活跃     |
| `dim_fitness`               | 体能   | 体测成绩综合，正数=优于平均       |
| `dim_development`           | 发展成就 | 奖学金/竞赛/四六级，正数=投入/成就高 |


> 所有维度分数的基准为 0（全体平均），**正数代表高于平均，负数代表低于平均**，绝对值越大差距越明显。

---

## 文件详细说明

---

### 1. `student_profiles.csv` — 学生个体画像

**用途**：按学号（XH）查询某个学生某学期的行为画像，支持"个人主页"功能。

**维度**：20,014 行 × 44 列（每个学生-学期一行）


| 字段名                         | 类型  | 说明                                   |
| --------------------------- | --- | ------------------------------------ |
| `XH`                        | 字符串 | 学号（主键之一）                             |
| `TERM_KEY`                  | 字符串 | 学期，格式 `YYYY-YYYY-N`，如 `2022-2023-1`  |
| `XB`                        | 字符串 | 性别（男/女）                              |
| `XSM`                       | 字符串 | 学院名称                                 |
| `ZYM`                       | 字符串 | 专业名称                                 |
| `CLASS_NAME`                | 字符串 | 班级名称（部分学生有值）                         |
| `mode_id`                   | 整数  | 模式编号（0–7）                            |
| `mode_name`                 | 字符串 | 模式名称，如 `主流均衡型`                       |
| `mode_pmax`                 | 浮点  | 模式归属概率（0–1），越高表示归属越确定                |
| `mode_margin`               | 浮点  | 最高概率与第二概率之差，越大越明确                    |
| `mode_entropy`              | 浮点  | 模式概率分布熵，越低越确定，越高越模糊                  |
| `p_mode_0` – `p_mode_7`     | 浮点  | 该学生-学期归属各模式的概率（8 列，合计≈1），反映完整概率分布    |
| `subtype_id`                | 整数  | 子类编号，格式 `mode_id*100 + 子类序号`，如 `702` |
| `subtype_name`              | 字符串 | 子类名称，如 `主流均衡基线群（人数最多）`               |
| `subtype_note`              | 字符串 | 机器可读子类标签，如 `mode7_sub2`              |
| `mode_evidence`             | 字符串 | **模式归属依据文本**（规则自动生成，可直接展示，也可作为 LLM prompt 上下文）例：`①学业明显偏低（课程成绩均值68分，挂科率18%）；②出勤参与偏低（出勤率61%）。` |
| `dim_academic`              | 浮点  | 学业维度分数（相对全体均值）                       |
| `dim_attendance_engagement` | 浮点  | 出勤参与维度分数                             |
| `dim_homework_behavior`     | 浮点  | 作业行为维度分数                             |
| `dim_online_learning`       | 浮点  | 线上学习维度分数                             |
| `dim_fitness`               | 浮点  | 体能维度分数                               |
| `dim_development`           | 浮点  | 发展成就维度分数                             |
| `kccj_mean`                 | 浮点  | 课程成绩均值（百分制）                          |
| `kccj_fail_rate`            | 浮点  | 挂科率（0–1）                             |
| `jdcj_mean`                 | 浮点  | 绩点均值                                 |
| `by1_mean`                  | 浮点  | 百分制成绩均值                              |
| `att_present_rate`          | 浮点  | 出勤率（0–1）                             |
| `att_absent_rate`           | 浮点  | 缺勤率（0–1）                             |
| `att_event_cnt`             | 浮点  | 参与活动数（该学期累计）                         |
| `hw_submit_cnt`             | 浮点  | 作业提交数（该学期累计）                         |
| `hw_ungraded_rate`          | 浮点  | 作业未批阅率（0–1）                          |
| `hw_duration_median`        | 浮点  | 作业耗时中位数（秒）                           |
| `online_bfb`                | 浮点  | 线上学习完成比例（0–100）                      |
| `sch_amt_sum_term`          | 浮点  | 该学期奖学金金额（元）                          |
| `comp_cnt_term`             | 浮点  | 该学期竞赛次数                              |
| `cet_score_max`             | 浮点  | 四六级最高分                               |
| `fit3_zf_mean`              | 浮点  | 体测综合分均值                              |


> **NaN 说明**：部分字段缺失是正常现象（如某学期没有体测/竞赛记录）。前端展示时可处理为"暂无数据"。

**典型查询场景**：

```python
# 按学号查该学生所有学期画像
student_data = df[df['XH'] == 'pjxyqwbj585']

# 按学号+学期查某学期详情
row = df[(df['XH'] == 'pjxyqwbj585') & (df['TERM_KEY'] == '2022-2023-1')]
```

---

### 2. `group_profile_by_class.csv` — 班级群体画像

**用途**：展示某个班级在某学期的模式分布和整体画像，支持"班级看板"。

**维度**：1,591 行（班级 × 学期组合）


| 字段名                              | 类型  | 说明                 |
| -------------------------------- | --- | ------------------ |
| `CLASS_NAME`                     | 字符串 | 班级名称               |
| `TERM_KEY`                       | 字符串 | 学期                 |
| `n_records`                      | 整数  | 该班该学期的学生人数         |
| `mode_0_pct`                     | 浮点  | 模式 0（参与稳定-学业偏弱型）占比 |
| `mode_1_pct`                     | 浮点  | 模式 1（高风险波动型）占比     |
| `mode_2_pct`                     | 浮点  | 模式 2（学业优势-参与积极型）占比 |
| `mode_3_pct`                     | 浮点  | 模式 3（发展成就突出型）占比    |
| `mode_4_pct`                     | 浮点  | 模式 4（学业中上-参与偏低型）占比 |
| `mode_5_pct`                     | 浮点  | 模式 5（线上低活跃型）占比     |
| `mode_6_pct`                     | 浮点  | 模式 6（学业薄弱-参与偏低型）占比 |
| `mode_7_pct`                     | 浮点  | 模式 7（主流均衡型）占比      |
| `dominant_mode_id`               | 整数  | 人数最多的模式编号          |
| `dominant_mode_name`             | 字符串 | 人数最多的模式名称          |
| `dim_academic_mean`              | 浮点  | 班级学业维度均值           |
| `dim_attendance_engagement_mean` | 浮点  | 班级出勤参与均值           |
| `dim_homework_behavior_mean`     | 浮点  | 班级作业行为均值           |
| `dim_online_learning_mean`       | 浮点  | 班级线上学习均值           |
| `dim_fitness_mean`               | 浮点  | 班级体能均值             |
| `dim_development_mean`           | 浮点  | 班级发展成就均值           |
| `kccj_mean_avg`                  | 浮点  | 班级课程成绩均值均值（即均值的均值） |
| `kccj_fail_rate_avg`             | 浮点  | 班级挂科率均值            |
| *(其余原始指标均值)*                     | 浮点  | 命名规则：`{原始指标名}_avg` |


`group_profile_by_major.csv`（按专业）和 `group_profile_by_college.csv`（按学院）结构相同，分别以 `ZYM`（专业名）和 `XSM`（学院名）替换 `CLASS_NAME` 作为分组键。

---

### 3. `mode_definitions.json` — 模式说明

**用途**：渲染 8 个模式的标签、描述和维度雷达图。

**结构**（数组，每个元素是一个模式对象）：

```json
{
  "mode_id": 7,
  "name": "主流均衡型",
  "description": "各维度接近整体平均，是规模最大、最典型的均衡中间群体。",
  "size": 7145,
  "pct": 0.357,
  "dim_profile": {
    "dim_academic": 0.4253,
    "dim_academic_name": "学业",
    "dim_attendance_engagement": 0.1819,
    "dim_attendance_engagement_name": "出勤参与",
    "dim_homework_behavior": -0.0933,
    "dim_homework_behavior_name": "作业行为",
    "dim_online_learning": 0.2467,
    "dim_online_learning_name": "线上学习",
    "dim_fitness": 0.0,
    "dim_fitness_name": "体能",
    "dim_development": 0.0118,
    "dim_development_name": "发展成就"
  }
}
```

**8 个模式速查**：


| mode_id | 名称         | 规模    | 占比     | 核心特征             |
| ------- | ---------- | ----- | ------ | ---------------- |
| 0       | 参与稳定-学业偏弱型 | 2,421 | 12.10% | 学业偏弱，出勤参与尚可      |
| 1       | 高风险波动型     | 633   | 3.16%  | 学业最差，多项指标波动，风险最高 |
| 2       | 学业优势-参与积极型 | 4,200 | 20.99% | 学业突出，参与积极，整体均衡偏好 |
| 3       | 发展成就突出型    | 494   | 2.47%  | 奖学金/竞赛/四六级远高于平均  |
| 4       | 学业中上-参与偏低型 | 1,654 | 8.26%  | 学业尚可，课外参与偏低      |
| 5       | 线上低活跃型     | 591   | 2.95%  | 线上学习活跃度显著最低      |
| 6       | 学业薄弱-参与偏低型 | 2,876 | 14.37% | 学业偏弱，参与偏低，需关注干预  |
| 7       | 主流均衡型      | 7,145 | 35.70% | 各维度接近均值，规模最大     |


---

### 4. `subtype_definitions.json` — 子类说明

**用途**：渲染子类标签（在已知 mode_id 的基础上进一步细化解释）。

**结构**（数组，每个元素是一个子类对象）：

```json
{
  "mode_id": 7,
  "subtype_id": 702,
  "mode_name": "主流均衡型",
  "name": "主流均衡基线群（人数最多）",
  "size": 6232,
  "pct_in_mode": 0.8722
}
```

**子类编号规则**：`subtype_id = mode_id × 100 + 子类序号`，例如：

- `702` = mode 7 的第 2 个子类
- `100` = mode 1 的第 0 个子类

> 注意：mode 0 的子类编号为 0、1、2、3（不带百位），其余模式均为三位数。

---

### 5. `dim_score_formulas.json` — 维度分数公式说明

**用途**：在学生画像页面展示"这个分数是怎么算出来的"的透明度说明。

**结构**：

```json
{
  "dim_id": "dim_academic",
  "name_cn": "学业",
  "description": "学业综合得分。正数表示高于全体平均，负数表示低于全体平均。越高代表成绩与绩点越好、挂科越少。",
  "components": [
    {"metric": "kccj_mean",      "name_cn": "课程成绩均值",   "weight": 0.40, "direction": "正向（越高越好）"},
    {"metric": "jdcj_mean",      "name_cn": "绩点均值",       "weight": 0.30, "direction": "正向（越高越好）"},
    {"metric": "by1_mean",       "name_cn": "百分制成绩均值", "weight": 0.30, "direction": "正向（越高越好）"},
    {"metric": "kccj_fail_rate", "name_cn": "挂科率",         "weight": 0.50, "direction": "负向（越高越差）"}
  ]
}
```

---

## 前端使用建议

### 学生个人主页

推荐展示元素：

1. **模式标签**：`mode_name`（如"主流均衡型"）+ `mode_pmax`（置信度，如 0.87 = 87%）
2. **归属依据**：`mode_evidence`（直接显示即可，无需额外处理）
3. **子类说明**：`subtype_name`（一句话描述该学生在模式内的细分特征）
4. **维度雷达图**：6 个 `dim_`* 分数（以 0 为基准，展示高于/低于平均的幅度）
5. **原始指标卡片**：如挂科率、出勤率、作业提交数等（有值时显示，NaN 时显示"暂无数据"）
6. **学期对比**：按 `TERM_KEY` 展示同一学生的跨学期变化折线图

> **`mode_evidence` 的两种用法**：
> - **直接展示**：文字已是人类可读格式，直接放在画像卡片里作"分析依据"展示。无需调用 AI，和其他字段一样快。
> - **喂给 LLM（未来拓展）**：把 `mode_evidence` 拼进 prompt，大模型在有数据支撑的基础上润色扩写，避免 AI 凭空生成与数据不符的描述。

**置信度说明建议**（向用户解释）：


| mode_pmax 范围 | 建议文案     |
| ------------ | -------- |
| ≥ 0.80       | 高度确定     |
| 0.60 – 0.80  | 较为确定     |
| 0.40 – 0.60  | 中等确定     |
| < 0.40       | 可能跨越多个模式 |


### 群体看板（班级/专业/学院）

推荐展示元素：

1. **模式分布饼图/条形图**：`mode_0_pct` – `mode_7_pct` + 对应模式名
2. **主要模式标签**：`dominant_mode_name`
3. **维度均值雷达图**：6 个 `dim_*_mean` 字段（与全体均值 0 对比）
4. **关键指标均值表**：`kccj_mean_avg`、`att_present_rate_avg`、`online_bfb_avg` 等
5. **跨学期趋势**：按 `TERM_KEY` 过滤后展示模式分布变化

### 数据更新说明

- 数据由聚类模块生成，**当前版本为 mode=8 口径**（已确认稳定）
- 当模型调整后，重新运行 `python code/cluster/build_frontend_data.py` 即可刷新 `front/data/` 下所有文件
- **数据结构（字段名、文件名）设计为稳定版本**，模型迭代后字段不变，前端无需改代码

---

## 常见问题

**Q: `mode_pmax` 很低（比如 0.40）怎么显示？**
A: 说明该学生处于多个模式的边界区，这是软聚类的正常现象。可在界面上显示"该学生行为模式跨越多个类型"，或同时展示 top-2 模式。

**Q: 某字段是 NaN / null 怎么处理？**
A: 该学生在该学期没有此类数据（如没有参加竞赛、没有四六级记录等）。展示为"—"或"暂无数据"即可。

**Q: 班级 CLASS_NAME 缺失怎么办？**
A: 班级信息从出勤/作业记录中提取，少部分学生的班级无法匹配，显示为空值时可省略。

**Q: 同一个学生有多行（多学期），个人主页怎么选？**
A: 可展示全部学期列表供选择，默认选最近一学期（TERM_KEY 最大值）。

**Q: 6 个维度分数的"0"代表什么？**
A: 代表该学生该维度得分等于全体 20,014 条记录的平均水平。正值高于平均，负值低于平均。

---

## AI 大模型集成（未来拓展）

当前数据结构**已为接入大模型个性化解释预留好了所有必要字段**，后端无需改动结构，前端只需在合适时机调用 LLM API 并传入以下上下文即可。

### 为什么现有结构够用？

大模型要生成"这个学生为什么属于这个模式/他的问题在哪"这类解释，需要三类信息：


| 信息类型                  | 已有字段                                                                | 用途                |
| --------------------- | ------------------------------------------------------------------- | ----------------- |
| **模式语义**（"你属于什么类型"）   | `mode_name`、`subtype_name`、`mode_definitions.json` 里的 `description` | 给大模型提供文字化的分组含义    |
| **维度定量**（"你在哪些方面高/低"） | `dim_academic` ~ `dim_development`（6个值，0为基准）                        | 让大模型用具体数字描述强弱     |
| **原始指标**（"具体数据是多少"）   | `kccj_mean`、`att_present_rate`、`online_bfb` 等 15 个指标                | 让大模型引用真实数字，避免空洞描述 |
| **归属置信度**（"分类有多确定"）   | `mode_pmax`、`mode_entropy`、`p_mode_0`~`p_mode_7`                    | 让大模型对边界学生措辞谨慎     |


### 建议传给大模型的字段（最小集）

从 `student_profiles.csv` 取一行，至少传以下字段：

```
【身份】  XH, TERM_KEY, XSM（学院）, ZYM（专业）
【模式】  mode_name, mode_pmax
         若 mode_pmax < 0.6，还需传 p_mode_* 中排名第二的模式名（从 mode_definitions.json 查）
【依据】  mode_evidence  ← 直接描述了关键维度的高/低及具体数值，是 prompt 最重要的上下文
【子类】  subtype_name
【维度】  dim_academic, dim_attendance_engagement, dim_homework_behavior,
         dim_online_learning, dim_fitness, dim_development
【原始】  kccj_mean, kccj_fail_rate, att_present_rate, hw_submit_cnt,
         online_bfb, sch_amt_sum_term, comp_cnt_term, cet_score_max, fit3_zf_mean
```

### 参考 Prompt 结构

```
你是一位学生行为数据分析师，请根据以下数据为该学生生成个性化学习行为分析报告。

【基本信息】
学号: {XH}，学期: {TERM_KEY}，学院: {XSM}，专业: {ZYM}

【行为模式】
该学生被识别为「{mode_name}」（归属置信度: {mode_pmax:.0%}）
细分类型：{subtype_name}
模式说明：{从 mode_definitions.json 的 description 字段取}
归属依据：{mode_evidence}  ← 直接从 student_profiles.csv 取，已包含关键维度与数值

【6维度表现】（相对全体平均，0=平均水平，正数=高于平均，负数=低于平均）
- 学业:     {dim_academic:.2f}
- 出勤参与: {dim_attendance_engagement:.2f}
- 作业行为: {dim_homework_behavior:.2f}
- 线上学习: {dim_online_learning:.2f}
- 体能:     {dim_fitness:.2f}
- 发展成就: {dim_development:.2f}

【具体指标】
- 课程成绩均值: {kccj_mean:.1f}分，挂科率: {kccj_fail_rate:.1%}
- 出勤率: {att_present_rate:.1%}，参与活动数: {att_event_cnt:.0f}次
- 作业提交数: {hw_submit_cnt:.0f}次
- 线上学习完成比例: {online_bfb:.1f}
- 该学期奖学金: {sch_amt_sum_term}元，竞赛次数: {comp_cnt_term:.0f}次
- 四六级最高分: {cet_score_max}，体测综合分: {fit3_zf_mean:.1f}

请生成约200字的个性化分析，指出该学生的突出优势、潜在风险，并给出2~3条具体建议。
```

### 置信度处理建议


| `mode_pmax` 范围 | 大模型 prompt 措辞建议                    |
| -------------- | ---------------------------------- |
| ≥ 0.80         | 直接说"该学生属于…模式"                      |
| 0.60 – 0.80    | 说"该学生较符合…模式特征"                     |
| 0.40 – 0.60    | 说"该学生行为特征兼具…和…两种模式" + 传入概率排名第二的模式名 |
| < 0.40         | 建议不突出模式标签，直接基于 6 个维度分数描述           |


> 此处"概率排名第二的模式名"的获取方式：读取 `p_mode_0`~`p_mode_7`，找到第二大值对应的 index，从` mode_definitions.json`查对应`name`。

---

