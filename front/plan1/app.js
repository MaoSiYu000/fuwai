const DATA_BASE = "../data";

const FILES = {
  students: `${DATA_BASE}/student_profiles.csv`,
  groupClass: `${DATA_BASE}/group_profile_by_class.csv`,
  groupMajor: `${DATA_BASE}/group_profile_by_major.csv`,
  groupCollege: `${DATA_BASE}/group_profile_by_college.csv`,
  modeDefs: `${DATA_BASE}/mode_definitions.json`,
  subtypeDefs: `${DATA_BASE}/subtype_definitions.json`,
  formulas: `${DATA_BASE}/dim_score_formulas.json`,
};

const DIM_KEYS = [
  "dim_academic",
  "dim_attendance_engagement",
  "dim_homework_behavior",
  "dim_online_learning",
  "dim_fitness",
  "dim_development",
];
const DIM_LABELS = ["学业", "出勤参与", "作业行为", "线上学习", "体能", "发展成就"];

const MODE_KEYS = Array.from({ length: 8 }, (_, i) => `p_mode_${i}`);

const state = {
  loaded: false,
  studentRows: [],
  studentCols: [],
  group: {
    class: { rows: [], cols: [] },
    major: { rows: [], cols: [] },
    college: { rows: [], cols: [] },
  },
  modeDefs: null,
  subtypeDefs: null,
  formulas: null,
  selectedStudent: null,
};

let chartModeDist = null;
let chartRadar = null;
let chartProb = null;
let chartGroupMode = null;
let chartGroupRadar = null;

function $(sel) {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`Missing element: ${sel}`);
  return el;
}

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function fmtPct(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

function setStatus(text, kind = "info") {
  const el = $("#kpiStatus");
  el.textContent = text;
  el.style.color =
    kind === "ok" ? "var(--good)" : kind === "bad" ? "var(--bad)" : "var(--text)";
}

function parseCSV(text) {
  // 简化 CSV 解析器：支持引号、逗号、换行。足够应对本项目输出。
  const rows = [];
  let row = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cur += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      continue;
    }
    if (ch === ",") {
      row.push(cur);
      cur = "";
      continue;
    }
    if (ch === "\n") {
      row.push(cur);
      cur = "";
      rows.push(row);
      row = [];
      continue;
    }
    if (ch === "\r") continue;
    cur += ch;
  }
  row.push(cur);
  rows.push(row);

  // 去掉尾部空行
  while (rows.length && rows[rows.length - 1].every((v) => v === "")) rows.pop();

  if (!rows.length) return { cols: [], rows: [] };
  const cols = rows[0];
  const out = [];
  for (let r = 1; r < rows.length; r++) {
    const arr = rows[r];
    const obj = {};
    for (let c = 0; c < cols.length; c++) obj[cols[c]] = arr[c] ?? "";
    out.push(obj);
  }
  return { cols, rows: out };
}

function autoNumber(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (s === "" || s.toLowerCase() === "nan") return null;
  const x = Number(s);
  if (Number.isFinite(x)) return x;
  return s;
}

async function fetchText(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} @ ${url}`);
  return await res.text();
}

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} @ ${url}`);
  return await res.json();
}

function destroyChart(ch) {
  if (!ch) return null;
  try {
    ch.destroy();
  } catch {
    // ignore
  }
  return null;
}

function safeModeName(modeId) {
  if (state.modeDefs && state.modeDefs.modes) {
    const m = state.modeDefs.modes.find((x) => Number(x.mode_id) === Number(modeId));
    if (m && m.mode_name) return m.mode_name;
  }
  return `mode ${modeId}`;
}

function buildTable(container, rows, columns, { onRowClick, selectedKey } = {}) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty">暂无数据。</div>`;
    return;
  }
  const cols = columns.slice(0);
  const thead = `<thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>`;
  const tbody = rows
    .map((r) => {
      const key = selectedKey ? r[selectedKey] : null;
      return `<tr data-key="${escapeAttr(key ?? "")}">
        ${cols.map((c) => `<td>${escapeHtml(String(r[c] ?? ""))}</td>`).join("")}
      </tr>`;
    })
    .join("");
  container.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;

  if (onRowClick) {
    const tb = container.querySelector("tbody");
    tb.addEventListener("click", (e) => {
      const tr = e.target.closest("tr");
      if (!tr) return;
      const idx = Array.from(tb.children).indexOf(tr);
      if (idx < 0) return;
      onRowClick(rows[idx], tr);
    });
  }
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(s) {
  return String(s).replaceAll('"', "&quot;");
}

function openModal(title, html) {
  const modal = $("#modal");
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = html;
  modal.hidden = false;
}

function closeModal() {
  $("#modal").hidden = true;
}

function wireModal() {
  $("#modal").addEventListener("click", (e) => {
    const t = e.target;
    if (t && t.getAttribute && t.getAttribute("data-close") === "1") closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

function routeTo(page) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.route === page);
  });
  document.querySelectorAll("[data-page]").forEach((sec) => {
    sec.hidden = sec.dataset.page !== page;
  });
  history.replaceState({}, "", `#${page}`);
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => routeTo(btn.dataset.route));
  });
  const initial = (location.hash || "#overview").slice(1);
  routeTo(["overview", "student", "group", "modes", "llm"].includes(initial) ? initial : "overview");
}

function buildModeCards(modeDist) {
  const wrap = $("#modeCards");
  const entries = Object.entries(modeDist)
    .map(([k, v]) => ({ modeId: Number(k), n: v }))
    .sort((a, b) => b.n - a.n);
  const top = entries.slice(0, 8);
  wrap.innerHTML = `<div class="mode-cards">
    ${top
      .map((x) => {
        const name = safeModeName(x.modeId);
        const pct = state.studentRows.length ? (x.n / state.studentRows.length) : 0;
        const desc = (state.modeDefs?.modes || []).find((m) => Number(m.mode_id) === x.modeId)?.mode_desc || "";
        return `<div class="mode-card" data-mode="${x.modeId}">
          <div class="mode-name">${escapeHtml(name)} <span class="pill" style="margin-left:8px">${fmtPct(pct)}</span></div>
          <div class="mode-desc">${escapeHtml(desc || "点击查看该模式的维度画像与说明。")}</div>
        </div>`;
      })
      .join("")}
  </div>`;

  wrap.querySelectorAll(".mode-card").forEach((card) => {
    card.addEventListener("click", () => {
      const modeId = Number(card.dataset.mode);
      openModeModal(modeId);
    });
  });
}

function openModeModal(modeId) {
  const m = (state.modeDefs?.modes || []).find((x) => Number(x.mode_id) === modeId);
  const name = m?.mode_name || `mode ${modeId}`;
  const dims = m?.dim_profile || m?.dim_profile_cn || null;

  const dimHtml = (() => {
    if (!dims) return `<div class="muted">未找到该 mode 的维度画像字段（仍可正常使用聚类输出）。</div>`;
    if (Array.isArray(dims)) {
      return `<ul>${dims.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`;
    }
    if (typeof dims === "object") {
      const keys = DIM_KEYS.filter((k) => k in dims);
      if (!keys.length) return `<pre class="mono">${escapeHtml(JSON.stringify(dims, null, 2))}</pre>`;
      return `<table>
        <thead><tr><th>维度</th><th>值</th></tr></thead>
        <tbody>
          ${keys
            .map((k) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(dims[k]))}</td></tr>`)
            .join("")}
        </tbody>
      </table>`;
    }
    return `<pre class="mono">${escapeHtml(String(dims))}</pre>`;
  })();

  const html = `
    <div class="detail-head">
      <div>
        <div class="detail-title">${escapeHtml(name)}</div>
        <div class="detail-sub">${escapeHtml(m?.mode_desc || "")}</div>
      </div>
      <div class="badge">mode ${modeId}</div>
    </div>
    <div class="mt"></div>
    <div class="panel">
      <div class="panel-hd"><div class="panel-title">维度画像（来自 mode_definitions.json）</div></div>
      <div class="panel-bd">${dimHtml}</div>
    </div>
  `;
  openModal(`模式详情：${name}`, html);
}

function computeModeDistribution(rows) {
  const dist = {};
  for (const r of rows) {
    const k = String(r.mode_id ?? "");
    if (!k) continue;
    dist[k] = (dist[k] || 0) + 1;
  }
  return dist;
}

function renderOverview() {
  const dist = computeModeDistribution(state.studentRows);
  const labels = Array.from({ length: 8 }, (_, i) => safeModeName(i));
  const data = Array.from({ length: 8 }, (_, i) => dist[String(i)] || 0);

  chartModeDist = destroyChart(chartModeDist);
  const ctx = $("#chartModeDist").getContext("2d");
  chartModeDist = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "样本数",
          data,
          borderWidth: 0,
          backgroundColor: "rgba(255,207,90,.35)",
          hoverBackgroundColor: "rgba(142,240,255,.35)",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx2) => {
              const total = state.studentRows.length || 1;
              const pct = ctx2.raw / total;
              return `占比：${fmtPct(pct)}`;
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: "#e9e7ff" }, grid: { color: "rgba(255,255,255,.06)" } },
        y: { ticks: { color: "#e9e7ff" }, grid: { color: "rgba(255,255,255,.06)" } },
      },
    },
  });

  $("#overviewHint").textContent =
    "提示：如果你直接双击打开 HTML，浏览器可能会拦截 fetch 读取本地 CSV。建议用本地静态服务器方式打开。";

  buildModeCards(dist);
}

function studentQuery(xh, term) {
  const XH = (xh || "").trim();
  const TERM = (term || "").trim();
  if (!XH) return [];
  let rows = state.studentRows.filter((r) => String(r.XH) === XH);
  if (TERM) rows = rows.filter((r) => String(r.TERM_KEY) === TERM);
  rows.sort((a, b) => String(a.TERM_KEY).localeCompare(String(b.TERM_KEY)));
  return rows;
}

function renderStudentTable(rows) {
  $("#studentCount").textContent = `${rows.length} 条`;
  const wrap = $("#studentTableWrap");
  const cols = ["XH", "TERM_KEY", "mode_id", "mode_name", "mode_pmax", "mode_entropy", "subtype_id", "subtype_name"];
  const safeRows = rows.map((r, idx) => ({ ...r, _idx: String(idx) }));
  buildTable(wrap, safeRows, cols.filter((c) => c in (safeRows[0] || {})), {
    onRowClick: (row, tr) => {
      wrap.querySelectorAll("tr").forEach((x) => x.classList.remove("row-selected"));
      tr.classList.add("row-selected");
      showStudentDetail(row);
    },
  });
}

function showStudentDetail(row) {
  state.selectedStudent = row;
  $("#studentDetailEmpty").hidden = true;
  $("#studentDetail").hidden = false;
  $("#btnOpenLLMFromStudent").disabled = false;
  $("#btnLLMTest").disabled = false;

  const xh = row.XH;
  const term = row.TERM_KEY;
  const modeId = row.mode_id;
  const modeName = row.mode_name || safeModeName(modeId);
  const subtype = row.subtype_name ? ` · ${row.subtype_name}` : "";

  $("#dTitle").textContent = `${xh} / ${term}`;
  $("#dSub").textContent = `${modeName}${subtype}`;
  $("#dBadge").textContent = `mode ${modeId}`;
  $("#dPmax").textContent = fmt(autoNumber(row.mode_pmax), 3);
  $("#dMargin").textContent = fmt(autoNumber(row.mode_margin), 3);
  $("#dEntropy").textContent = fmt(autoNumber(row.mode_entropy), 3);
  $("#dEvidence").textContent = row.mode_evidence || "—";

  const dimVals = DIM_KEYS.map((k) => autoNumber(row[k]) ?? 0);
  chartRadar = destroyChart(chartRadar);
  chartRadar = new Chart($("#chartRadar").getContext("2d"), {
    type: "radar",
    data: {
      labels: DIM_LABELS,
      datasets: [
        {
          label: "维度分数（相对全体均值=0）",
          data: dimVals,
          borderColor: "rgba(142,240,255,.70)",
          backgroundColor: "rgba(142,240,255,.12)",
          pointBackgroundColor: "rgba(255,207,90,.85)",
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          ticks: { color: "#a39ecb" },
          grid: { color: "rgba(255,255,255,.08)" },
          angleLines: { color: "rgba(255,255,255,.08)" },
          pointLabels: { color: "#e9e7ff", font: { size: 12, weight: "700" } },
        },
      },
    },
  });

  const probs = MODE_KEYS.map((k) => autoNumber(row[k]) ?? 0);
  chartProb = destroyChart(chartProb);
  chartProb = new Chart($("#chartProb").getContext("2d"), {
    type: "bar",
    data: {
      labels: Array.from({ length: 8 }, (_, i) => `mode ${i}`),
      datasets: [
        {
          label: "概率",
          data: probs,
          borderWidth: 0,
          backgroundColor: "rgba(255,207,90,.35)",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#e9e7ff" }, grid: { color: "rgba(255,255,255,.06)" } },
        y: {
          ticks: { color: "#e9e7ff" },
          grid: { color: "rgba(255,255,255,.06)" },
          min: 0,
          max: 1,
        },
      },
    },
  });

  // 原始指标表：把非概率字段拆出来，避免太密
  const omit = new Set(["_idx", ...MODE_KEYS]);
  const keep = Object.keys(row)
    .filter((k) => !omit.has(k))
    .filter((k) => !k.startsWith("dim_"))
    .filter((k) => !["mode_evidence"].includes(k));

  const pairs = keep
    .map((k) => ({
      指标: k,
      值: row[k],
    }))
    .slice(0, 60); // 控制密度：多余的仍可通过 JSON modal 查看

  buildTable($("#dRawTable"), pairs, ["指标", "值"]);

  // LLM payload preview
  $("#llmPayloadPreview").textContent = JSON.stringify(buildLLMPayload(row), null, 2);
}

function getGroupState(level) {
  const g = state.group[level];
  if (!g || !g.rows) return { rows: [], cols: [] };
  return g;
}

function groupFilter(level, query, term) {
  const g = getGroupState(level);
  const q = (query || "").trim();
  const t = (term || "").trim();
  let rows = g.rows;
  if (t) rows = rows.filter((r) => String(r.TERM_KEY || "") === t);
  if (q) {
    const q2 = q.toLowerCase();
    rows = rows.filter((r) => JSON.stringify(r).toLowerCase().includes(q2));
  }
  return rows.slice(0, 400); // 控制渲染压力
}

function detectGroupNameKey(level, row) {
  const candidates =
    level === "class"
      ? ["CLASS_NAME", "class_name", "班级", "group", "name"]
      : level === "major"
      ? ["ZYM", "major", "专业", "group", "name"]
      : ["XSM", "college", "学院", "group", "name"];
  for (const k of candidates) if (k in row) return k;
  // fallback: 找一个字符串列
  const ks = Object.keys(row);
  for (const k of ks) if (typeof autoNumber(row[k]) === "string") return k;
  return ks[0];
}

function renderGroupTable(level, rows) {
  $("#groupCount").textContent = `${rows.length} 条（最多展示 400）`;
  const wrap = $("#groupTableWrap");
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty">暂无匹配群体。</div>`;
    return;
  }

  const nameKey = detectGroupNameKey(level, rows[0]);
  const cols = [nameKey, "TERM_KEY", "n_records"].filter((c) => c in rows[0]);
  // 追加几个模式占比列（如果存在）
  for (let i = 0; i < 8; i++) {
    const k = `cluster_${i}_pct`;
    if (k in rows[0]) cols.push(k);
  }

  buildTable(wrap, rows, cols, {
    onRowClick: (row, tr) => {
      wrap.querySelectorAll("tr").forEach((x) => x.classList.remove("row-selected"));
      tr.classList.add("row-selected");
      showGroupDetail(level, row);
    },
  });
}

function showGroupDetail(level, row) {
  $("#groupDetailEmpty").hidden = true;
  $("#groupDetail").hidden = false;

  const nameKey = detectGroupNameKey(level, row);
  const title = row[nameKey] || "—";
  const term = row.TERM_KEY || "（全部学期）";
  $("#gTitle").textContent = String(title);
  $("#gSub").textContent = `学期：${term}`;
  $("#gBadge").textContent = level === "class" ? "班级" : level === "major" ? "专业" : "学院";

  const modePct = Array.from({ length: 8 }, (_, i) => autoNumber(row[`cluster_${i}_pct`]) ?? 0);
  chartGroupMode = destroyChart(chartGroupMode);
  chartGroupMode = new Chart($("#chartGroupMode").getContext("2d"), {
    type: "bar",
    data: {
      labels: Array.from({ length: 8 }, (_, i) => `mode ${i}`),
      datasets: [
        {
          label: "占比",
          data: modePct,
          borderWidth: 0,
          backgroundColor: "rgba(255,207,90,.35)",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#e9e7ff" }, grid: { color: "rgba(255,255,255,.06)" } },
        y: { ticks: { color: "#e9e7ff", callback: (v) => `${v}` }, grid: { color: "rgba(255,255,255,.06)" }, min: 0, max: 1 },
      },
    },
  });

  const dimMeans = DIM_KEYS.map((k) => autoNumber(row[k]) ?? 0);
  chartGroupRadar = destroyChart(chartGroupRadar);
  chartGroupRadar = new Chart($("#chartGroupRadar").getContext("2d"), {
    type: "radar",
    data: {
      labels: DIM_LABELS,
      datasets: [
        {
          label: "维度均值",
          data: dimMeans,
          borderColor: "rgba(142,240,255,.70)",
          backgroundColor: "rgba(142,240,255,.12)",
          pointBackgroundColor: "rgba(255,207,90,.85)",
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          ticks: { color: "#a39ecb" },
          grid: { color: "rgba(255,255,255,.08)" },
          angleLines: { color: "rgba(255,255,255,.08)" },
          pointLabels: { color: "#e9e7ff", font: { size: 12, weight: "700" } },
        },
      },
    },
  });

  const omit = new Set([]);
  const keys = Object.keys(row).filter((k) => !omit.has(k));
  const pairs = keys.map((k) => ({ 指标: k, 值: row[k] }));
  buildTable($("#gRawTable"), pairs, ["指标", "值"]);
}

function renderModeDefs() {
  const wrap = $("#modeDefWrap");
  if (!state.modeDefs) {
    wrap.innerHTML = `<div class="empty">未加载 mode_definitions.json（不影响学生/群体页使用）。</div>`;
    return;
  }
  const modes = state.modeDefs.modes || state.modeDefs || [];
  wrap.innerHTML = `<div class="muted">点击某个 mode 卡片，也可以从“总览”进入。</div>
  <div class="mode-cards mt">
    ${(modes || [])
      .map((m) => {
        const id = m.mode_id ?? m.id ?? "";
        const name = m.mode_name ?? `mode ${id}`;
        const desc = m.mode_desc ?? "";
        return `<div class="mode-card" data-mode="${escapeAttr(String(id))}">
          <div class="mode-name">${escapeHtml(String(name))}</div>
          <div class="mode-desc">${escapeHtml(String(desc))}</div>
        </div>`;
      })
      .join("")}
  </div>`;

  wrap.querySelectorAll(".mode-card").forEach((card) => {
    card.addEventListener("click", () => openModeModal(Number(card.dataset.mode)));
  });
}

function renderFormulas() {
  const wrap = $("#formulaWrap");
  if (!state.formulas) {
    wrap.innerHTML = `<div class="empty">未加载 dim_score_formulas.json。</div>`;
    return;
  }
  wrap.innerHTML = `<pre class="mono">${escapeHtml(JSON.stringify(state.formulas, null, 2))}</pre>`;
}

function renderSubtypeDefs() {
  const wrap = $("#subtypeDefWrap");
  if (!state.subtypeDefs) {
    wrap.innerHTML = `<div class="empty">未加载 subtype_definitions.json。</div>`;
    return;
  }
  wrap.innerHTML = `<pre class="mono">${escapeHtml(JSON.stringify(state.subtypeDefs, null, 2))}</pre>`;
}

function buildLLMPayload(studentRow) {
  // 统一 payload：后续你接后端时，按这个结构做 prompt 即可
  const probs = {};
  for (const k of MODE_KEYS) probs[k] = autoNumber(studentRow[k]);

  const dims = {};
  for (const k of DIM_KEYS) dims[k] = autoNumber(studentRow[k]);

  const core = {
    XH: studentRow.XH,
    TERM_KEY: studentRow.TERM_KEY,
    mode_id: autoNumber(studentRow.mode_id),
    mode_name: studentRow.mode_name,
    subtype_id: autoNumber(studentRow.subtype_id),
    subtype_name: studentRow.subtype_name,
    mode_pmax: autoNumber(studentRow.mode_pmax),
    mode_margin: autoNumber(studentRow.mode_margin),
    mode_entropy: autoNumber(studentRow.mode_entropy),
    mode_evidence: studentRow.mode_evidence,
    dim_scores: dims,
    mode_probs: probs,
  };

  // 附加：一些关键指标（如果存在）
  const extraKeys = [
    "kccj_mean",
    "kccj_fail_rate",
    "jdcj_mean",
    "att_present_rate",
    "hw_submit_cnt",
    "hw_ungraded_rate",
    "online_bfb",
    "sch_amt_sum_term",
    "comp_cnt_term",
    "cet_score_max",
    "fit3_zf_mean",
  ];
  const extras = {};
  for (const k of extraKeys) if (k in studentRow) extras[k] = autoNumber(studentRow[k]);

  return {
    schema_version: 1,
    task: "explain_student_profile",
    input: { student_term_profile: core, extra_metrics: extras },
    context: {
      mode_definitions: state.modeDefs,
      subtype_definitions: state.subtypeDefs,
      dim_score_formulas: state.formulas,
    },
  };
}

async function callLLM(payload) {
  // 预留接口：你把 endpoint 指向自己的后端（推荐），后端再去调用大模型
  const endpoint = $("#llmEndpoint").value.trim();
  if (!endpoint) throw new Error("请先在“LLM接口”页面填写 Endpoint。");
  const key = $("#llmKey").value.trim();

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(key ? { Authorization: `Bearer ${key}` } : {}),
    },
    body: JSON.stringify(payload),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`LLM接口错误：${res.status} ${res.statusText}\n${text}`);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function wireStudentPage() {
  $("#btnFindStudent").addEventListener("click", () => {
    const rows = studentQuery($("#inpXH").value, $("#inpTerm").value);
    renderStudentTable(rows);
  });
  $("#inpXH").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("#btnFindStudent").click();
  });
  $("#inpTerm").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("#btnFindStudent").click();
  });

  $("#btnOpenLLMFromStudent").addEventListener("click", () => {
    routeTo("llm");
    $("#llmPayloadPreview").textContent = state.selectedStudent
      ? JSON.stringify(buildLLMPayload(state.selectedStudent), null, 2)
      : "—";
  });
}

function wireGroupPage() {
  $("#btnFindGroup").addEventListener("click", () => {
    const level = $("#selGroupLevel").value;
    const rows = groupFilter(level, $("#inpGroupQuery").value, $("#inpGroupTerm").value);
    renderGroupTable(level, rows);
  });
}

function wireLLMPage() {
  $("#btnLLMTest").addEventListener("click", async () => {
    $("#llmOutput").textContent = "请求中…";
    try {
      if (!state.selectedStudent) throw new Error("请先在“学生页”选中一条记录。");
      const payload = buildLLMPayload(state.selectedStudent);
      $("#llmPayloadPreview").textContent = JSON.stringify(payload, null, 2);
      const out = await callLLM(payload);
      $("#llmOutput").textContent =
        typeof out === "string" ? out : JSON.stringify(out, null, 2);
    } catch (e) {
      $("#llmOutput").textContent = String(e?.message || e);
    }
  });
}

async function loadAll() {
  setStatus("加载中…");
  $("#kpiRows").textContent = "—";
  $("#kpiSub").textContent = "—";
  state.loaded = false;
  state.selectedStudent = null;
  $("#btnOpenLLMFromStudent").disabled = true;
  $("#btnLLMTest").disabled = true;

  const errors = [];
  try {
    const text = await fetchText(FILES.students);
    const parsed = parseCSV(text);
    state.studentCols = parsed.cols;
    state.studentRows = parsed.rows.map((r) => {
      const out = { ...r };
      for (const k of Object.keys(out)) out[k] = autoNumber(out[k]);
      return out;
    });
  } catch (e) {
    errors.push(`student_profiles.csv: ${String(e?.message || e)}`);
  }

  // 群体画像（可缺省）
  const groupLoads = [
    ["class", FILES.groupClass],
    ["major", FILES.groupMajor],
    ["college", FILES.groupCollege],
  ];
  for (const [level, path] of groupLoads) {
    try {
      const text = await fetchText(path);
      const parsed = parseCSV(text);
      state.group[level].cols = parsed.cols;
      state.group[level].rows = parsed.rows.map((r) => {
        const out = { ...r };
        for (const k of Object.keys(out)) out[k] = autoNumber(out[k]);
        return out;
      });
    } catch (e) {
      // 群体画像不是硬依赖
    }
  }

  // JSON（可缺省）
  try {
    state.modeDefs = await fetchJSON(FILES.modeDefs);
  } catch (e) {
    state.modeDefs = null;
  }
  try {
    state.subtypeDefs = await fetchJSON(FILES.subtypeDefs);
  } catch (e) {
    state.subtypeDefs = null;
  }
  try {
    state.formulas = await fetchJSON(FILES.formulas);
  } catch (e) {
    state.formulas = null;
  }

  if (!state.studentRows.length) {
    setStatus("加载失败", "bad");
    $("#overviewHint").textContent = errors.length
      ? `缺少核心数据文件：\n- ${errors.join("\n- ")}\n\n请确认这些文件位于 front/data/ 下，并用静态服务器方式打开页面。`
      : "缺少核心数据文件 student_profiles.csv。";
    return;
  }

  state.loaded = true;
  setStatus("已加载", "ok");
  $("#kpiRows").textContent = String(state.studentRows.length);
  const subtypeSet = new Set(state.studentRows.map((r) => String(r.subtype_id ?? "")).filter((x) => x));
  $("#kpiSub").textContent = String(subtypeSet.size || 32);

  renderOverview();
  renderModeDefs();
  renderSubtypeDefs();
  renderFormulas();
}

function wireReload() {
  $("#btnReload").addEventListener("click", loadAll);
}

function main() {
  wireModal();
  wireTabs();
  wireReload();
  wireStudentPage();
  wireGroupPage();
  wireLLMPage();
  loadAll();
}

main();

