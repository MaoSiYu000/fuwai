/* ===== AppData: shared data store ===== */
window.AppData = {
  college: [], major: [], clazz: [],
  modes: [], subtypes: [], dims: [],
  loaded: false
};

/* ===== Mode meta ===== */
window.MODE_COLORS = ['#F59E0B','#EF4444','#10B981','#EAB308','#6366F1','#8B5CF6','#F97316','#94A3B8'];
window.MODE_VARS   = ['--c0','--c1','--c2','--c3','--c4','--c5','--c6','--c7'];
window.DIM_KEYS  = ['dim_academic','dim_attendance_engagement','dim_homework_behavior','dim_online_learning','dim_fitness','dim_development'];
window.DIM_NAMES = ['学业','出勤参与','作业行为','线上学习','体能','发展成就'];

/* ===== Utility ===== */
window.fmt = {
  pct: v => (v == null || isNaN(v)) ? '—' : (v*100).toFixed(1)+'%',
  num: (v,d=1) => (v == null || isNaN(v)) ? '—' : Number(v).toFixed(d),
  dim: v => (v == null || isNaN(v)) ? '—' : (v>=0?'+':'')+Number(v).toFixed(2),
};

window.riskLevel = row => {
  const p1 = parseFloat(row.mode_1_pct)||0;
  const p6 = parseFloat(row.mode_6_pct)||0;
  const r  = p1 + p6;
  if (r > .30) return { lv:'高风险', cls:'badge-danger' };
  if (r > .15) return { lv:'需关注', cls:'badge-warn' };
  return { lv:'良好', cls:'badge-good' };
};

window.modeRiskPct = row => {
  const p1 = parseFloat(row.mode_1_pct)||0;
  const p6 = parseFloat(row.mode_6_pct)||0;
  return p1 + p6;
};

/* ===== CSV loader (PapaParse) ===== */
function loadCSV(path) {
  return new Promise((res, rej) => {
    Papa.parse(path, {
      download: true, header: true, skipEmptyLines: true,
      complete: r => res(r.data),
      error: e => rej(e)
    });
  });
}

/* ===== Bootstrap ===== */
window.initAppData = async function() {
  if (AppData.loaded) return;
  const base = getBasePath();
  try {
    const [college, major, clazz, modes, subtypes, dims] = await Promise.all([
      loadCSV(base + 'data/group_profile_by_college.csv'),
      loadCSV(base + 'data/group_profile_by_major.csv'),
      loadCSV(base + 'data/group_profile_by_class.csv'),
      fetch(base + 'data/mode_definitions.json').then(r=>r.json()),
      fetch(base + 'data/subtype_definitions.json').then(r=>r.json()),
      fetch(base + 'data/dim_score_formulas.json').then(r=>r.json()),
    ]);
    Object.assign(AppData, { college, major, clazz, modes, subtypes, dims, loaded: true });
    document.dispatchEvent(new Event('appReady'));
  } catch(e) {
    console.error(e);
    document.getElementById('app-error')?.removeAttribute('hidden');
  }
};

/* ===== Figure out where data/ is relative to current page ===== */
function getBasePath() {
  const p = location.pathname;
  if (p.includes('/pages/')) return '../';
  return './';
}

/* ===== Nav active state ===== */
window.setNavActive = function() {
  document.querySelectorAll('.nav-link').forEach(a => {
    const href = a.getAttribute('href');
    if (href && location.pathname.endsWith(href.replace(/^.*\//,''))) {
      a.classList.add('active');
    }
  });
};

/* ===== Light UI cleanups ===== */
window.cleanNavText = function() {
  // 去掉导航里类似 "📊 总览" 的 emoji（尽量不改结构，只清理显示）
  document.querySelectorAll('.nav-brand, .nav-link').forEach(el => {
    el.textContent = (el.textContent || '')
      .replace(/^[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]+\s*/u, '')
      .replace(/\s+/g, ' ')
      .trim();
  });
};

/* ===== AI text normalization (remove markdown artifacts) ===== */
window.normalizeAIText = function(text) {
  const s = String(text ?? '');
  // 1) 统一换行
  let out = s.replace(/\r\n/g, '\n');
  // 2) 去掉常见 markdown 强调符号（避免出现 ** ）
  out = out.replace(/\*\*/g, '');
  // 有些模型会输出单个 * 作为强调，尽量去掉（保留像 3*5 这种）
  out = out.replace(/(^|[^\d])\*(?!\d)(.*?)\*(?!\d)/g, '$1$2');
  // 3) 去掉整行分割线（--- / ***）
  out = out.replace(/^\s*[-*_]{3,}\s*$/gm, '');
  // 4) 把行内分割线也转成段落
  out = out.replace(/\s*---+\s*/g, '\n\n');

  // 5) 把常见“段落标记/列表”补成换行（模型经常只用空格不换行）
  // 标题/小节：xxx： 后面常跟正文，给它断段
  out = out.replace(/(总览解读|趋势解读|风险总览解读|模式分析|当前选中模式解释|当前选中模式解读|当前模式解释|可执行建议|建议)\s*：/g, '\n\n$1：\n');
  // 兼容 “一、二、三、四” 这种
  out = out.replace(/([一二三四五六七八九十]+、)/g, '\n\n$1');
  // 1. 2. 3. 这种编号列表
  out = out.replace(/\s(?=\d+\.\s)/g, '\n');
  // emoji/符号提示点常用于分段
  out = out.replace(/\s(?=[👉✅📌➡️]+)/g, '\n');
  // 句尾后面跟新段落关键词时也断一下
  out = out.replace(/([。！？])\s+(?=[一二三四五六七八九十]+、|👉|✅|📌|\d+\.\s)/g, '$1\n');

  // 4) 清理多余空行
  out = out.replace(/\n{3,}/g, '\n\n').trim();
  return out;
};

/* ===== Embedded AI (optional) ===== */
window.AIEmbed = {
  async callOpenAICompatible({ url, key, model, messages, stream, onDelta }) {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(key ? { 'Authorization': `Bearer ${key}` } : {})
      },
      body: JSON.stringify({
        model,
        messages,
        stream: !!stream,
        temperature: 0.7,
        max_tokens: 1200
      })
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`请求失败 (${res.status}): ${err.slice(0, 300)}`);
    }
    if (!stream) {
      const j = await res.json();
      const text = j?.choices?.[0]?.message?.content || '';
      onDelta?.(text);
      return text;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(l => l.startsWith('data: '));
      for (const line of lines) {
        const data = line.slice(6);
        if (data === '[DONE]') break;
        try {
          const obj = JSON.parse(data);
          const delta = obj?.choices?.[0]?.delta?.content || '';
          if (delta) {
            full += delta;
            onDelta?.(delta, full);
          }
        } catch {
          // ignore partial lines
        }
      }
    }
    return full;
  }
};

/* ===== Modal helpers ===== */
window.openModal = id => { document.getElementById(id)?.classList.add('open'); };
window.closeModal = id => { document.getElementById(id)?.classList.remove('open'); };
window.initModals = function() {
  document.querySelectorAll('.overlay').forEach(el => {
    el.addEventListener('click', e => { if (e.target===el) el.classList.remove('open'); });
  });
  document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('.overlay')?.classList.remove('open'));
  });
};

/* ===== Dim bar renderer ===== */
window.renderDimBars = function(container, dimVals, maxAbs) {
  const max = maxAbs || 1.5;
  const html = DIM_KEYS.map((k,i) => {
    const v = parseFloat(dimVals[k]) || 0;
    const pct = Math.min(Math.abs(v)/max*50, 50);
    const pos = v >= 0;
    const color = pos ? '#10B981' : '#EF4444';
    const left = pos ? '50%' : `${50-pct}%`;
    return `<div class="dim-row">
      <div class="dim-lbl">${DIM_NAMES[i]}</div>
      <div class="dim-track">
        <div class="dim-axis"></div>
        <div class="dim-fill" style="left:${left};width:${pct}%;background:${color}"></div>
      </div>
      <div class="dim-num" style="color:${color}">${fmt.dim(v)}</div>
    </div>`;
  }).join('');
  container.innerHTML = `<div class="dim-rows">${html}</div>`;
};

/* ===== ECharts theme defaults ===== */
window.chartOpts = {
  tooltip: { confine: true, textStyle: { fontSize: 12 } },
  grid: { left:14, right:14, top:32, bottom:14, containLabel:true }
};

document.addEventListener('DOMContentLoaded', () => {
  setNavActive();
  initModals();
  cleanNavText();
  initAppData();
});
