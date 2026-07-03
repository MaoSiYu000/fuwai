import os
import json
import csv
import re
import time
import secrets
from typing import Any, Dict, Optional, List, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# 自动读取本机 .env（已在 .gitignore 忽略，不会进仓库）
load_dotenv()

app = FastAPI(title="fuwai api", version="0.1.0")

# 允许前端本地/同源调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSION_COOKIE = "guanlan_session"
# 轻量本地会话（只为前端联调/演示；非生产级）
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_USERS: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (role, username) -> {password, must_change_password, profile}


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else str(v)


def _parse_csv_env(name: str) -> List[str]:
    raw = _env(name, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _infer_intent(message: str) -> str:
    """
    返回：explain / analyze / suggest
    主要依据前端三按钮在 message 里写的“页面解读/页面分析/建议”等字样。
    """
    m = (message or "").strip()
    if any(k in m for k in ("页面解读", "页面讲解")):
        return "explain"
    if "页面分析" in m:
        return "analyze"
    if "建议" in m:
        return "suggest"
    return "analyze"


def _system_prompt(intent: str) -> str:
    base = (
        "你是高校学生行为画像分析助手。"
        "输出必须是【纯中文纯文本】。"
        "禁止使用任何Markdown/格式符号：不要输出###、**、*、---、>、代码块、表格。"
        "不要复述“问题/上下文/指标名(括号字段名)”等提示词。"
        "分段用换行。"
        "重要约束：只能基于用户提供的上下文数据（context/view/selected_mode等JSON）作答。"
        "禁止引用外部/网上数据，禁止编造数值，禁止做无依据的猜测。"
        "不要输出“证据不足/无法判断”等措辞。"
        "只回答你能从项目数据中直接支持的部分；缺失的信息直接跳过不说。"
        "允许做非常克制的推测，但必须显式标注为“推测”，且只能基于已提供数据的关联做推测，不能当成结论。"
        "重要展示规则：禁止在最终回答中出现任何代码内部字段名/变量名（例如 dim_academic、kccj_mean_avg、p_mode_7 等）。"
        "如果需要引用指标，请使用中文名称（例如 学业维度、成绩均值、模式7概率）。"
    )

    if intent == "explain":
        return (
            base
            + "当前意图=页面讲解：只讲这个页面各模块/图表怎么读、点哪里会发生什么、常见误读。"
            + "不要输出建议，不需要列证据。"
            + "允许用“例如：”解释坐标轴/图例含义，但不得虚构具体数值。"
        )

    if intent == "suggest":
        return (
            base
            + "当前意图=建议：输出必须包含“理由”和“建议”两部分（按用户要求的条数）。"
            + "理由必须内嵌引用项目数据，写成“根据<字段/指标>=<数值/现象>，因此……”。"
            + "不要单独列‘证据’小节。"
            + "如果关键前提数据缺失：不要硬编，最多给1条标注为“推测”的可能解释（可省略），再给建议。"
        )

    # analyze
    return (
        base
        + "当前意图=页面分析：给结论与依据。"
        + "每条关键判断都必须内嵌引用项目数据，写成“根据<字段/指标>=<数值/现象>，因此……”。"
        + "不要单独列‘证据’小节。"
        + "不要输出建议（除非用户明确要求）。"
        + "如果数据缺失：不要说无法判断，直接只写你能用数据支撑的部分；可选给出最多1条标注为“推测”的可能解释。"
    )


_DATA_CACHE: Dict[str, Any] = {"loaded": False}


def _front_dir() -> str:
    """
    需要托管/读取数据的前端目录。
    - 默认：<repo>/front
    - 可通过环境变量 FRONT_DIR 覆盖（支持相对路径/绝对路径）
    """
    raw = _env("FRONT_DIR", "front").strip()
    if os.path.isabs(raw):
        return raw
    return os.path.join(os.path.dirname(__file__), raw)


def _data_dir() -> str:
    return os.path.join(_front_dir(), "data")


def _now() -> int:
    return int(time.time())


def _new_sid() -> str:
    return secrets.token_urlsafe(24)


def _get_sid(req: Request) -> Optional[str]:
    return req.cookies.get(_SESSION_COOKIE)


def _get_session(req: Request) -> Optional[Dict[str, Any]]:
    sid = _get_sid(req)
    if not sid:
        return None
    return _SESSIONS.get(sid)


def _set_session(resp: JSONResponse, session: Dict[str, Any]) -> None:
    sid = session.get("sid") or _new_sid()
    session["sid"] = sid
    session["ts"] = _now()
    _SESSIONS[sid] = session
    # 简单 cookie（演示用途）
    resp.set_cookie(
        key=_SESSION_COOKIE,
        value=sid,
        httponly=False,
        samesite="lax",
        secure=False,
        path="/",
    )


def _clear_session(resp: JSONResponse, req: Request) -> None:
    sid = _get_sid(req)
    if sid and sid in _SESSIONS:
        _SESSIONS.pop(sid, None)
    resp.delete_cookie(_SESSION_COOKIE, path="/")


def _public_session(sess: Dict[str, Any]) -> Dict[str, Any]:
    # 前端 auth.js 需要这些字段
    return {
        "role": sess.get("role"),
        "username": sess.get("username"),
        "displayName": sess.get("displayName") or sess.get("username"),
        "mustChangePassword": bool(sess.get("mustChangePassword")),
    }


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _ensure_data_loaded() -> None:
    # 若切换了 FRONT_DIR，需要重新加载缓存
    cur_front = _front_dir()
    if _DATA_CACHE.get("loaded") and _DATA_CACHE.get("front_dir") == cur_front:
        return
    _DATA_CACHE.clear()
    _DATA_CACHE["loaded"] = False
    _DATA_CACHE["front_dir"] = cur_front
    d = _data_dir()
    _DATA_CACHE["mode_defs"] = _load_json(os.path.join(d, "mode_definitions.json"))
    _DATA_CACHE["subtype_defs"] = _load_json(os.path.join(d, "subtype_definitions.json"))
    _DATA_CACHE["group_college"] = _load_csv(os.path.join(d, "group_profile_by_college.csv"))
    _DATA_CACHE["group_major"] = _load_csv(os.path.join(d, "group_profile_by_major.csv"))
    _DATA_CACHE["group_class"] = _load_csv(os.path.join(d, "group_profile_by_class.csv"))
    _DATA_CACHE["student_profiles_path"] = os.path.join(d, "student_profiles.csv")
    _DATA_CACHE["loaded"] = True


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def _group_rows(lv: str) -> List[Dict[str, str]]:
    if lv == "college":
        return _DATA_CACHE["group_college"]
    if lv == "major":
        return _DATA_CACHE["group_major"]
    return _DATA_CACHE["group_class"]


def _group_key(lv: str) -> str:
    return "XSM" if lv == "college" else ("ZYM" if lv == "major" else "CLASS_NAME")


def _aggregate_group(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {}
    num_keys = [k for k in rows[0].keys() if k not in ("XSM", "ZYM", "CLASS_NAME", "TERM_KEY", "dominant_mode_id", "dominant_mode_name")]
    out: Dict[str, Any] = {}
    for k in num_keys:
        vals = [_to_float(r.get(k)) for r in rows]
        vals2 = [v for v in vals if v is not None]
        out[k] = (sum(vals2) / len(vals2)) if vals2 else None
    # n_records 走求和更合理
    n = 0
    for r in rows:
        v = _to_float(r.get("n_records"))
        if v is not None:
            n += int(v)
    out["n_records"] = n or None
    return out


def _enrich_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    从项目数据文件中补全对比信息，让 AI 不局限于当前页面传来的摘要。
    只做轻量聚合，避免把整库塞进上下文。
    """
    _ensure_data_loaded()
    ctx = payload.get("context") or payload.get("view") or {}
    if not isinstance(ctx, dict):
        return {"raw_context": ctx}

    page = str(ctx.get("page") or payload.get("task") or "")
    enriched: Dict[str, Any] = dict(ctx)

    # 总览/风险/趋势：群体聚合表里就能拿到“总体 vs 群体(某学期)”的对比证据
    # - 总体：mode_definitions.json 的 pct + dim_profile
    enriched["_db_overall_modes"] = _DATA_CACHE.get("mode_defs")

    # 群体口径（如果 context 里带了 lv/group/term）
    lv = ""
    group = ""
    term = ""
    if isinstance(ctx.get("group"), dict):
        g = ctx.get("group") or {}
        lv = str(g.get("lv") or "")
        group = str(g.get("name") or "")
        term = str(g.get("term") or "")
    else:
        lv = str(ctx.get("lv") or ctx.get("level") or "")
        group = str(ctx.get("group") or ctx.get("group_name") or "")
        term = str(ctx.get("term") or ctx.get("TERM_KEY") or "")

    if lv in ("college", "major", "class") and group:
        rows = _group_rows(lv)
        key = _group_key(lv)
        filt = [r for r in rows if str(r.get(key) or "") == group and (not term or str(r.get("TERM_KEY") or "") == term)]
        agg = _aggregate_group(filt)
        enriched["_db_group_agg"] = {"lv": lv, "group": group, "term": term or "", "agg": agg}

    return enriched


_FIELD_NAME_MAP: Dict[str, str] = {
    # six dims
    "dim_academic": "学业维度",
    "dim_attendance_engagement": "出勤参与维度",
    "dim_homework_behavior": "作业行为维度",
    "dim_online_learning": "线上学习维度",
    "dim_fitness": "体能维度",
    "dim_development": "发展成就维度",
    # group metrics (common)
    "kccj_mean_avg": "课程成绩均值",
    "kccj_fail_rate_avg": "挂科率",
    "att_present_rate_avg": "出勤率",
    "online_bfb_avg": "线上完成率",
    # misc
    "risk_pct": "高风险比例",
}


def _sanitize_output(text: str) -> str:
    """
    防止模型把内部字段名直接暴露给用户。
    将常见字段名替换为中文指标名，并去掉类似 “（dim_xxx=0.123）” 的写法。
    """
    out = str(text or "")
    # 1) 替换已知字段名 -> 中文
    for k, v in _FIELD_NAME_MAP.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out)
    # 2) 把 p_mode_7 这种替换成 “模式7概率”
    out = re.sub(r"\bp_mode_(\d+)\b", r"模式\1概率", out)
    out = re.sub(r"\bmode_(\d+)_pct\b", r"模式\1占比", out)
    # 3) 兜底：仍残留 dim_xxx 的直接去掉字段名，只保留语义提示
    out = re.sub(r"\bdim_[a-z_]+\b", "维度分", out)
    return out


def _proxy_openai_compatible(payload: Dict[str, Any]) -> str:
    """
    可选：代理到 OpenAI-Compatible 接口（你们后续接大模型代理时用）。
    通过环境变量启用：
      OPENAI_BASE_URL 例如 http://localhost:8001/v1
      OPENAI_API_KEY  可空（若代理不需要）
      OPENAI_MODEL    单模型（兼容旧配置）
      OPENAI_MODELS   多模型，逗号分隔，按顺序自动切换
    """
    # 百炼（DashScope）OpenAI-Compatible 默认基址（北京）
    # 文档：POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    base = _env("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    url = base + "/chat/completions"
    key = _env("OPENAI_API_KEY")
    models = _parse_csv_env("OPENAI_MODELS")
    if not models:
        models = [_env("OPENAI_MODEL", "gpt-4o-mini")]

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        # 兼容当前前端：task/context/message
        task = payload.get("task", "unknown")
        ctx0 = payload.get("context") or payload.get("view") or payload
        # 后端基于项目数据补全证据（不依赖前端是否传了对比数据）
        ctx = _enrich_context({"task": task, "context": ctx0})
        msg = payload.get("message", "")
        intent = _infer_intent(str(msg))
        messages = [
            {
                "role": "system",
                "content": _system_prompt(intent),
            },
            {"role": "user", "content": f"任务：{task}\n上下文：{json.dumps(ctx, ensure_ascii=False)}\n\n问题：{msg}"},
        ]

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_err: Optional[Exception] = None
    for model in models:
        req_body = {
            "model": model,
            "messages": messages,
            # 降低随机性，减少“发散/想象”的概率
            "temperature": 0.2,
            "max_tokens": 1200,
            "stream": False,
        }
        try:
            r = requests.post(url, headers=headers, json=req_body, timeout=90)
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code} {r.text[:400]}")
            j = r.json()
            text = (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            if text:
                return _sanitize_output(text)
            last_err = RuntimeError(f"model {model}: empty content")
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"all models failed; last_error={last_err}")


def _fallback_answer(payload: Dict[str, Any]) -> str:
    # 没接大模型时的兜底，确保“开始分析”不会报错，便于前端联调
    task = payload.get("task", "ai")
    msg = str(payload.get("message") or "").strip()
    ctx = payload.get("context") or payload.get("view") or payload.get("selected_mode") or {}
    return (
        "（占位回答：后端已联通，但尚未接入真实大模型）\n"
        f"- task: {task}\n"
        f"- message: {msg[:300]}\n"
        f"- context_keys: {', '.join(list(ctx.keys())[:30]) if isinstance(ctx, dict) else '—'}\n\n"
        "接入方式：设置 OPENAI_API_KEY（百炼 sk-...），可选设置 OPENAI_MODELS（逗号分隔自动切换）。"
    )


@app.post("/api/ai", response_class=PlainTextResponse)
async def api_ai(req: Request) -> PlainTextResponse:
    payload: Dict[str, Any] = {}
    try:
        payload = await req.json()
    except Exception:
        payload = {}

    # 只要配置了 Key（或显式 base_url），就走百炼 OpenAI-Compatible 代理
    if _env("OPENAI_API_KEY") or _env("OPENAI_BASE_URL"):
        try:
            text = _proxy_openai_compatible(payload)
            return PlainTextResponse(text or "（模型未返回文本）")
        except Exception as e:
            # 代理失败也别让前端挂：返回可读错误
            return PlainTextResponse(f"代理调用失败：{e}", status_code=502)

    return PlainTextResponse(_fallback_answer(payload))


@app.get("/api/health")
async def api_health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "fastapi", "front_dir": _front_dir()})


@app.post("/api/login")
async def api_login(req: Request) -> JSONResponse:
    """
    兼容 front (3) 的 js/auth.js（原本预期 Node+Express+MySQL）。
    这里提供一个“本地演示版”（便于前端联调/演示）：
    - student：默认初始密码=学号；首次登录后必须改密（mustChangePassword=True）
    - admin：默认密码=账号（例如 admin001/admin001），不需要改密
    """
    try:
        payload = await req.json()
    except Exception:
        payload = {}

    role = str(payload.get("role") or "").strip()
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")

    if not username or not password:
        return JSONResponse({"error": "missing_credentials"}, status_code=400)
    if role not in ("student", "admin"):
        return JSONResponse({"error": "invalid_role"}, status_code=400)

    key = (role, username)
    u = _USERS.get(key)

    if u is None:
        # 首次出现的账号：写入默认规则（固定、可预测，避免“第一次输错就锁死”）
        if role == "student":
            u = {"password": username, "must_change_password": True, "profile": {"name": "", "phone": "", "avatar": ""}}
        else:
            u = {"password": username, "must_change_password": False, "profile": {"name": "", "phone": "", "avatar": ""}}
        _USERS[key] = u

    # 校验密码
    if str(u.get("password") or "") != password:
        # 兼容：若之前版本把 admin 密码写成了“首次输入的密码”，这里允许用“密码=账号”登录并自动修正
        if role == "admin" and password == username:
            u["password"] = username
            u["must_change_password"] = False
            _USERS[key] = u
        else:
            return JSONResponse({"error": "invalid_credentials"}, status_code=401)

    must_change = bool(u.get("must_change_password")) and role == "student"

    sess = {
        "role": role,
        "username": username,
        "displayName": username,
        "mustChangePassword": must_change,
    }
    resp = JSONResponse({"session": _public_session(sess)})
    _set_session(resp, sess)
    return resp


@app.get("/api/session")
async def api_session(req: Request) -> JSONResponse:
    sess = _get_session(req)
    if not sess:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    return JSONResponse({"session": _public_session(sess)})


@app.post("/api/logout")
async def api_logout(req: Request) -> JSONResponse:
    resp = JSONResponse({"ok": True})
    _clear_session(resp, req)
    return resp


@app.post("/api/password")
async def api_password(req: Request) -> JSONResponse:
    sess = _get_session(req)
    if not sess:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    if sess.get("role") != "student":
        return JSONResponse({"error": "invalid_role"}, status_code=400)
    try:
        payload = await req.json()
    except Exception:
        payload = {}
    new_pwd = str(payload.get("newPassword") or "")
    if len(new_pwd) < 6:
        return JSONResponse({"error": "password_too_short"}, status_code=400)

    key = (sess["role"], sess["username"])
    u = _USERS.get(key) or {"password": sess["username"], "must_change_password": True, "profile": {}}
    u["password"] = new_pwd
    u["must_change_password"] = False
    _USERS[key] = u

    sess["mustChangePassword"] = False
    resp = JSONResponse({"session": _public_session(sess)})
    _set_session(resp, sess)
    return resp


# 可选：直接用后端同时托管前端页面
# 访问： http://127.0.0.1:8000/overview.html
front_dir = _front_dir()
if os.path.isdir(front_dir):
    app.mount("/", StaticFiles(directory=front_dir, html=True), name="front")

