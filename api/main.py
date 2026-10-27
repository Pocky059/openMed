"""
OpenMed 智能问诊分流系统 — FastAPI 入口

HTTP 层只做绑定与序列化（v2 阶段 1 第 1 步重构后）：
- 组件组装逻辑在 core/app_factory.py（与基线评测 CLI 共用，杜绝双份漂移）
- /chat 五步主链路在 api/chat_service.py
- 生命周期管理、全局组件分发、其余各端点仍在本文文件
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())  #根目录加入sys.path
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from api.chat_service import ChatResponse, ChatService  #v2 阶段 1 第 1 步：五步链路与响应模型已外移
from core.app_factory import assemble, build_llm_cfg  #v2 阶段 1 第 1 步：组件组装已外移为工厂（lifespan 调用）

load_dotenv()  #导入env配置到环境变量

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),       #调用logging类。从环境变量读取日志level，没有就用INFO，logging.INFO = 20
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",   #定义每条日志的格式。时间+级别+logger名字+日志内容
)                                                                 #如 2026-09-17 10:23:45,123 [INFO] main: 知识库已加载: 128 个文档片段
logger = logging.getLogger(__name__)          #当前logger的名字定义为当前模块的名字__name__

BANNER = r"""

  OpenMed   v2.0.0
  智能问诊分流系统

"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None    #agen路由与编排器
_memory       = None    #上下文记忆管理器
_tool_manager = None    #mcp工具管理器
_monitor      = None    #监测
_evaluator    = None    #评估
_skill_manager = None   #skill热加载管理器
_chat_service = None    #/chat 五步链路服务

# LLM 配置读取已外移到 core/app_factory.build_llm_cfg（v2 阶段 1 第 1 步）。
# 保留别名：旧交互式 CLI（_cli）仍按原名调用，历史代码不动。
_anthropic_cfg = build_llm_cfg


#核心函数（使用装饰器，把lifespan用这个类包一层再赋值回去这个名字） |  先执行yield前面的代码，然后系统初始化完成，yield，fastapi开始接受请求，然后服务器关闭，继续执行yield后面的代码
@asynccontextmanager
async def lifespan(app: FastAPI):  #lifespan是一个异步事件循环函数，哪个事件“就绪”了，就执行对应异步await函数
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _chat_service #后面要修改这些全局变量，前面初始化为了None

    print(BANNER, flush=True)  #打印启动横幅

    # 全套组件由组装工厂统一构建（v2 阶段 1 第 1 步）：生产与基线评测 CLI
    # 共用同一份组装逻辑（core/app_factory.assemble），杜绝双份组装漂移。
    # 工厂内部已按生产顺序完成：意图识别器 → skills → 编排器 → 记忆 →
    # 工具管理器 → 知识库 → 工具注册 → 共享工具注入 → 监控启动 → 评测器
    # ——顺序有讲究，勿在生产与 CLI 两边各写一份。
    ctx = await assemble()

    # 分发给既有全局变量。其余端点（/health /skills /monitor /trace /search
    # /knowledge/* /eval/run）引用这些变量，全部保持不动。
    _skill_manager = ctx.skill_manager
    _orchestrator  = ctx.orchestrator
    _memory        = ctx.memory
    _tool_manager  = ctx.tool_manager
    _monitor       = ctx.monitor
    _evaluator     = ctx.evaluator
    _chat_service  = ChatService(ctx.orchestrator, ctx.memory)

    logger.info("OpenMed 已就绪")
    yield   #启动前会先执行yield前的所有代码，因此打印完日志文件才会启动服务。关闭后执行之后的代码。

    await _monitor.stop()       #程训结束，执行yield后面的代码。停止监控。
    if _memory is not None:     #关闭memory，释放redis连接，ChromaDB客户端等。兜底写法（同 _orchestrator：构造失败会终止启动，不会得到 None）
        await _memory.close()
    if _orchestrator is not None:   #停止轨迹写线程，把队列中剩余轨迹落盘（v2 阶段 0.1）。兜底写法（同 162 行）
        _orchestrator.close_trajectory_logger()
    logger.info("OpenMed 已关闭")  #打印日志，应用完全关闭


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(          #FastAPI是一个python的web框架，用来写HTTP API服务。核心特点包括原生异步处理，自动文档与swagger ui，性能好（基于pydantic），类型驱动等
    title="OpenMed 智能问诊分流系统",
    version="2.0.0",
    lifespan=lifespan,  #绑定上面的完整启动逻辑。fastapi启动时执行yield前的代码，关闭后执行之后的代码。
    docs_url="/docs",   #swagger ui的路径，访问这个网址能看到所有接口，还可以try it out测试。这是fastapi的招牌功能。
)

app.add_middleware(       #添加CORS中间件。中间件是请求与响应之间的中间处理层，常见的包括CORS,认证，日志，压缩，限流等。
    CORSMiddleware,       #CORS(Cross-Origin Resource Sharing)跨域资源共享。浏览器默认禁止一个网页向不同源（同源：端口，域名，协议都相同）的服务器发送请求，这个中间件就是说解除这个禁止，允许这些来源访问。
    allow_origins=["*"],  #允许任何来源访问（任何域名/端口）
    allow_methods=["*"],  #允许任何HTTP方法
    allow_headers=["*"],  #允许任何请求头
)                         #这里三个*还是很宽松的，只适合开发/演示。实际生产落地需要收紧。


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):          #BaseModel类来自于Pydantic。是FASTAPI的数据校验核心。pydantic会执行报错/转换/序列化/文档进swagger ui等操作。
    message:     str                   #必填字段：用户发的消息
    user_id:     str = "anonymous"     #可选字段：用户id。因为有默认值
    conv_id:     Optional[str] = None  #可选字段：对话id。同上。
                                       #后续api调用时(如/chat），FASTapi会自动把请求解析成ChatRequest对象（转JSON）

# ChatResponse 已外移到 api/chat_service.py（v2 阶段 1 第 1 步：服务层自洽，HTTP 层反向 import）


class ToolTraceResponse(BaseModel):    #工具追踪的响应体/结果体
    request_id: str     #必填：请求id
    found: bool         #必填：是否找到追踪记录
    trace: Dict[str, Any] = Field(default_factory=dict)      #追踪详情。字典，默认为空。


class RecentToolTracesResponse(BaseModel):     #最近工具追踪列表，用于返回最近若干条工具调用记录。
    items: List[Dict[str, Any]] = Field(default_factory=list)    #返回一个由字典组成的列表。默认为空。


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}    #get_stats方法见orchestrator.py


@app.post("/demo/reset", tags=["演示"])
async def reset_demo(user_id: str = "web-demo"):
    """重置演示用户的全部记忆（前端「重置记忆」按钮，面试演示用）。

    演示前端 user_id 写死 web-demo，且每刷新页面就开一个新 conv_id（会话数量
    不定），所以按用户维度全清：Redis 工作记忆/摘要 + ChromaDB 情景记忆/画像。
    （清空逻辑在 memory/conversation_memory.py 的 reset_user，返回删除计数。）
    """
    if _memory is None:
        raise HTTPException(503, "记忆管理器未就绪")
    deleted = await _memory.reset_user(user_id)
    return {"ok": True, "user_id": user_id, "deleted": deleted}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])              #重载skill（运行时就可以重载，即所谓热加载）
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()                                          #reload热加载方法，见skillmanager.py
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()

#核心流程
@app.post("/chat", response_model=ChatResponse)          #规定/chat接口的响应用ChatResponse回复体序列化
async def chat(req: ChatRequest):                        #需要一个输入，这里用到了前面的ChatRequest请求体
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    （五步实现已外移到 api/chat_service.ChatService，本端点只做 HTTP 绑定）
    """
    if _chat_service is None:
        # 兜底写法：lifespan 同步构造服务，失败会终止启动，正常运行时到不了这里；
        # 只防"未来初始化改为可失败/延迟创建"时，把未就绪状态如实报 503 而不是 500
        raise HTTPException(503, "服务未就绪")

    try:
        return await _chat_service.handle_chat_turn(req.message, req.user_id, req.conv_id)
    except Exception as ex:
        # 端点级兜底（2026-10-26 补齐）：LLM/依赖服务异常时返回降级响应而不是裸 500。
        # 前端拿到的是正常 ChatResponse 结构，能直接渲染「稍后再试」；错误详情进
        # 日志（完整堆栈可溯源），内部错误细节不泄漏给调用方。
        logger.exception(f"对话处理失败: {ex}")
        return ChatResponse(
            conv_id=req.conv_id or "",
            response="系统暂时不可用，请稍后再试。",
            intent="other",            # 降级响应不虚构意图，如实标记 other
            agent_type="none",         # 没有任何 agent 实际处理过
            escalated=False,
            latency_ms=0.0,
        )

@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)      #Response是FastAPI的底层响应类


@app.post("/search")           #只暴露检索这一步，为了单独测试其效果
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：BM25+向量混合召回 → RRF 融合 → Cross-Encoder 精排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rerank("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "布洛芬缓释胶囊说明书摘要", "content": "适应症：用于缓解轻至中度疼痛..."},
        {"title": "预约挂号与就诊流程说明", "content": "用户可以通过线上小程序或人工窗口预约挂号..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([{"title": d.title, "content": d.content} for d in body.documents])
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):   #UploadFile是FastAPI专门处理文件上传的类。File(...)表示参数来自表单文件的字段，...表示必填。
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")       #遇到无法解码的字节直接丢弃
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename  #从右边开始，按照. 分割，最多分一次。[0]即文件名，取文件名作为标题。
        docs = [{"title": title, "content": text}]

    count = await kb.add_documents_async(docs)
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": await kb.doc_count_async()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):  #可传一个评测集（请求体），也可不传（使用默认评测集）
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(    #运行评测
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("OpenMed CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("OPENMED_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("OPENMED_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nOpenMed [{result.agent_type.value}]: {result.response}\n")

    await mem.close()


# ── 前端静态挂载（2026-10-15 新增，单端口演示用）───────────────────────────────
# 前端是 frontend/ 下的 React+Vite 项目，`npm run build` 后产物在 frontend/dist/。
# 挂到根路径后浏览器打开 http://localhost 直接就是聊天界面；API 路由（/chat /docs
# /health 等）在 FastAPI 中优先级高于本挂载，不受影响。
# dist 目录不存在时跳过挂载——没 build 前端的纯后端运行不受任何影响。
_FRONTEND_DIST = pathlib.Path(_ROOT) / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        #跑FAST API
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
