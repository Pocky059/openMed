"""
OpenMed 智能问诊分流系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
负责生命周期管理、核心组件初始化、依赖组装，HTTP API请求记入记忆、意图识别和Agent编排主链路
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
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()  #导入env配置到环境变量

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),       #调用logging类。从环境变量读取日志level，没有就用INFO，logging.INFO = 20
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",   #定义每条日志的格式。时间+级别+logger名字+日志内容
)                                                                 #如 2026-09-17 10:23:45,123 [INFO] main: 知识库已加载: 128 个文档片段
logger = logging.getLogger(__name__)          #当前logger的名字定义为当前模块的名字__name__

BANNER = r"""

  OpenMed   v1.0
  智能问诊分流系统

"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None    #agen路由与编排器
_memory       = None    #上下文记忆管理器
_tool_manager = None    #mcp工具管理器
_monitor      = None    #监测
_evaluator    = None    #评估
_skill_manager = None   #skill热加载管理器

def _anthropic_cfg() -> Dict[str, Any]:     #从环境变量中读取大模型配置，返回一个字典。键是字符串，值不限。
    key = os.getenv("ANTHROPIC_API_KEY", "") #从环境变量中读取key。
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")  #没有key则显示未设置
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),  #读anthropic model，没有的话默认用claude3.5sonnet
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:          #读可选代理地址，有的话再放进字典
        cfg["base_url"] = base_url
    return cfg


#核心函数（使用装饰器，把lifespan用这个类包一层再赋值回去这个名字） |  先执行yield前面的代码，然后系统初始化完成，yield，fastapi开始接受请求，然后服务器关闭，继续执行yield后面的代码
@asynccontextmanager
async def lifespan(app: FastAPI):  #lifespan是一个异步事件循环函数，哪个事件“就绪”了，就执行对应异步await函数
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager #后面要修改这些全局变量，前面初始化为了None

    print(BANNER, flush=True)  #打印logo

    from agents.agent_orchestrator import AgentOrchestrator, Request, build_shared_rag_tools  #从编排器库导入编排器，请求体
    from core.intent_recognizer import IntentRecognizer                                       #从意图识别器库导入意图识别器
    from evaluation.evaluator import EndToEndEvaluator                                        #从评估库导入端到端评估器
    from mcp.knowledge_base import KnowledgeBase                                              #从知识库导入知识库类
    from mcp.tool_manager import MCPToolManager, Tool                                         #从工具管理器库导入mcp管理器类，工具类
    from memory.conversation_memory import MemoryManager                                      #从记忆管理库导入记忆管理器类
    from monitor.performance_monitor import PerformanceMonitor                                #从监管器库导入监管器
    from core.skill_loader import SkillManager                                                #从skill加载器库导入skill管理器

    cfg = _anthropic_cfg()  #导入并在日志输出前面配置
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（这里是单独给 Evaluator用的，Orchestrator 内部还会创建一个）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],            #意图识别器接收三个输入：api key, base_url, model
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从环境变量（没有的话，默认值用skills目录下的）加载skills（主要是业务能力说明），并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("OPENMED_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,      #skill管理器接收两个参数，skill_dir目录和最大prompt字符数，后者没有的话，默认值5000
        max_prompt_chars=int(os.getenv("OPENMED_SKILLS_MAX_PROMPT_CHARS", "5000")), #防止注入的prompt长度太长
    )
    _skill_manager.load()

    # Agent 编排器(orchestrator)，多Agent编排的核心
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],             #agent编排器接收四个参数，api，baseurl，model，skill管理器
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(                #记忆管理器接收七个参数,a+b+m, 三个chromaDB配置，一个redis配置
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),  #本地持久化目录
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],             #mcp工具管理器接收三个参数,abm。为什么工具管理器也需要llm？要决定调用哪个工具
        base_url=cfg.get("base_url"),       #见tool_manager.py
        model=cfg["model"],
    )
    kb = KnowledgeBase(                     #知识库类，参数和chromaDB记忆管理器配置的一致，后称为kb
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")  #异步拿到从知识库加载返回的文档数，打印到日志


    #知识库降级函数，防止一次查询失败拖垮整个流程. | params:工具调用参数，如query:.....， topk: 5....  | context：可选上下文 | error: 错误信息字符串
    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")     #拿到query，没有就赋值为空
        return [{                           #此处返回的结果和正常结果一致，所以上层不用改代码
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的检索。请稍后重试，或转人工/急诊确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(         #把知识库注册成工具,本体是knowledge base
        name="knowledge_search",
        description="搜索医疗知识库（BM25 + 向量混合召回，RRF 融合排序）",
        handler=kb.search_handler,       #这个是工具被agent调用后，实际的调用对象。这里调用之前创建的知识库类实例kb
        schema={                         #这个是JSON Schema，用来定义工具参数，约束模型输出不乱编（参数不对直接拦下）。
            "type": "object",            #object是JSON的数据类型之一（键值对集合，有序列表，字符串，数字，布尔，空）
            "properties": {
                "query": {"type": "string"},    #用户搜了什么
                "top_k": {"type": "integer"},   #返回的前k条结果
            },
            "required": ["query"],        #必填字段：query。不填会校验失败。
        },
        cache_ttl=300.0,                  #结果缓存，同样的搜索在5分钟内重读调用直接返回。
        fallback=knowledge_fallback,      #fallback使用上面定义的fallback函数
    ))
    if _orchestrator is not None:         #防御性检查，防止前面初始化失败
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))   #从工具管理器类中，构建agent编排器可用的rag工具集。意思是多个agent可以用同一套rag工具

    # 性能监控（可选启动 Prometheus，默认为0，即不启动）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,   #要监控的对象有两个：编排器和工具管理器（看重延迟，资源占用，系统稳定性，调用成功率等等）
        tool_manager=_tool_manager,   #工具模块是很容易报错的，如超时，返回异常，缓存命中率低等
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),      #采集间隔，默认10秒
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,         #告警地址，默认none。用于模型数据表现异常时发送信息。一般放到环境变量里，且不提交到仓库
        prometheus_port=prom_port,
    )
    await _monitor.start()  #异步启动

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,   #要评测的对象有两个：编排器和意图识别器。使用llm端到端、四维度评测（看重回答质量，意图生成质量等）
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),  #基线文件路径，默认后者
    )

    logger.info("OpenMed 已就绪")
    yield   #启动前会先执行yield前的所有代码，因此打印完日志文件才会启动服务。关闭后执行之后的代码。

    await _monitor.stop()       #程训结束，执行yield后面的代码。停止监控。
    if _memory is not None:     #关闭memory，释放redis连接，ChromaDB客户端等。防御性写法，防止初始化失败
        await _memory.close()
    if _orchestrator is not None:   #停止轨迹写线程，把队列中剩余轨迹落盘（v2 阶段 0.1）
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

class ChatResponse(BaseModel):         #这里就是回答体了。他会自动把回答转成JSON输出。
    conv_id:     str                   #必填字段：对话id
    request_id:  str = ""
    response:    str                   #必填字段：模型回答
    intent:      str                   #必填字段：识别出的意图
    intent_group: str = "other"
    agent_type:  str                   #必填字段：处理该请求的agent类型
    agent_types: List[str] = Field(default_factory=list)         #使用可变默认值写法
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool                  #必填字段：是否转人工
    latency_ms:  float                 #必填字段：处理耗时（毫秒）
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)


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
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())            #uuid：python库，用于生成唯一标识符，基于随机数

    # 1. 读取记忆上下文（工作记忆+情景记忆+用户画像一起保存成的上下文；异步执行。注意这里的mem_ctx和后面的full_context都不是ui上显示的文本）
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

    # 2. 构建编排请求（含对话历史，用于意图识别上下文）
    history = [
        {"role": m.role.value, "content": m.content} #从mem_ctx中抽取每个m形成字典列表，即格式化上下文
        for m in mem_ctx.recent_messages[-5:]        #只取最近的五条消息，因此主要是redis工作记忆
    ] if mem_ctx.recent_messages else None

    intent_result = await _orchestrator.recognize_intent(req.message, history=history)   #把请求体里的message（当前信息）和工作记忆作为输入发送到意图识别模块，得到返回
    full_context = mem_ctx.to_prompt_text()   #将整个mem_ctx记忆上下文格式化为大模型可用的prompt，起整体参考作用

    orch_req = OrcReq( #把结果先包装成一个请求体，再喂给编排器。这里的请求体OrcReq是Request类，它是在orchestrator里被定义的。
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,           #这两个记忆现在都被送到Request中，计算agent路由了：
        context=full_context,      #并入历史（格式化综合上下文记忆）
        history=history,           #工作记忆（结构化）
        entities=intent_result.entities,       
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        intent_confidence=intent_result.confidence,
    )

    # 3. 送入编排器，决定接下来交给哪个agent，用什么工具，要不要多agent等
    result = await _orchestrator.run(orch_req)

    # 4. 写入记忆（写入redis，工作记忆）
    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)   #写入用户提问
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)    #写入模型回答

    # 5. 异步更新用户画像（不阻塞响应）
    asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

    return ChatResponse(    #前面定义的返回体，包含若干参数。
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=intent_result.intent_group,
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],    #result里有两个字段，一个agent_type, 一个agent_types
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,  #此外还有一个primary_agent字段。没有primary就用agent_type
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],     #还有一个supporting_agents字段
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used="search_knowledge_base" in result.tools_used,
        entities=intent_result.entities,
        intent_confidence=round(intent_result.confidence, 4),
        intent_source_scores=intent_result.source_scores,
    )

#这个构建rag上下文函数并没有在/chat中用上。推测是因为rag后来被封装到工具层中，由agent决定是否调用。
async def _build_knowledge_context(message: str, intent=None, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的混合召回（BM25+向量）、RRF 融合、Cross-Encoder 精排、fallback 能力。
    """
    if _tool_manager is None: #0 排除掉未初始化和不用查知识库的情况。该函数定义在下面。
        return "", False
    if not _should_use_knowledge(message, intent=intent):
        return "", False
    try:        #把拿到的检索结果result放到上下文中。是完整的知识库查询链路，包括混合召回，RRF融合，精排，topk等。用try防止任何报错挂掉（网络失败，超时，返回格式不对等等）
        result = await _tool_manager.search_with_rerank("knowledge_search", message, top_k=top_k)
        if not result.success or not isinstance(result.data, list) or not result.data:  #1 判断搜索失败，数据不是列表，数据为空时不返回
            return "", False

        parts = ["[知识库检索结果]"]
        used = False   #该处used配合下方if not used使用。防御性编程，先设置为false，只有真正处理了一条有效结果，通过层层判断，才设置为true
        for i, item in enumerate(result.data[:top_k], start=1):   #遍历结果，最多取k条
            if not isinstance(item, dict):                        #2 上面判断了result是不是列表，还要判断里面的item是不是字典
                continue
            title = str(item.get("title", "未命名文档"))           #每条取标题，内容，相关度分数
            content = str(item.get("content", "")).strip()        
            score = item.get("score", "")
            if not content:                                       #3 判断完字典，还要判断里面是不是空内容
                continue
            used = True       #查询成功后，是否使用了rag的标签-used设置为true
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")  #组装输出，限制600字符防止过长

        if not used:
            return "", False                                                                  #如果未使用rag则什么都不返回
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，再结合通用医疗问诊能力说明，不要补造诊断结论。")   #若使用rag，默认prompt模板
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False           #任何降级，异常都吞掉，返回空避免影响主程序。记录日志。


def _should_use_knowledge(message: str, intent=None) -> bool:
    """跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:                                                                           #消息为空，不检索
        return False
    intent_value = getattr(intent, "value", intent)
    if intent_value in {"greeting", "feedback", "emergency", "human_handoff", "other"}:   #1监测意图。寒暄、急症（走急症流程不查知识库）、人工接管问题和other不检索
        return False
    if intent_value in {
        "query", "request", "medication", "appointment", "complaint",
        "symptom_check", "department_guide", "appointment_manage",
        "appointment_reschedule", "appointment_cancel", "medical_receipt",
        "appointment_payment_issue", "medication_dosage", "medication_interaction",
    }:
        return True                                                                       #意图包含业务关键词，可以检索。
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}            #2监测意图之后的粗略关键词。直接检测到消息属于“寒暄”，不检索
    if msg in greetings:
        return False
    business_keywords = [
        "挂号", "预约", "科室", "门诊", "专家号", "退号", "改期",
        "吃药", "用药", "服药", "药品", "说明书", "剂量", "副作用", "不良反应",
        "发烧", "咳嗽", "头晕", "肚子疼", "腹泻", "皮疹", "呕吐", "症状",
        "appointment", "medication", "symptom", "department",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords) #消息长度 >= 4 个字符，或者包含任一业务关键词，就认为该查知识库。
                                                                       #最粗略，属于意图和词表都判断不出来TF之后，使用的托底方法

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
    context: Optional[Dict[str, Any]] = None


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
                context=c.context,
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
