"""
应用组装工厂（v2 阶段 1 第 1 步：重构地基）。

把 api/main.py lifespan 里的组件组装逻辑抽成唯一真相源：
生产服务（api/main.py）与基线评测 CLI（scripts/run_baseline_eval.py，第 7 步）
都从这里取组装逻辑，保证「跑基线」与「线上」是同一套系统，杜绝双份组装漂移。

历史教训：api/main.py 旧交互式 CLI 手抄过一份组装代码，redis/chroma 默认值
已与生产分叉（localhost vs 容器服务名）——两份手写组装必然漂移，所以抽工厂。

设计约定：
- 每个 build_* 函数对应一个组件；环境变量读取与默认值与生产现状逐字一致，
  修改组装逻辑只改这里。
- 组件 import 放在函数内部（与旧 lifespan 的延迟 import 同款）：模块导入
  保持轻量，chromadb/redis/anthropic 等重型依赖仍在 assemble() 运行时
  才加载，启动日志时机与重构前一致。
- assemble() 是生产完整形态，内部顺序与旧 lifespan 一字不差（monitor.start
  在 evaluator 构造之前——顺序有讲究，勿调换）。
- 基线评测 CLI 不需要 monitor/evaluator 时，可以只调用它需要的 build_* 组合，
  不必非用 assemble()。
"""
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    # 只给类型检查器看的导入（运行时组件仍在各 build_* 函数内部延迟 import，
    # 保证模块导入轻量——见模块 docstring 设计约定）
    from agents.agent_orchestrator import AgentOrchestrator
    from core.intent_recognizer import IntentRecognizer
    from core.skill_loader import SkillManager
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

# 项目根目录（本文件在 core/ 下，上一级即根），用于 skills 等目录的默认值
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())


def build_llm_cfg() -> Dict[str, Any]:
    """从环境变量中读取大模型配置，返回一个字典。键是字符串，值不限。"""
    key = os.getenv("ANTHROPIC_API_KEY", "")    #从环境变量中读取key。
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")   #没有key则显示未设置
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),  #读anthropic model，没有的话默认用claude3.5sonnet
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:          #读可选代理地址，有的话再放进字典
        cfg["base_url"] = base_url
    return cfg


def build_judge_cfg(system_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """判卷模型配置：与被测系统的 LLM 配置分开（2026-10-15 Qwen3 基线准备）。

    为什么要分开：换 Qwen3 跑基线时，被测系统改走 vLLM（OPENMED_LLM_PROTOCOL=openai
    + ANTHROPIC_MODEL 改成 Qwen3），判卷必须仍是固定的第三方 DeepSeek——否则判卷
    跟着 build_llm_cfg 一起变成 Qwen3，等于被测模型给自己打分，而且训练前后
    判卷标准不同，数字没法比。

    读 OPENMED_JUDGE_API_KEY / OPENMED_JUDGE_BASE_URL / OPENMED_JUDGE_MODEL /
    OPENMED_JUDGE_PROTOCOL；没设的项沿用被测系统配置，协议默认 anthropic（不跟着
    OPENMED_LLM_PROTOCOL 变）——一个 OPENMED_JUDGE_* 都不设时，与改动前完全一样。
    explicit = 是否显式设了判卷模型，基线 CLI 用它拦「被测系统走自部署模型、
    判卷却没单独配」的情况。
    """
    system_cfg = system_cfg if system_cfg is not None else build_llm_cfg()
    judge_model = os.getenv("OPENMED_JUDGE_MODEL", "").strip()
    cfg: Dict[str, Any] = {
        "api_key":  os.getenv("OPENMED_JUDGE_API_KEY", "").strip() or system_cfg["api_key"],
        "model":    judge_model or system_cfg["model"],
        "protocol": os.getenv("OPENMED_JUDGE_PROTOCOL", "").strip().lower() or "anthropic",
        "explicit": bool(judge_model),
    }
    base_url = os.getenv("OPENMED_JUDGE_BASE_URL", "").strip() or system_cfg.get("base_url", "")
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def build_intent_recognizer(cfg: Dict[str, Any]) -> "IntentRecognizer":
    """独立意图识别器实例（给 EndToEndEvaluator 用；Orchestrator 内部还会创建一个）。"""
    from core.intent_recognizer import IntentRecognizer
    return IntentRecognizer(
        api_key=cfg["api_key"],            #意图识别器接收三个输入：api key, base_url, model
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )


def build_skill_manager() -> "SkillManager":
    """Skills：从环境变量（没有的话，默认值用skills目录下的）加载业务能力说明，并在 Agent 调用 LLM 时动态注入。"""
    from core.skill_loader import SkillManager
    skills_dir = os.getenv("OPENMED_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    manager = SkillManager(
        root_dir=skills_dir,      #skill管理器接收两个参数，skill_dir目录和最大prompt字符数，后者没有的话，默认值5000
        max_prompt_chars=int(os.getenv("OPENMED_SKILLS_MAX_PROMPT_CHARS", "5000")), #防止注入的prompt长度太长
    )
    manager.load()
    return manager


def build_orchestrator(cfg: Dict[str, Any], skill_manager: "SkillManager") -> "AgentOrchestrator":
    """Agent 编排器（多 Agent 编排的核心）。

    RAG 工具不在这里传入：与生产现状一致，先构造编排器，待知识库工具注册完成后
    由 assemble() 显式调用 set_shared_tools 注入（两步顺序保持，勿合并）。
    """
    from agents.agent_orchestrator import AgentOrchestrator
    return AgentOrchestrator(
        api_key=cfg["api_key"],             #agent编排器接收四个参数，api，baseurl，model，skill管理器
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )


def build_memory(cfg: Dict[str, Any]) -> "MemoryManager":
    """记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）。"""
    from memory.conversation_memory import MemoryManager
    return MemoryManager(                #记忆管理器接收七个参数,a+b+m, 三个chromaDB配置，一个redis配置
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),  #本地持久化目录
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )


def build_tool_manager(cfg: Dict[str, Any]) -> "MCPToolManager":
    """MCP 工具管理器。

    eval_mode（v2 评测模式）：True 时禁用工具缓存，保证轨迹记录真实检索行为。
    默认关闭——线上开启会显著增加检索延迟（每次查询都真实检索+精排）。
    由 OPENMED_EVAL_MODE 环境变量统一读取后传入。
    """
    from mcp.tool_manager import MCPToolManager
    return MCPToolManager(
        api_key=cfg["api_key"],             #mcp工具管理器接收三个参数,abm。为什么工具管理器也需要llm？要决定调用哪个工具
        base_url=cfg.get("base_url"),       #见tool_manager.py
        model=cfg["model"],
        eval_mode=os.getenv("OPENMED_EVAL_MODE", "0").strip().lower() in ("1", "true", "on", "yes"),
    )


def build_kb() -> "KnowledgeBase":
    """RAG 知识库（基于 ChromaDB 的真实检索），chroma 配置与记忆管理器一致。"""
    from mcp.knowledge_base import KnowledgeBase
    return KnowledgeBase(                     #知识库类，参数和chromaDB记忆管理器配置的一致，后称为kb
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )


def register_knowledge_tool(tool_manager: "MCPToolManager", kb: "KnowledgeBase") -> None:
    """把知识库注册成 knowledge_search 工具（本体是 KnowledgeBase）。

    含降级 fallback：防止一次查询失败拖垮整个流程，返回结构与正常结果一致，
    上层不用改代码。
    """
    from mcp.tool_manager import Tool

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")     #拿到query，没有就赋值为空
        return [{                           #此处返回的结果和正常结果一致，所以上层不用改代码
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的检索。请稍后重试，或转人工/急诊确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    tool_manager.register(Tool(
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


def build_monitor(orchestrator: "AgentOrchestrator", tool_manager: "MCPToolManager") -> "PerformanceMonitor":
    """性能监控（可选启动 Prometheus，默认为0，即不启动）。"""
    from monitor.performance_monitor import PerformanceMonitor
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    return PerformanceMonitor(
        orchestrator=orchestrator,   #要监控的对象有两个：编排器和工具管理器（看重延迟，资源占用，系统稳定性，调用成功率等等）
        tool_manager=tool_manager,   #工具模块是很容易报错的，如超时，返回异常，缓存命中率低等
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),      #采集间隔，默认10秒
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,         #告警地址，默认none。用于模型数据表现异常时发送信息。一般放到环境变量里，且不提交到仓库
        prometheus_port=prom_port,
    )


def build_evaluator(
    cfg: Dict[str, Any],
    orchestrator: "AgentOrchestrator",
    recognizer: "IntentRecognizer",
) -> "EndToEndEvaluator":
    """端到端评测器（旧四维 judge 评测，/eval/run 用；v2 基线评测走第 5/7 步的新判分器）。

    判卷用 build_judge_cfg（与被测系统分开，见其 docstring）；不设 OPENMED_JUDGE_*
    时与被测系统同一配置，行为与改动前一致。
    """
    from evaluation.evaluator import EndToEndEvaluator
    judge_cfg = build_judge_cfg(cfg)
    return EndToEndEvaluator(
        orchestrator=orchestrator,   #要评测的对象有两个：编排器和意图识别器。使用llm端到端、四维度评测（看重回答质量，意图生成质量等）
        recognizer=recognizer,
        api_key=judge_cfg["api_key"],
        base_url=judge_cfg.get("base_url"),
        model=judge_cfg["model"],
        protocol=judge_cfg["protocol"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),  #基线文件路径，默认后者
    )


@dataclass
class AppContext:
    """assemble() 的产物：生产链路全套组件。

    生产 lifespan 把字段分发给全局变量供各端点使用；
    基线评测 CLI 直接持有本对象跑题（此时 monitor/evaluator 用不上，但保持
    与生产同构，不另起炉灶）。
    """
    cfg:           Dict[str, Any]                 # LLM 配置（api_key/model/base_url）
    skill_manager: "SkillManager"                 # skill 热加载管理器
    recognizer:    "IntentRecognizer"             # 独立意图识别器（evaluator 用）
    orchestrator:  "AgentOrchestrator"            # Agent 路由与编排器
    memory:        "MemoryManager"                # 上下文记忆管理器
    tool_manager:  "MCPToolManager"               # MCP 工具管理器（含 knowledge_search）
    kb:            "KnowledgeBase"                # RAG 知识库
    monitor:       "Optional[PerformanceMonitor]"  # 性能监控（评测 CLI 分支为 None）
    evaluator:     "EndToEndEvaluator"            # 端到端评测器


async def assemble(*, start_monitor: bool = True) -> AppContext:
    """按生产顺序组装全套组件（与旧 api/main.py lifespan 顺序一字不差）。

    顺序有讲究：orchestrator 先构造（内部先挂一套空工具白名单）→ 知识库
    注册进 tool_manager → 再用 build_shared_rag_tools 覆盖注入，两步缺一不可。
    monitor.start 在 evaluator 构造之前，保持旧启动时序。

    start_monitor：生产 lifespan 用默认 True；基线评测 CLI 传 False——
    容器里生产 API 已占着 Prometheus 端口，而监控的端口绑定发生在
    PerformanceMonitor 构造阶段（prometheus_client.start_http_server 在
    __init__ 里调用），CLI 再构造也会「地址已被占用」直接崩
    （2026-10-13 pilot 实测）。评测用不上监控，连构造一起跳过，monitor 置 None。
    """
    from agents.agent_orchestrator import build_shared_rag_tools

    cfg = build_llm_cfg()  #导入并在日志输出前面配置
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    recognizer = build_intent_recognizer(cfg)
    skill_manager = build_skill_manager()
    orchestrator = build_orchestrator(cfg, skill_manager)
    memory = build_memory(cfg)
    tool_manager = build_tool_manager(cfg)
    kb = build_kb()
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")  #异步拿到从知识库加载返回的文档数，打印到日志

    register_knowledge_tool(tool_manager, kb)
    orchestrator.set_shared_tools(build_shared_rag_tools(tool_manager))  #从工具管理器类中，构建agent编排器可用的rag工具集。意思是多个agent可以用同一套rag工具

    monitor = build_monitor(orchestrator, tool_manager) if start_monitor else None
    if monitor is not None:
        await monitor.start()  #异步启动监控（旧时序：先于 evaluator 构造）

    evaluator = build_evaluator(cfg, orchestrator, recognizer)

    return AppContext(
        cfg=cfg,
        skill_manager=skill_manager,
        recognizer=recognizer,
        orchestrator=orchestrator,
        memory=memory,
        tool_manager=tool_manager,
        kb=kb,
        monitor=monitor,
        evaluator=evaluator,
    )
