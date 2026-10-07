"""
亮点：MCP 工具调用框架

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 混合召回（BM25 + 向量）—— 检索类工具（如 knowledge_search）在
     mcp/knowledge_base.py 内部做 BM25 关键词召回 + 向量语义召回，
     解决单路召回"漏召回"的问题（例如纯向量检索对药品名等短词不稳定，
     纯关键词检索理解不了语义相近但字面不同的问法）。
  2. RRF 融合 + Cross-Encoder 精排（Reranking）—— 两路召回结果先用
     Reciprocal Rank Fusion 融合成一个候选池，再用 core/reranker.py 的
     Cross-Encoder 对候选池做精排，解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.llm_utils import create_llm_client

from core.reranker import (
    CrossEncoderReranker,
    RERANK_CROSS_ENCODER,
    RERANK_LEXICAL_FALLBACK,
)

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:          bool
    data:             Any
    tool_name:        str
    error:            Optional[str] = None
    cached:           bool = False
    latency_ms:       float = 0.0
    reranked:         bool = False   # 是否经过 Cross-Encoder 真精排（词重叠降级不算）
    rerank_degraded:  bool = False   # 精排是否降级（模型不可用/推理失败，用了词重叠兜底）


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    检索类工具的核心优化链路：
      混合召回（BM25 + 向量，在工具 handler 内部完成，见 mcp/knowledge_base.py）
      → RRF 融合 → Cross-Encoder 精排（search_with_rerank）→ 返回 Top-K
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        eval_mode: bool = False,
    ):
        """
        eval_mode（v2 评测模式）：True 时禁用工具缓存读写。
        原因：缓存会吞掉重复搜索、跨请求共享结果，轨迹失真，评测数字不可信。
        由 OPENMED_EVAL_MODE 环境变量在 api/main.py 统一读取后传入，默认关闭。
        """
        #双协议客户端（v2 阶段 0.5）：按 OPENMED_LLM_PROTOCOL 切换 Anthropic/OpenAI 协议
        self._client = create_llm_client(api_key=api_key, base_url=base_url)
        self._model  = model
        self._eval_mode = eval_mode
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at, reranked)
        self._reranker = CrossEncoderReranker()

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        # 缓存命中（评测模式下跳过：保证每次调用都是真实执行）
        if use_cache and not self._eval_mode and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached_data,
                    tool_name=name,
                    cached=True,
                    reranked=cached_reranked,
                )

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            if tool.cache_ttl > 0 and not self._eval_mode:
                self._set_cache(name, params, data, tool.cache_ttl, reranked=False)

            return ToolResult(success=True, data=data, tool_name=name, latency_ms=latency)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 检索优化链路：混合召回 → Cross-Encoder 精排 ──────────────────────────

    async def search_with_rerank(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        recall_k: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        完整的检索优化链路：混合召回（BM25+向量，工具内部完成）→ RRF 融合
        （同样在工具内部完成）→ Cross-Encoder 精排 → Top-K

        recall_k 控制向工具请求的候选池大小，默认取 top_k 的 4 倍，
        为精排留出足够的候选空间；工具 handler（如 KnowledgeBase.search_handler）
        负责返回已按 BM25+向量+RRF 融合排序的候选列表。
        """
        recall_k = recall_k or max(top_k * 4, 20)
        result = await self.call(tool_name, {"query": query, "top_k": recall_k}, context, use_cache=True)

        if not result.success or not isinstance(result.data, list) or not result.data:
            return result

        if len(result.data) <= top_k:
            return result

        # 用 rerank_with_status 拿到真实状态：只有 Cross-Encoder 模型真实精排才算 reranked=True，
        # 降级为词重叠打分时 reranked=False + rerank_degraded=True，评测/轨迹才能分辨
        reranked_data, rerank_status = await asyncio.to_thread(
            self._reranker.rerank_with_status, query, result.data, top_k
        )
        return ToolResult(
            success=True,
            data=reranked_data,
            tool_name=tool_name,
            cached=result.cached,   # 保留缓存命中标志：召回阶段是否走了缓存（精排总是新鲜执行的）
            latency_ms=result.latency_ms,
            reranked=(rerank_status == RERANK_CROSS_ENCODER),
            rerank_degraded=(rerank_status == RERANK_LEXICAL_FALLBACK),
        )

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict) -> str:
        payload = {"params": params}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(name, params)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        reranked: bool = False,
    ) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params)] = (data, time.monotonic() + ttl, reranked)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
