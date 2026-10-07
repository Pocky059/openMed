"""
v2 阶段 0.2 评测可控性开关的单元测试。

覆盖三块行为：
1. OPENMED_EVAL_MODE 对应 eval_mode=True 时，工具缓存读写全部跳过；
2. 普通模式下缓存仍正常工作（回归保护）；
3. search_with_rerank 忠实传递 cached / reranked / rerank_degraded 三个标志。

全部离线运行，不依赖网络或模型文件。
"""
import asyncio

from core.reranker import (
    CrossEncoderReranker,
    RERANK_CROSS_ENCODER,
    RERANK_LEXICAL_FALLBACK,
    RERANK_SKIPPED,
)
from mcp.tool_manager import MCPToolManager, Tool


def make_manager(eval_mode: bool):
    """构造一个不联网的工具管理器（AsyncAnthropic 构造时不发请求）。"""
    return MCPToolManager(api_key="test", model="test-model", eval_mode=eval_mode)


def make_counting_tool(counter: dict):
    """注册一个带缓存、且能统计真实执行次数的工具。"""

    async def handler(params, context):
        counter["calls"] += 1
        return {"value": counter["calls"]}

    return Tool(
        name="counter",
        description="测试用计数工具",
        handler=handler,
        schema={"type": "object", "properties": {"q": {"type": "integer"}}, "required": ["q"]},
        cache_ttl=60.0,
    )


def test_eval_mode_skips_cache_read_and_write():
    """评测模式下：同样参数调用两次，两次都必须真实执行。"""
    mgr = make_manager(eval_mode=True)
    counter = {"calls": 0}
    mgr.register(make_counting_tool(counter))

    r1 = asyncio.run(mgr.call("counter", {"q": 1}))
    r2 = asyncio.run(mgr.call("counter", {"q": 1}))

    assert r1.cached is False and r1.success
    assert r2.cached is False and r2.success
    assert counter["calls"] == 2, "评测模式下缓存必须完全失效"


def test_normal_mode_cache_still_works():
    """普通模式下：第二次同参调用应命中缓存，不再真实执行（回归保护）。"""
    mgr = make_manager(eval_mode=False)
    counter = {"calls": 0}
    mgr.register(make_counting_tool(counter))

    r1 = asyncio.run(mgr.call("counter", {"q": 1}))
    r2 = asyncio.run(mgr.call("counter", {"q": 1}))

    assert r1.cached is False and r1.success
    assert r2.cached is True, "普通模式下第二次调用应命中缓存"
    assert counter["calls"] == 1


class FakeReranker:
    """替身精排器：按预设状态返回，顺便记录调用次数（精排应永远新鲜执行，不被缓存）。"""

    def __init__(self, status: str = RERANK_CROSS_ENCODER):
        self.status = status
        self.calls = 0

    def rerank_with_status(self, query, items, top_k):
        self.calls += 1
        return list(items[:top_k]), self.status


def make_search_manager(eval_mode: bool, reranker: FakeReranker):
    """构造一个带知识检索工具的 manager，handler 返回 10 条候选。"""
    mgr = make_manager(eval_mode=eval_mode)
    mgr._reranker = reranker  # 替换成替身，避免测试加载真实模型

    async def handler(params, context):
        return [{"title": f"doc{i}", "content": f"内容{i}"} for i in range(10)]

    mgr.register(Tool(
        name="knowledge_search",
        description="测试用检索工具",
        handler=handler,
        schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
        cache_ttl=60.0,
    ))
    return mgr


def test_search_with_rerank_preserves_cached_flag():
    """缓存命中时，search_with_rerank 必须保留 cached=True（此前被丢弃）。"""
    mgr = make_search_manager(eval_mode=False, reranker=FakeReranker())
    search = mgr.search_with_rerank

    r1 = asyncio.run(search("knowledge_search", "query", top_k=2))
    r2 = asyncio.run(search("knowledge_search", "query", top_k=2))

    assert r1.cached is False
    assert r2.cached is True, "缓存命中的标志必须透传到结果上"
    assert r1.reranked is True and r2.reranked is True, "精排每次都新鲜执行（即使候选来自缓存）"


def test_search_with_rerank_marks_cross_encoder_as_reranked():
    """真精排路径：reranked=True，rerank_degraded=False。"""
    mgr = make_search_manager(eval_mode=False, reranker=FakeReranker(RERANK_CROSS_ENCODER))
    result = asyncio.run(mgr.search_with_rerank("knowledge_search", "query", top_k=2))

    assert result.reranked is True
    assert result.rerank_degraded is False


def test_search_with_rerank_marks_lexical_fallback_as_degraded():
    """降级路径：reranked 必须为 False（不得假装精排生效），rerank_degraded=True。"""
    mgr = make_search_manager(eval_mode=False, reranker=FakeReranker(RERANK_LEXICAL_FALLBACK))
    result = asyncio.run(mgr.search_with_rerank("knowledge_search", "query", top_k=2))

    assert result.reranked is False
    assert result.rerank_degraded is True


def test_reranker_status_empty_and_single_item_skipped():
    """候选不足时不执行精排，状态为 skipped；旧接口 rerank() 行为不变。"""
    reranker = CrossEncoderReranker()

    items, status = reranker.rerank_with_status("查询", [], top_k=5)
    assert items == [] and status == RERANK_SKIPPED

    single = [{"content": "只有一条"}]
    items, status = reranker.rerank_with_status("查询", single, top_k=5)
    assert items == single and status == RERANK_SKIPPED
    assert reranker.rerank("查询", single, top_k=5) == single


def test_reranker_status_lexical_fallback_when_model_unavailable():
    """模型加载失败（白盒置 _load_failed）时：词重叠兜底，状态 lexical_fallback，结果带 rerank_score。"""
    reranker = CrossEncoderReranker()
    reranker._load_failed = True  # 白盒注入：跳过真实模型加载线程

    items = [
        {"content": "银杏叶提取物抑制血小板聚集"},
        {"content": "华法林是抗凝药物"},
        {"content": "感冒药与退烧药"},
    ]
    ranked, status = reranker.rerank_with_status("银杏叶 华法林", items, top_k=2)

    assert status == RERANK_LEXICAL_FALLBACK
    assert len(ranked) == 2
    assert all("rerank_score" in item for item in ranked), "兜底打分也应写入 rerank_score 字段"
