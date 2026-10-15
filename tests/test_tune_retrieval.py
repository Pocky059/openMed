"""
检索调参脚本（scripts/tune_retrieval.py）的单元测试。

覆盖：
1. weighted_rrf 与生产等权 RRF 的等价性（w=0.5 时 id 序列与分数逐位一致——
   这是「调参结论能直接套回生产」的前提，防止调参脚本自造一套融合逻辑）；
2. retrieve_top5 从缓存截断与独立计算的等价性（缓存优化的数学前提）；
3. 配置组合生成数量与去重；
4. 统计函数空输入防除零。

全部离线：不连 ChromaDB、不加载模型。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.knowledge_base import KnowledgeBase
from scripts.tune_retrieval import (
    PRODUCTION_DEFAULTS,
    aggregate,
    gen_configs,
    retrieve_top5,
    score_case,
    weighted_rrf,
)


def _fake_hits(ids: list) -> list:
    """按给定 id 顺序构造假召回列表（id 越靠前排名越高）。"""
    return [{"id": doc_id, "title": f"篇{doc_id}", "content": f"内容{doc_id}"} for doc_id in ids]


def test_weighted_rrf_matches_production_equal_weight():
    """w=1.0 时必须与生产 _reciprocal_rank_fusion 等权版逐位一致（回归锁）。"""
    v = _fake_hits(["a", "b", "c", "d"])
    b = _fake_hits(["c", "a", "e", "f"])
    fused = weighted_rrf(v, b, k=60, w=1.0)
    prod = KnowledgeBase._reciprocal_rank_fusion([v, b], k=60)
    assert [item["id"] for item in fused] == [item["id"] for item in prod]
    assert [item["rrf_score"] for item in fused] == [item["rrf_score"] for item in prod]


def test_weighted_rrf_prefers_vector_when_weight_high():
    """w=1.4 时向量路排名更值钱：向量路独有的 doc 应超过 BM25 独有的同级 doc。"""
    v = _fake_hits(["v1", "b1"])
    b = _fake_hits(["b1", "v1"])
    fused = weighted_rrf(v, b, k=60, w=1.4)
    assert fused[0]["id"] == "v1"  # v1 向量路第 1、BM25 第 2；b1 反之


def test_retrieve_top5_truncation_equals_independent():
    """从大缓存截断 recall_pool 再融合 == 直接用截断后的列表融合（缓存优化前提）。"""
    v = _fake_hits(["a", "b", "c", "d", "e", "f"])
    b = _fake_hits(["c", "a", "e", "g", "h", "i"])
    cfg = {"recall_pool": 4, "rrf_k": 60, "vector_w": 1.0, "recall_k": 10, "rerank": "off"}
    got = retrieve_top5(v, b, cfg, {})
    want = [item["id"] for item in weighted_rrf(v[:4], b[:4], k=60, w=1.0)[:5]]
    assert got == want


def test_retrieve_top5_rerank_on_uses_scores():
    """rerank=on 时 top5 由精排分决定，与 RRF 顺序无关（精排是最终裁决）。"""
    v = _fake_hits(["a", "b", "c"])
    b = _fake_hits(["x", "y", "z"])
    cfg = {"recall_pool": 3, "rrf_k": 60, "vector_w": 1.0, "recall_k": 4, "rerank": "on"}
    scores = {"a": 0.1, "b": 0.9, "c": 0.5, "x": 0.3, "y": 0.7, "z": 0.2}
    got = retrieve_top5(v, b, cfg, scores)
    assert got[:2] == ["b", "y"]  # 精排分最高的两个


def test_retrieve_top5_rerank_on_recall_k_limits_candidates():
    """recall_k=3 时精排只看 RRF 前 3，第 4 名分数再高也进不了 top5。"""
    v = _fake_hits(["a", "b", "c", "d"])
    b = _fake_hits(["c", "b", "a", "d"])  # d 在两路都垫底，RRF 明确排第 4
    cfg = {"recall_pool": 4, "rrf_k": 60, "vector_w": 1.0, "recall_k": 3, "rerank": "on"}
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.99}  # d 分最高但 RRF 第 4，被 recall_k 挡在精排外
    got = retrieve_top5(v, b, cfg, scores)
    assert "d" not in got


def test_gen_configs_count_and_unique():
    """组合数 = pool4×k3×w3×on(recall_k2)+off1 = 72+36 = 108，且无重复。"""
    configs = gen_configs()
    assert len(configs) == 108
    keys = [tuple(sorted(c.items())) for c in configs]
    assert len(set(keys)) == len(keys)


def test_gen_configs_off_has_no_recall_k():
    """rerank=off 时链路不经精排，recall_k 不参与（None，组合不重复展开）。"""
    offs = [c for c in gen_configs() if c["rerank"] == "off"]
    assert len(offs) == 36  # pool4 × k3 × w3
    assert all(c["recall_k"] is None for c in offs)


def test_production_defaults_in_grid():
    """生产默认参数必须是网格中的一个取值（否则「默认值排名」无从谈起）。"""
    configs = gen_configs()
    assert PRODUCTION_DEFAULTS in configs


def test_score_case_and_aggregate():
    """统计口径与基线一致（复用 qa_single_stats：MRR/hit@1/hit@5）；聚合空输入返回 0（防除零）。"""
    row = score_case(["x", "a", "c", "d", "e"], ["a", "alt"])
    assert row == {"mrr": 0.5, "hit_at_1": 0, "hit_at_5": 1}
    # 补标的正确片段（alt）排第 1 也算对（rev3 口径：正确片段 = gold ∪ alt）
    assert score_case(["alt", "a"], ["a", "alt"])["mrr"] == 1.0
    assert aggregate([])["mrr"] == 0.0
    assert aggregate([row])["hit_at_5"] == 1
