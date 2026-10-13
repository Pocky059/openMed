"""
evaluation/ragas_metrics.py 的离线单测（v2 阶段 1 第 5 步配套）。

LLM 判卷用假客户端（按序弹出预设响应），向量用假 embedder 注入——
全程不联网；重试/兜底/坏响应路径逐条锁定（判卷函数的容错约定：
重试 2 次，最终失败兜底并标 judge_failed，不能把脏解析算成真分数）。
"""
import asyncio
import json

from evaluation.ragas_metrics import (
    cosine_similarity,
    f1_at_k,
    hit_at_k,
    judge_answer_relevancy,
    judge_faithfulness,
    mrr,
    precision_at_k,
    recall_at_k,
    split_sentences,
)


# ── 假 LLM 客户端 ─────────────────────────────────────────────────────────────

class FakeResp:
    """模拟 LLM 响应：content 用 dict 块（extract_text_content 双兼容之一）。"""

    def __init__(self, text: str):
        self.content = [{"type": "text", "text": text}]


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)  # 按序弹出；元素可以是 FakeResp 或 Exception
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def json_resp(obj) -> FakeResp:
    return FakeResp(json.dumps(obj, ensure_ascii=False))


# ── 句切分与余弦 ──────────────────────────────────────────────────────────────

def test_split_sentences_chinese_punctuation():
    """中文标点与换行切句，空白句丢弃。"""
    assert split_sentences("不能同服。出血风险增加！真的吗？") == \
        ["不能同服", "出血风险增加", "真的吗"]
    assert split_sentences("第一句\n第二句；第三句") == ["第一句", "第二句", "第三句"]
    assert split_sentences("") == []


def test_cosine_similarity_basics():
    """同向向量=1、正交=0、零向量/长度不一致=0。"""
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 1.0]) == 0.0


# ── Faithfulness ──────────────────────────────────────────────────────────────

def test_faithfulness_scores_supported_ratio():
    """3 句里判 2 句有依据 → score = 2/3。"""
    async def scenario():
        client = FakeClient([json_resp({"supported": [True, False, True]})])
        return await judge_faithfulness(
            "华法林能和银杏叶同服吗",
            "不建议同服。出血风险增加。建议咨询医生。",
            ["银杏叶抑制血小板聚集，与华法林合用出血风险增加"],
            client, "test-model",
        )

    result = asyncio.run(scenario())
    assert result.score == 0.6667
    assert result.supported == 2 and result.total == 3
    assert result.judge_failed is False


def test_faithfulness_prompt_contains_contexts_and_numbered_sentences():
    """判卷 prompt 必须带检索片段与编号句子（判句依据）。"""
    async def scenario():
        client = FakeClient([json_resp({"supported": [True]})])
        await judge_faithfulness("问题", "唯一一句。", ["片段内容甲乙丙"], client, "m")
        return client.messages.calls[0]["messages"][0]["content"]

    prompt = asyncio.run(scenario())
    assert "片段内容甲乙丙" in prompt
    assert "1. 唯一一句" in prompt
    assert "supported" in prompt


def test_faithfulness_retries_on_wrong_length():
    """judge 返回数组长度不对 → 重试，第二次正确长度被采用（共 2 次调用）。

    重试必须把校验错误反馈给 judge（temperature=0 下原样重发必然得到同样
    的错误输出，2026-10-13 全量基线三场数错句数的实测教训）。
    """
    async def scenario():
        client = FakeClient([
            json_resp({"supported": [True]}),           # 长度 1 != 3 → 重试
            json_resp({"supported": [True, True, True]}),
        ])
        result = await judge_faithfulness("q", "一句。两句。三句。", [], client, "m")
        return result, client.messages

    result, messages = asyncio.run(scenario())
    assert result.score == 1.0
    assert len(messages.calls) == 2
    # 第一次原样 prompt，第二次必须带上错误反馈
    assert "上一次的输出没有通过校验" not in messages.calls[0]["messages"][0]["content"]
    retry_prompt = messages.calls[1]["messages"][0]["content"]
    assert "上一次的输出没有通过校验" in retry_prompt
    assert "长度 1" in retry_prompt  # 错误细节要跟着反馈回去


def test_faithfulness_retries_network_error_without_feedback():
    """网络类异常重发原 prompt（错误信息对网络问题没有纠正价值）。"""
    async def scenario():
        client = FakeClient([RuntimeError("boom"), json_resp({"supported": [True]})])
        result = await judge_faithfulness("q", "一句。", [], client, "m")
        return result, client.messages

    result, messages = asyncio.run(scenario())
    assert result.score == 1.0
    assert len(messages.calls) == 2
    assert messages.calls[0]["messages"][0]["content"] == \
        messages.calls[1]["messages"][0]["content"]  # 原样重发
    assert "上一次的输出没有通过校验" not in messages.calls[1]["messages"][0]["content"]


def test_faithfulness_fallback_after_retries():
    """重试 2 次仍失败（抛异常）→ 兜底 0.5 + judge_failed，共 3 次调用。"""
    async def scenario():
        client = FakeClient([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
        result = await judge_faithfulness("q", "一句。", [], client, "m")
        return result, client.messages

    result, messages = asyncio.run(scenario())
    assert result.score == 0.5 and result.judge_failed is True
    assert len(messages.calls) == 3  # 首次 + 2 次重试


def test_faithfulness_empty_answer_scores_zero_without_calling_llm():
    """空回答无可判句，直接 0 分，不调 LLM（省钱也防无意义调用）。"""
    async def scenario():
        client = FakeClient([])
        result = await judge_faithfulness("q", "", ["ctx"], client, "m")
        return result, client.messages

    result, messages = asyncio.run(scenario())
    assert result.score == 0.0 and result.total == 0
    assert messages.calls == []


# ── Answer Relevancy ──────────────────────────────────────────────────────────

def _fixed_embedder(mapping):
    """按文本查表的假向量器：返回预设向量（真实模型不可离线加载）。"""
    def embed(text):
        return mapping.get(text)
    return embed


def test_answer_relevancy_mean_cosine():
    """3 个反推问题的余弦均值：两个 1.0、一个 0.0 → 2/3。"""
    async def scenario():
        client = FakeClient([json_resp({"questions": ["甲", "乙", "丙"]})])
        embed = _fixed_embedder({
            "原始问题": [1.0, 0.0],
            "甲": [1.0, 0.0],   # 同向 → 1.0
            "乙": [1.0, 0.0],   # 同向 → 1.0
            "丙": [0.0, 1.0],   # 正交 → 0.0
        })
        return await judge_answer_relevancy(
            "原始问题", "助手回答内容", client, "m", embedder=embed,
        )

    result = asyncio.run(scenario())
    assert result.score == 0.6667
    assert result.judge_failed is False


def test_answer_relevancy_retries_on_bad_json():
    """反推问题返回坏 JSON → 重试后成功。"""
    async def scenario():
        client = FakeClient([
            FakeResp("这不是JSON"),
            json_resp({"questions": ["甲", "乙", "丙"]}),
        ])
        embed = _fixed_embedder({"原始问题": [1.0, 0.0], "甲": [1.0, 0.0],
                                 "乙": [1.0, 0.0], "丙": [1.0, 0.0]})
        result = await judge_answer_relevancy("原始问题", "回答", client, "m", embedder=embed)
        return result, client.messages

    result, messages = asyncio.run(scenario())
    assert result.score == 1.0
    assert len(messages.calls) == 2


def test_answer_relevancy_retries_on_wrong_length():
    """questions 数组长度不对（结构错）→ 同样走重试，而不是直接判 0 分。"""
    async def scenario():
        client = FakeClient([
            json_resp({"questions": ["只有一个"]}),          # 长度 1 != 3 → 重试
            json_resp({"questions": ["甲", "乙", "丙"]}),
        ])
        embed = _fixed_embedder({"原始问题": [1.0, 0.0], "甲": [1.0, 0.0],
                                 "乙": [1.0, 0.0], "丙": [1.0, 0.0]})
        result = await judge_answer_relevancy("原始问题", "回答", client, "m", embedder=embed)
        return result, client.messages

    result, messages = asyncio.run(scenario())
    assert result.score == 1.0
    assert len(messages.calls) == 2


def test_answer_relevancy_judge_failure_scores_zero():
    """重试耗尽仍失败 → score 0.0 + judge_failed。"""
    async def scenario():
        client = FakeClient([RuntimeError("boom")] * 3)
        result = await judge_answer_relevancy(
            "原始问题", "回答", client, "m",
            embedder=_fixed_embedder({}),
        )
        return result

    result = asyncio.run(scenario())
    assert result.score == 0.0 and result.judge_failed is True


def test_answer_relevancy_embedder_unavailable_returns_none():
    """本地 embedding 模型不可用（embedder 返回 None）→ 指标不可算，返回 None。"""
    async def scenario():
        client = FakeClient([json_resp({"questions": ["甲", "乙", "丙"]})])
        return await judge_answer_relevancy(
            "原始问题", "回答", client, "m",
            embedder=lambda text: None,
        )

    assert asyncio.run(scenario()) is None


# ── 检索质量确定性统计 ────────────────────────────────────────────────────────

def test_precision_recall_f1_at_k():
    """P@3 / R@3 / F1 的标准定义与 k 截断（函数内部统一 round 到 4 位小数）。"""
    retrieved = ["g1", "x1", "g2", "x2"]
    gold = ["g1", "g2", "g3"]
    assert precision_at_k(retrieved, gold, k=3) == 0.6667
    assert recall_at_k(retrieved, gold, k=3) == 0.6667
    assert f1_at_k(0.6667, 0.6667) == 0.6667
    assert precision_at_k([], gold) == 0.0
    assert recall_at_k(retrieved, []) == 0.0
    assert f1_at_k(0.0, 0.0) == 0.0


def test_mrr_rank_of_first_gold_hit():
    """金文档排名越靠前分数越高，没捞到为 0。"""
    assert mrr(["g1", "x"], ["g1"]) == 1.0
    assert mrr(["x", "g1"], ["g1"]) == 0.5
    assert mrr(["x", "y"], ["g1"]) == 0.0


def test_hit_at_k():
    """前 k 内命中任一金标准即为 True。"""
    assert hit_at_k(["x", "g1", "y"], ["g1"], k=2) is True
    assert hit_at_k(["x", "y", "g1"], ["g1"], k=2) is False
