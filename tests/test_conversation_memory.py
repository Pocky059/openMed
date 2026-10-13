"""四字段结构化摘要的纯函数单测（parse_summary / merge_field_summaries）。

只测确定性逻辑，不测 LLM 路径（离线纪律，conftest 设 HF_HUB_OFFLINE=1）。
背景：压缩摘要从「2-3 句话自由文本」改为四字段 JSON（allergies/medications/
symptoms/history），字段与长会话考卷 gold_items.type 一一对应——长会话判分
直接读 summary key 的 JSON 逐字段查关键病史是否保留。
"""
import json

import pytest

from memory.conversation_memory import SUMMARY_FIELDS, merge_field_summaries, parse_summary


def _summary(**fields) -> str:
    """按四字段构造摘要 JSON，未给字段填空数组。"""
    data = {f: fields.get(f, []) for f in SUMMARY_FIELDS}
    return json.dumps(data, ensure_ascii=False)


class TestParseSummary:
    def test_valid_four_fields(self):
        parsed = parse_summary(_summary(allergies=["青霉素过敏"], medications=["华法林 3mg/天"]))
        assert parsed == {
            "allergies": ["青霉素过敏"],
            "medications": ["华法林 3mg/天"],
            "symptoms": [],
            "history": [],
        }

    def test_extra_fields_ignored(self):
        """多余字段不拦（宽松处理），只要求四字段齐全合法。"""
        raw = json.dumps({"allergies": [], "medications": [], "symptoms": [],
                          "history": [], "preferences": ["喜欢安静"]}, ensure_ascii=False)
        assert parse_summary(raw) is not None

    @pytest.mark.parametrize("bad", [
        "这不是 JSON 的自由文本摘要",           # 旧版自由文本
        "",                                      # 空串
        "[]",                                    # JSON 但不是 dict
        _summary().replace('"history"', '"hist"'),  # 缺一个字段
        json.dumps({"allergies": "青霉素", "medications": [], "symptoms": [], "history": []},
                   ensure_ascii=False),           # 字段不是 list
        json.dumps({"allergies": [1], "medications": [], "symptoms": [], "history": []},
                   ensure_ascii=False),           # 元素不是 str
    ])
    def test_invalid_returns_none(self, bad):
        """非法输入一律 None——调用方据此回退 LLM 合并，不吞半个 JSON。"""
        assert parse_summary(bad) is None


class TestMergeFieldSummaries:
    def test_field_wise_merge_dedup_keep_order(self):
        old = _summary(allergies=["青霉素过敏"], medications=["华法林 3mg/天"])
        new = _summary(medications=["华法林 3mg/天", "银杏叶提取物（已停）"],
                       symptoms=["胸闷两周"])
        merged = merge_field_summaries(old, new)
        assert json.loads(merged) == {
            "allergies": ["青霉素过敏"],
            "medications": ["华法林 3mg/天", "银杏叶提取物（已停）"],  # 去重 + 保序
            "symptoms": ["胸闷两周"],
            "history": [],
        }

    def test_empty_fields(self):
        old = _summary()
        new = _summary(allergies=["花粉过敏"])
        assert json.loads(merge_field_summaries(old, new))["allergies"] == ["花粉过敏"]

    @pytest.mark.parametrize("old,new", [
        ("旧版自由文本", _summary()),
        (_summary(), "旧版自由文本"),
    ])
    def test_legacy_text_returns_none(self, old, new):
        """任一段是旧版自由文本 → 返回 None，由 _merge_summary 回退 LLM 路径。"""
        assert merge_field_summaries(old, new) is None


class FakeRedis:
    """极简假 Redis：只实现 get（get_summary 唯一用到的操作）。"""

    def __init__(self, store=None):
        self.store = dict(store or {})

    async def get(self, key):
        return self.store.get(key)


def make_memory(fake_redis):
    """用 __new__ 绕过 MemoryManager 构造器（构造器要连真实 Redis/ChromaDB），
    只注入 get_summary 依赖的 _redis 属性。"""
    from memory.conversation_memory import MemoryManager
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._redis = fake_redis
    return mgr


class TestGetSummary:
    def test_returns_stored_summary(self):
        """有摘要时原样返回；key 格式 summary:{user_id}:{conv_id}。"""
        fake = FakeRedis({"summary:u1:c1": '{"allergies":["青霉素过敏"],"medications":[],"symptoms":[],"history":[]}'})
        mgr = make_memory(fake)
        import asyncio
        got = asyncio.run(mgr.get_summary("u1", "c1"))
        assert "青霉素过敏" in got

    def test_empty_when_not_compressed(self):
        """没压缩过（key 不存在）返回空串——判分器据此全 miss 走 judge 复核。"""
        import asyncio
        assert asyncio.run(make_memory(FakeRedis()).get_summary("u1", "c1")) == ""