"""四字段结构化摘要的纯函数单测（parse_summary / merge_field_summaries）。

只测确定性逻辑，不测 LLM 路径（离线纪律，conftest 设 HF_HUB_OFFLINE=1）。
背景：压缩摘要从「2-3 句话自由文本」改为四字段 JSON（allergies/medications/
symptoms/history），字段与长会话考卷 gold_items.type 一一对应——长会话判分
直接读 summary key 的 JSON 逐字段查关键病史是否保留。
"""
import json

import pytest

from memory.conversation_memory import SUMMARY_FIELDS, merge_field_summaries, parse_summary, truncate_summary


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


class TestTruncateSummary:
    """按安全优先级截断（2026-10-26：原硬截 merged[:MAX] 不分字段，
    安全关键字段可能被 history 挤出）。"""

    def test_under_limit_unchanged(self):
        """不超长时原样返回（正常路径零变化）。"""
        text = _summary(allergies=["青霉素过敏"], medications=["华法林 3mg/天"])
        assert truncate_summary(text, 1600) == text

    def test_drops_lowest_priority_field_first(self):
        """超长时先丢 history；allergies/medications 安全关键字段完整。"""
        long_history = [f"就诊记录{i:03d}：因某病就诊，检查结果正常，医嘱定期随访复查" for i in range(60)]
        text = _summary(allergies=["青霉素过敏"], medications=["华法林 3mg/天"],
                        history=long_history)
        out = truncate_summary(text, 300)
        assert len(out) <= 300
        parsed = json.loads(out)
        assert parsed["allergies"] == ["青霉素过敏"]
        assert parsed["medications"] == ["华法林 3mg/天"]
        assert parsed["symptoms"] == []
        assert len(parsed["history"]) < 60          # 牺牲的是 history
        assert parsed["history"][0] == "就诊记录000：因某病就诊，检查结果正常，医嘱定期随访复查"  # 字段内最旧条目保住了

    def test_field_order_sacrifice_history_then_symptoms(self):
        """history 丢光还超长时，接着牺牲 symptoms。"""
        text = _summary(
            allergies=["青霉素过敏"],
            medications=["华法林 3mg/天"],
            symptoms=[f"症状记录{i:02d}：胸闷、气短、乏力" for i in range(30)],
            history=[f"就诊记录{i:02d}：因某病就诊" for i in range(30)],
        )
        out = truncate_summary(text, 200)
        assert len(out) <= 200
        parsed = json.loads(out)
        assert parsed["allergies"] == ["青霉素过敏"]
        assert parsed["medications"] == ["华法林 3mg/天"]
        assert parsed["history"] == []              # 最低优先级先丢光

    def test_legacy_text_hard_truncate(self):
        """自由文本不处理：硬截保头部（与旧行为一致）。"""
        text = "旧版自由文本摘要" * 200
        out = truncate_summary(text, 50)
        assert len(out) <= 50
        assert out == text[:50]


class TestExtractDocs:
    """情景记忆返回带时间戳（2026-10-26：ts 是时间维度，医疗场景不能丢）。"""

    @staticmethod
    def _extract(results):
        from memory.conversation_memory import MemoryManager
        return MemoryManager._extract_docs(results)

    def test_ts_prefix(self):
        """metadata 的 ts 拼成「[日期 的对话]」前缀，ISO 取日期部分。"""
        docs = self._extract({
            "documents": [["对青霉素过敏", "正在服用华法林"]],
            "metadatas": [[{"ts": "2026-10-15T14:30:00", "user_id": "u1"},
                           {"ts": "2026-09-01T09:00:00", "user_id": "u1"}]],
        })
        assert docs == ["[2026-10-15 的对话] 对青霉素过敏", "[2026-09-01 的对话] 正在服用华法林"]

    def test_missing_metadata_degrades_to_plain_text(self):
        """metadatas 缺失/ts 缺失时降级为不带前缀，不抛。"""
        assert self._extract({"documents": [["无元数据片段"]]}) == ["无元数据片段"]
        assert self._extract({
            "documents": [["片段"]],
            "metadatas": [[{"user_id": "u1"}]],  # 没有 ts
        }) == ["片段"]

    def test_single_layer_documents(self):
        """documents 非双层嵌套（单层列表）时兼容。"""
        assert self._extract({
            "documents": ["单层片段"],
            "metadatas": [{"ts": "2026-10-15T14:30:00"}],
        }) == ["[2026-10-15 的对话] 单层片段"]


class BrokenRedis:
    """所有操作都抛的假 Redis——模拟 Redis 服务不可用。"""

    async def get(self, *args, **kwargs):
        raise RuntimeError("redis down")

    async def lpush(self, *args, **kwargs):
        raise RuntimeError("redis down")

    async def expire(self, *args, **kwargs):
        raise RuntimeError("redis down")

    async def llen(self, *args, **kwargs):
        raise RuntimeError("redis down")

    async def lrange(self, *args, **kwargs):
        raise RuntimeError("redis down")


def make_memory_broken_redis():
    """构造只有坏 Redis 的 MemoryManager（ChromaDB 侧另注入假 collection）。"""
    from memory.conversation_memory import MemoryManager
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._redis = BrokenRedis()
    return mgr


class FakeEmptyCollection:
    """返回空结果的假 ChromaDB collection（query/get 都空）。"""

    def query(self, *args, **kwargs):
        return {"documents": [], "metadatas": []}

    def get(self, *args, **kwargs):
        return {"documents": []}


class TestRedisDegradation:
    """Redis 读写降级（2026-10-26 补齐）：Redis 挂了 /chat 不 500，
    读降级为空、写失败不阻塞。ChromaDB 侧一直有 try/except，Redis 侧裸奔。"""

    def test_working_memory_read_degrades_to_empty(self):
        """读失败降级为空列表——本轮按无记忆处理，主链路照走。"""
        import asyncio
        assert asyncio.run(make_memory_broken_redis()._get_working_memory("u1", "c1")) == []

    def test_add_message_write_failure_not_raised(self):
        """写失败不抛：主链路不因 Redis 故障 500。"""
        import asyncio
        from memory.conversation_memory import MsgRole
        asyncio.run(make_memory_broken_redis().add_message("u1", "c1", MsgRole.USER, "你好"))

    def test_get_context_degrades_without_redis(self):
        """get_context 全链路降级：工作记忆/摘要为空，情景记忆/画像走 ChromaDB 假件。"""
        import asyncio
        mgr = make_memory_broken_redis()
        mgr._episodic = FakeEmptyCollection()
        mgr._profile = FakeEmptyCollection()
        ctx = asyncio.run(mgr.get_context("u1", "c1", query="你好"))
        assert ctx.recent_messages == []
        assert ctx.summary == ""
        assert ctx.relevant_history == []
        assert ctx.user_profile == {}


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


class FakeDeletableRedis:
    """记录 delete 收到哪些 key 的假 Redis。"""

    def __init__(self):
        self.deleted = []

    async def delete(self, *keys):
        self.deleted.extend(keys)


class FakeCollection:
    """记录 delete 过滤条件的假 ChromaDB collection。"""

    def __init__(self):
        self.deletes = []

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


class TestForgetUser:
    def test_clears_all_three_levels(self):
        """三级记忆都清：Redis 工作记忆+摘要按 (user, conv) 删，情景记忆+画像按 user_id 删。"""
        import asyncio
        from memory.conversation_memory import MemoryManager
        mgr = MemoryManager.__new__(MemoryManager)
        mgr._redis = FakeDeletableRedis()
        mgr._episodic = FakeCollection()
        mgr._profile = FakeCollection()
        asyncio.run(mgr.forget_user("eval-LS-001", "eval-LS-001"))
        assert mgr._redis.deleted == ["wm:eval-LS-001:eval-LS-001", "summary:eval-LS-001:eval-LS-001"]
        assert mgr._episodic.deletes == [{"where": {"user_id": "eval-LS-001"}}]
        assert mgr._profile.deletes == [{"where": {"user_id": "eval-LS-001"}}]

    def test_failure_is_not_swallowed(self):
        """删除失败直接抛（评测要确知记忆已清干净，静默失败等于带着脏记忆继续跑）。"""
        import asyncio
        from memory.conversation_memory import MemoryManager

        class BrokenCollection:
            def delete(self, **kwargs):
                raise RuntimeError("chromadb down")

        mgr = MemoryManager.__new__(MemoryManager)
        mgr._redis = FakeDeletableRedis()
        mgr._episodic = BrokenCollection()
        mgr._profile = FakeCollection()
        with pytest.raises(RuntimeError, match="chromadb down"):
            asyncio.run(mgr.forget_user("eval-LS-001", "eval-LS-001"))