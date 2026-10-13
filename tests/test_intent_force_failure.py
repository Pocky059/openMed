"""
core/intent_recognizer.py 强制 LLM 失败开关的离线单测（v2 阶段 1 第 6 步配套）。

锁定四件事：
  1. 显式构造参数优先于 env（env 全开 + 参数 False 也不生效——评测 CLI
     可以精确控制单个实例，不受宿主机环境变量干扰）
  2. env 双门控：OPENMED_EVAL_MODE 与 OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE
     必须同时 =1 才生效（生产环境不设这两个变量，恒 False）
  3. 短路结果与真实失败分支同构（键集合/取值语义一致，reasoning 可区分）
  4. 短路零 LLM 调用（强制 LLM 失败子集测降级链路，白花 API 调用为浪费）
"""
import asyncio

import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer


# ── 假客户端 ──────────────────────────────────────────────────────────────────

class FakeMessages:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        # forced-fail 下不该被调到；万一被调到，返回一个合法响应防测试误导
        from types import SimpleNamespace
        return SimpleNamespace(content=[{"type": "text", "text":
            '{"intent": "query", "confidence": 0.9, "reasoning": "不该被调用"}'}])


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


@pytest.fixture
def fake_client(monkeypatch):
    """把 create_llm_client 换成假实现，构造 IntentRecognizer 不联网。"""
    client = FakeClient()
    monkeypatch.setattr("core.intent_recognizer.create_llm_client",
                        lambda api_key, base_url: client)
    return client


def make_recognizer(fake_client, force=None):
    return IntentRecognizer(api_key="test", model="test-model",
                            force_llm_failure=force)


# ── 显式参数优先级 ────────────────────────────────────────────────────────────

def test_explicit_true_short_circuits_llm(fake_client):
    """参数 True：_llm_recognize 短路返回 failed 降级值，零 LLM 调用。"""
    rec = make_recognizer(fake_client, force=True)
    result = asyncio.run(rec._llm_recognize("感冒了怎么办", None))
    assert result["intent"] == IntentCategory.OTHER
    assert result["confidence"] == 0.0
    assert result["failed"] is True
    assert "评测强制" in result["reasoning"]
    assert fake_client.messages.calls == []


def test_explicit_false_overrides_env(fake_client, monkeypatch):
    """参数 False 优先于 env：双门全开也不生效。"""
    monkeypatch.setenv("OPENMED_EVAL_MODE", "1")
    monkeypatch.setenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", "1")
    rec = make_recognizer(fake_client, force=False)
    assert rec.force_llm_failure is False
    result = asyncio.run(rec._llm_recognize("感冒了怎么办", None))
    assert result.get("failed") is not True
    assert len(fake_client.messages.calls) == 1  # 正常走了 LLM


def test_explicit_true_overrides_env_off(fake_client, monkeypatch):
    """参数 True 优先于 env：env 全关也生效（CLI 显式注入场景）。"""
    monkeypatch.delenv("OPENMED_EVAL_MODE", raising=False)
    monkeypatch.delenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", raising=False)
    rec = make_recognizer(fake_client, force=True)
    assert rec.force_llm_failure is True


# ── env 双门控 ────────────────────────────────────────────────────────────────

def test_env_single_gate_not_enough(fake_client, monkeypatch):
    """只开一个门不生效（双门控的意义：评测开关之外还要显式要失败）。"""
    monkeypatch.setenv("OPENMED_EVAL_MODE", "1")
    monkeypatch.delenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", raising=False)
    assert make_recognizer(fake_client).force_llm_failure is False

    monkeypatch.delenv("OPENMED_EVAL_MODE", raising=False)
    monkeypatch.setenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", "1")
    assert make_recognizer(fake_client).force_llm_failure is False


def test_env_double_gate_enables(fake_client, monkeypatch):
    """双门全开：构造时解析 env，生效。"""
    monkeypatch.setenv("OPENMED_EVAL_MODE", "1")
    monkeypatch.setenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", "1")
    assert make_recognizer(fake_client).force_llm_failure is True


def test_env_snapshot_at_construction(fake_client, monkeypatch):
    """env 在构造时快照：构造后改 env 不影响已建实例（行为可预测）。"""
    monkeypatch.setenv("OPENMED_EVAL_MODE", "1")
    monkeypatch.setenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", "1")
    rec = make_recognizer(fake_client)
    monkeypatch.delenv("OPENMED_EVAL_MODE", raising=False)
    monkeypatch.delenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", raising=False)
    assert rec.force_llm_failure is True


def test_default_off_without_env(fake_client, monkeypatch):
    """两个 env 都没设（生产常态）：恒 False。"""
    monkeypatch.delenv("OPENMED_EVAL_MODE", raising=False)
    monkeypatch.delenv("OPENMED_EVAL_FORCE_LLM_INTENT_FAILURE", raising=False)
    assert make_recognizer(fake_client).force_llm_failure is False


# ── 短路与真实失败同构 ────────────────────────────────────────────────────────

def test_forced_result_same_shape_as_real_failure(fake_client):
    """短路结果的键集合与真实失败分支一致（_vote 依赖 failed 键走降级）。"""
    rec = make_recognizer(fake_client, force=True)
    forced = asyncio.run(rec._llm_recognize("x", None))
    assert set(forced) == {"intent", "confidence", "reasoning", "failed"}


def test_recognize_full_chain_uses_fallback_vote(fake_client):
    """完整 recognize()：forced 时 LLM 路 failed → 投票走 emb/pat 降级分支。

    结果来源只能是 embedding 路或关键词路（或 OTHER），不可能是 LLM 的
    判断；reasoning 透传 LLM 路的「评测强制」标注；LLM 零调用。
    """
    rec = make_recognizer(fake_client, force=True)
    result = asyncio.run(rec.recognize("胸口剧烈疼痛喘不上气"))
    assert result.reasoning == "LLM 失败（评测强制）"
    assert fake_client.messages.calls == []
    # 降级投票：LLM 失败后取 emb 或 pat（该消息含「胸痛/喘不上气」关键词，
    # 两条降级路都会给出 emergency）；confidence 来自降级路得分而非 LLM
    assert result.source_scores["llm"] == 0.0
    assert result.intent in (IntentCategory.EMERGENCY, IntentCategory.OTHER)


def test_forced_recognize_no_cache_side_effect(fake_client):
    """缓存正常生效（同消息二次命中），forced 结果不污染其他实例。"""
    rec = make_recognizer(fake_client, force=True)
    first = asyncio.run(rec.recognize("肚子疼"))
    second = asyncio.run(rec.recognize("肚子疼"))
    assert first.intent == second.intent
    assert rec.cache_hits == 1  # 第二次走缓存
    # 另一个未 forced 的实例不受影响
    normal = make_recognizer(fake_client)
    assert normal.force_llm_failure is False
