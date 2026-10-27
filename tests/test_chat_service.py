"""
api/chat_service.py 的离线单测（v2 阶段 1 第 1 步重构配套）。

用假编排器/假记忆锁死五步主链路的调用顺序与响应映射——这是
/api/chat「纯重构行为不变」验收的代码级保障，也是将来基线评测
CLI 复用 ChatService 时的防漂移锚点。
"""
import asyncio
from types import SimpleNamespace

import pytest

from api.chat_service import ChatResponse, ChatService
from memory.conversation_memory import MsgRole


class FakeMemory:
    """记录调用顺序的假记忆管理器（get_context/add_message/update_profile 三件套）。"""

    def __init__(self, recent_messages):
        self._recent = recent_messages
        self.calls = []          # 按调用先后追加元组，测试断言顺序用

    async def get_context(self, user_id, conv_id, query=""):
        self.calls.append(("get_context", user_id, conv_id, query))
        return SimpleNamespace(   # 只用得到 recent_messages 和 to_prompt_text，其余字段省略
            recent_messages=self._recent,
            to_prompt_text=lambda: "综合上下文",
        )

    async def add_message(self, user_id, conv_id, role, content):
        self.calls.append(("add_message", role, content))

    async def update_profile(self, user_id, conv_id):
        self.calls.append(("update_profile", user_id, conv_id))


class FakeOrchestrator:
    """返回固定结果的假编排器（recognize_intent/run 两个入口）。

    枚举用 SimpleNamespace(value=...) 模拟——只用到 .value，与真实
    IntentCategory/AgentType 的接口一致。
    """

    def __init__(self, intent=None):
        self._intent = intent  # None 时测「intent 缺失回退 other」分支
        self.calls = []

    async def recognize_intent(self, message, history=None):
        self.calls.append(("recognize_intent", message, history))
        result = SimpleNamespace(
            entities={"drug": ["布洛芬"]},
            intent=self._intent,
            intent_group="medication",
            urgency=SimpleNamespace(name="LOW", value=1),  # 模拟真实 UrgencyLevel 枚举（int 值 + 字符串名）
            confidence=0.8765,
            source_scores={"emb": 0.9, "llm": 0.8},
            reasoning="三路投票融合",
            latency_ms=10.0,
        )
        # 模拟真实 IntentResult.to_trajectory_detail：chat_service 用它做轨迹快照
        result.to_trajectory_detail = (
            lambda msg, hist=None: {
                "input": {"message": msg, "history": hist or []},
                "output": {"intent": getattr(result.intent, "value", None)},
            }
        )
        return result

    async def run(self, req):
        self.calls.append(("run", req))
        return SimpleNamespace(
            request_id="req-123",
            response="用药建议……",
            agent_type=SimpleNamespace(value="medication"),
            intent=self._intent,
            agent_types=[SimpleNamespace(value="medication")],
            primary_agent=None,              # 没有 primary 时响应应回退 agent_type
            supporting_agents=[],
            tools_used=["search_knowledge_base"],
            routing_reason="领域打分最高",
            routing_confidence=0.9,
            escalated=False,
            latency_ms=1234.56,
        )


def test_five_step_order_and_response_mapping():
    """五步顺序（记忆读→意图→编排→记忆写×2→画像更新）+ 响应字段映射。"""
    async def scenario():
        msgs = [SimpleNamespace(role=MsgRole.USER, content="我有点头疼")]
        mem = FakeMemory(recent_messages=msgs)
        orch = FakeOrchestrator(intent=SimpleNamespace(value="medication_advice"))
        svc = ChatService(orch, mem)

        resp = await svc.handle_chat_turn("布洛芬怎么吃", "u1", "c1")
        await asyncio.sleep(0)   # 让画像更新（create_task 异步任务）跑完再断言
        return mem, orch, resp

    # 项目零新依赖、不装 pytest-asyncio，异步代码统一 asyncio.run 包装（同 test_agent_orchestrator.py 惯例）
    mem, orch, resp = asyncio.run(scenario())

    # 五步顺序：记忆读 → 意图 → 编排 → 写用户消息 → 写助手回答 → 画像更新
    assert [c[0] for c in mem.calls] == ["get_context", "add_message", "add_message", "update_profile"]
    assert [c[0] for c in orch.calls] == ["recognize_intent", "run"]
    # 编排器收到的请求体带意图识别完整快照（轨迹 meta.intent_detail 的数据来源）
    orch_req = orch.calls[1][1]
    assert orch_req.intent_detail["input"]["message"] == "布洛芬怎么吃"
    assert orch_req.intent_detail["output"]["intent"] == "medication_advice"
    assert mem.calls[1][1] is MsgRole.USER          # 先写用户提问
    assert mem.calls[1][2] == "布洛芬怎么吃"
    assert mem.calls[2][1] is MsgRole.ASSISTANT     # 再写模型回答
    assert mem.calls[2][2] == "用药建议……"
    assert mem.calls[3][1:] == ("u1", "c1")         # 画像更新用同一对 user/conv

    # 响应字段映射
    assert isinstance(resp, ChatResponse)
    assert resp.conv_id == "c1"
    assert resp.request_id == "req-123"
    assert resp.response == "用药建议……"
    assert resp.intent == "medication_advice"
    assert resp.intent_group == "medication"
    assert resp.agent_type == "medication"
    assert resp.primary_agent == "medication"        # primary 为 None 回退 agent_type
    assert resp.agent_types == ["medication"]
    assert resp.supporting_agents == []
    assert resp.tools_used == ["search_knowledge_base"]
    assert resp.knowledge_used is True               # 由 tools_used 推导
    assert resp.routing_reason == "领域打分最高"
    assert resp.routing_confidence == 0.9
    assert resp.escalated is False
    assert resp.urgency == "LOW"                     # 急症集判分对比 expected_urgency 用
    assert resp.latency_ms == 1234.6                 # round(..., 1)
    assert resp.entities == {"drug": ["布洛芬"]}
    assert resp.intent_confidence == 0.8765          # round(..., 4)
    assert resp.intent_source_scores == {"emb": 0.9, "llm": 0.8}


def test_conv_id_generated_when_missing():
    """不传 conv_id 时生成 uuid4 格式的会话 id。"""
    svc = ChatService(FakeOrchestrator(), FakeMemory(recent_messages=[]))
    resp = asyncio.run(svc.handle_chat_turn("你好", "u1"))
    assert len(resp.conv_id) == 36                    # uuid4 字符串长度


def test_history_capped_at_last_five():
    """意图识别只拿到最近 5 条工作记忆（与旧 /chat 端点一致）。"""
    async def scenario():
        msgs = [
            SimpleNamespace(role=MsgRole.ASSISTANT if i % 2 == 0 else MsgRole.USER, content=f"msg{i}")
            for i in range(7)                          # 7 条 → 只取最后 5 条（msg2..msg6）
        ]
        mem = FakeMemory(recent_messages=msgs)
        orch = FakeOrchestrator()
        svc = ChatService(orch, mem)
        await svc.handle_chat_turn("继续", "u1", "c1")
        return orch

    orch = asyncio.run(scenario())

    history = orch.calls[0][2]                         # recognize_intent 的 history 参数
    assert history is not None and len(history) == 5
    assert history[0] == {"role": "assistant", "content": "msg2"}
    assert history[-1] == {"role": "assistant", "content": "msg6"}  # i=6 为偶数 → ASSISTANT


def test_intent_none_falls_back_to_other():
    """意图识别返回 None 时响应回退 "other"（与旧 /chat 端点一致）。"""
    svc = ChatService(FakeOrchestrator(intent=None), FakeMemory(recent_messages=[]))
    resp = asyncio.run(svc.handle_chat_turn("随便聊聊", "u1", "c1"))
    assert resp.intent == "other"
