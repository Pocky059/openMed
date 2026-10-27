"""
TrajectoryLogger 单测（v2 阶段 0.1，不联网）

覆盖点：
  1. 落盘格式 —— record() 后 JSONL 每行一条完整轨迹，文件名按天切分
  2. 隐私 —— user_id 只存 SHA256 前 16 位，原文不出现在文件中
  3. 开关 —— enabled=False 时不落盘、不崩溃
  4. kind 分类 —— classify_messages 纯函数对消息流补标记
  5. 编排器组装 —— _log_trajectory 产出 meta + messages + steps 三视图
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    OrchestratorResult,
    Request,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.trajectory_logger import TrajectoryLogger, classify_messages, hash_user_id


def _read_lines(tmp_path: Path):
    """读测试目录下唯一一个轨迹文件，返回逐行解析后的记录列表。"""
    files = list(tmp_path.glob("trajectories-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_logger_writes_one_jsonl_line_per_record(tmp_path):
    logger = TrajectoryLogger(log_dir=str(tmp_path), enabled=True)
    logger.record({"request_id": "r1", "n": 1})
    logger.record({"request_id": "r2", "n": 2})
    logger.close()

    records = _read_lines(tmp_path)
    assert [r["request_id"] for r in records] == ["r1", "r2"]
    today = datetime.now().strftime("%Y-%m-%d")
    assert (tmp_path / f"trajectories-{today}.jsonl").exists()


def test_logger_disabled_writes_nothing(tmp_path):
    logger = TrajectoryLogger(log_dir=str(tmp_path), enabled=False)
    logger.record({"request_id": "r1"})
    logger.close()

    assert not list(tmp_path.glob("trajectories-*.jsonl"))


def test_hash_user_id_is_deterministic_and_hides_raw_id():
    hashed = hash_user_id("patient-123")
    assert hashed == hash_user_id("patient-123")
    assert len(hashed) == 16
    assert "patient-123" not in hashed


def test_classify_messages_tags_kinds_by_structure():
    messages = [
        {"role": "user", "content": "[背景信息]\n患者有过敏史"},
        {"role": "assistant", "content": "好的，我已了解背景信息。"},
        {"role": "user", "content": "在吃华法林，能喝银杏叶茶吗？"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "search_knowledge_base", "input": {"query": "银杏叶 华法林"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "不建议同服。"}]},
    ]

    classified = classify_messages(messages, "medication")

    kinds = [entry["kind"] for entry in classified]
    assert kinds == ["context", "ack", "query", "tool_call", "tool_result", "final"]
    assert all(entry["agent_type"] == "medication" for entry in classified)


def test_classify_messages_recognizes_sdk_block_objects():
    """生产默认协议（anthropic）下消息块是 SDK 对象而非 dict——只认 dict 会把
    工具调用误标成 final（审查 H2 修复的回归测试）。"""

    class _FakeBlock:  # 模拟 anthropic SDK 的 ToolUseBlock（pydantic 对象形态）
        def __init__(self, type_, **fields):
            self.type = type_
            for key, value in fields.items():
                setattr(self, key, value)

    messages = [
        {"role": "user", "content": "在吃华法林，能喝银杏叶茶吗？"},
        {"role": "assistant", "content": [
            _FakeBlock("tool_use", id="toolu_1", name="search_knowledge_base", input={"query": "银杏叶 华法林"})
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]},
        {"role": "assistant", "content": "不建议同服。"},
    ]

    classified = classify_messages(messages, "medication")

    kinds = [entry["kind"] for entry in classified]
    assert kinds == ["query", "tool_call", "tool_result", "final"]


def test_orchestrator_logs_full_trajectory_with_three_views(tmp_path):
    logger = TrajectoryLogger(log_dir=str(tmp_path), enabled=True)
    orchestrator = AgentOrchestrator(api_key="test-key", trajectory_logger=logger)
    try:
        req = Request(
            message="在吃华法林，能喝银杏叶茶吗？",
            user_id="patient-123",
            conv_id="c1",
            intent=IntentCategory.MEDICATION_INTERACTION,
            urgency=UrgencyLevel.LOW,
            intent_confidence=0.9,
            history=[{"role": "assistant", "content": "请描述您正在服用的药物。"}],
            context="患者画像：青霉素过敏",
            intent_detail={
                "input": {"message": "在吃华法林，能喝银杏叶茶吗？", "history": []},
                "output": {"intent": "medication_interaction", "confidence": 0.9},
            },
        )
        system_prompt = "你是用药咨询助手。[角色契约]\n角色：用药咨询\n[动态 Skills]\n- 用药相互作用核查"
        response = AgentResponse(
            agent_type=AgentType.MEDICATION,
            content="不建议同服，出血风险增加。",
            success=True,
            system_prompt=system_prompt,
            messages=[
                {"role": "user", "content": "在吃华法林，能喝银杏叶茶吗？"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "search_knowledge_base", "input": {"query": "银杏叶 华法林 相互作用"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "不建议同服，出血风险增加。"}]},
            ],
            steps=[{
                "step_id": 1,
                "agent_type": "medication",
                "llm_latency_ms": 100.0,
                "tool_calls": [{"tool_name": "search_knowledge_base", "args": {"query": "银杏叶 华法林 相互作用"}, "tool_use_id": "toolu_1"}],
                "tool_results": [{"tool_name": "search_knowledge_base", "success": True, "cached": False, "reranked": True, "latency_ms": 50.0}],
            }],
        )
        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=AgentType.MEDICATION,
            intent=req.intent,
            agent_types=[AgentType.MEDICATION],
            primary_agent=AgentType.MEDICATION,
            routing_reason="intent=medication_interaction",
            routing_confidence=0.9,
        )
        orchestrator._log_trajectory(req, result, [response])
    finally:
        orchestrator.close_trajectory_logger()

    record = _read_lines(tmp_path)[0]
    assert record["request_id"] == req.request_id
    assert record["meta"]["user_id_hash"] == hash_user_id("patient-123")
    assert "patient-123" not in json.dumps(record, ensure_ascii=False)
    assert record["meta"]["phase"] == "normal"
    kinds = [m["kind"] for m in record["messages"]]
    assert kinds == ["system", "query", "tool_call", "tool_result", "final"]
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][0]["agent_type"] == "medication"
    assert record["messages"][0]["content"] == system_prompt
    # 编排器输入快照：message 之外的 context/entities/history 也落盘了
    assert record["meta"]["input"]["message"] == "在吃华法林，能喝银杏叶茶吗？"
    assert record["meta"]["input"]["context"] == "患者画像：青霉素过敏"
    assert record["meta"]["input"]["history"][0]["role"] == "assistant"
    # 意图识别完整快照原样透传
    assert record["meta"]["intent_detail"]["output"]["intent"] == "medication_interaction"
    assert record["steps"][0]["tool_results"][0]["reranked"] is True


def test_build_trajectory_messages_prepends_system_prompt_only_when_present():
    response = AgentResponse(
        agent_type=AgentType.MEDICATION,
        content="不建议同服。",
        success=True,
        system_prompt="你是用药咨询助手。",
        messages=[{"role": "user", "content": "华法林能和银杏叶同服吗？"}],
    )
    messages = AgentOrchestrator._build_trajectory_messages([response])
    assert messages[0] == {
        "role": "system",
        "agent_type": "medication",
        "content": "你是用药咨询助手。",
        "kind": "system",
    }
    assert [m["kind"] for m in messages] == ["system", "query"]

    # 没拼 system prompt 的响应（如异常早退路径）不产生 system 消息
    bare = AgentResponse(agent_type=AgentType.MEDICATION, content="x", success=True, messages=[])
    assert AgentOrchestrator._build_trajectory_messages([bare]) == []


def test_run_recognition_branch_snapshots_intent_detail(tmp_path):
    """编排器兜底识别意图（调用方没预设 intent）时，完整输入/输出快照进
    meta.intent_detail——澄清路径也落盘，不丢。"""
    from core.intent_recognizer import IntentResult

    class FakeRecognizer:
        async def recognize(self, message, history=None):
            return IntentResult(
                intent=IntentCategory.OTHER,
                confidence=0.3,
                urgency=UrgencyLevel.LOW,
                intent_group="other",
                entities={},
                reasoning="三路投票置信度均低，判为 other",
                latency_ms=12.3,
                source_scores={"llm": 0.2, "embedding": 0.3, "pattern": 0.1},
            )

    logger = TrajectoryLogger(log_dir=str(tmp_path), enabled=True)
    orchestrator = AgentOrchestrator(api_key="test-key", trajectory_logger=logger)
    orchestrator._intent_recognizer = FakeRecognizer()
    try:
        result = asyncio.run(orchestrator.run(Request(
            message="今天天气怎么样",
            user_id="u1",
            conv_id="c1",
            history=[{"role": "assistant", "content": "请问有什么可以帮您？"}],
        )))
        assert result.intent == IntentCategory.OTHER
    finally:
        orchestrator.close_trajectory_logger()

    record = _read_lines(tmp_path)[0]
    assert record["meta"]["phase"] == "clarification"
    detail = record["meta"]["intent_detail"]
    assert detail["input"]["message"] == "今天天气怎么样"
    assert detail["input"]["history"] == [{"role": "assistant", "content": "请问有什么可以帮您？"}]
    assert detail["output"]["intent"] == "other"
    assert detail["output"]["reasoning"] == "三路投票置信度均低，判为 other"
    assert detail["output"]["source_scores"]["llm"] == 0.2


def test_find_by_request_id_returns_matching_record(tmp_path):
    logger = TrajectoryLogger(log_dir=str(tmp_path), enabled=True)
    logger.record({"request_id": "r1", "n": 1})
    logger.record({"request_id": "r2", "n": 2})
    logger.close()

    assert logger.find_by_request_id("r2")["n"] == 2
    assert logger.find_by_request_id("r1")["n"] == 1
    assert logger.find_by_request_id("missing") is None


def test_find_by_request_id_disabled_or_empty_dir_returns_none(tmp_path):
    disabled = TrajectoryLogger(log_dir=str(tmp_path), enabled=False)
    assert disabled.find_by_request_id("r1") is None
    disabled.close()

    # 目录从没写过轨迹文件时同样返回 None，不抛异常
    empty = TrajectoryLogger(log_dir=str(tmp_path / "never-written"), enabled=True)
    assert empty.find_by_request_id("r1") is None
    empty.close()
