"""
TrajectoryLogger 单测（v2 阶段 0.1，不联网）

覆盖点：
  1. 落盘格式 —— record() 后 JSONL 每行一条完整轨迹，文件名按天切分
  2. 隐私 —— user_id 只存 SHA256 前 16 位，原文不出现在文件中
  3. 开关 —— enabled=False 时不落盘、不崩溃
  4. kind 分类 —— classify_messages 纯函数对消息流补标记
  5. 编排器组装 —— _log_trajectory 产出 meta + messages + steps 三视图
"""
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
        )
        response = AgentResponse(
            agent_type=AgentType.MEDICATION,
            content="不建议同服，出血风险增加。",
            success=True,
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
    assert kinds == ["query", "tool_call", "tool_result", "final"]
    assert record["steps"][0]["tool_results"][0]["reranked"] is True
