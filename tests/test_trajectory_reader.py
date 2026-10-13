"""
evaluation/trajectory_reader.py 的离线单测（v2 阶段 1 第 5 步配套）。

全部用 tmp_path 构造假轨迹 JSONL，不读真实 data/trajectories；解析容错
路径（坏行 / SDK 对象块 / 非检索工具结果）逐条锁定——判分器吃的是尽力
而为旁路落盘的轨迹，什么脏数据都可能有，解析必须跳过而非崩溃。
"""
import json

import pytest

from evaluation.trajectory_reader import (
    Trajectory,
    TrajectoryReader,
    assert_eval_data_quality,
    parse_search_results,
    parse_search_results_by_call,
)


# ── 假轨迹构造件 ──────────────────────────────────────────────────────────────

def make_result_item(chunk_id: str = "chunk-1", title: str = "金文档") -> dict:
    """构造一条检索结果 item（与 search_knowledge_base 返回项同构）。"""
    return {"id": chunk_id, "title": title, "content": "正文内容", "score": 0.9, "chunk": 0}


def make_search_payload(*items: dict) -> str:
    """构造 search_knowledge_base 返回 dict 的 JSON 字符串（落盘形态）。"""
    return json.dumps({
        "success": True, "query": "测试查询", "top_k": 5,
        "results": list(items),
        "cached": False, "reranked": True, "rerank_degraded": False,
    })


def make_messages(*tool_result_payloads: str) -> list:
    """构造 messages 视图：每段 payload 配一对 tool_call / tool_result 消息，末尾一条 final。

    tool_result 消息按 agent_orchestrator 的构造方式（json.dumps 字符串块）模拟。
    """
    msgs = []
    for i, payload in enumerate(tool_result_payloads):
        msgs.append({
            "role": "assistant", "agent_type": "medication", "kind": "tool_call",
            "content": [{"type": "tool_use", "id": f"toolu_{i}",
                         "name": "search_knowledge_base", "input": {"query": "q"}}],
        })
        msgs.append({
            "role": "user", "agent_type": "medication", "kind": "tool_result",
            "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": payload}],
        })
    msgs.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": [{"type": "text", "text": "最终结论"}],
    })
    return msgs


def make_trajectory(request_id: str = "req-1", timestamp: str = "2026-10-12T10:00:00",
                    messages=None, steps=None) -> dict:
    """构造一条完整轨迹 dict（三视图齐全的最小形态）。"""
    return {
        "trajectory_id": f"t-{request_id}",
        "request_id": request_id,
        "timestamp": timestamp,
        "meta": {"intent": "medication_interaction", "urgency": "LOW", "escalated": False},
        "messages": messages if messages is not None else make_messages(make_search_payload()),
        "steps": steps if steps is not None else [],
    }


def make_step(tool_calls: list, tool_results: list) -> dict:
    """构造 steps 视图的一步（与编排器 step 结构同构）。"""
    return {"step_id": 1, "agent_type": "medication", "llm_latency_ms": 100.0,
            "tool_calls": tool_calls, "tool_results": tool_results}


def make_clean_search_step(query: str = "q") -> dict:
    """一条干净的 search_knowledge_base 调用步骤（数据质量断言应通过）。"""
    return make_step(
        tool_calls=[{"tool_name": "search_knowledge_base",
                     "args": {"query": query}, "tool_use_id": "toolu_0"}],
        tool_results=[{"tool_name": "search_knowledge_base", "success": True,
                       "result_success": True, "latency_ms": 300.0,
                       "cached": False, "reranked": True, "rerank_degraded": False,
                       "error": ""}],
    )


# ── 索引构建 ──────────────────────────────────────────────────────────────────

def test_load_index_builds_from_day_files(tmp_path):
    """两个按天切分的文件都能进索引，按 request_id 取到对应轨迹。"""
    (tmp_path / "trajectories-2026-10-11.jsonl").write_text(
        json.dumps(make_trajectory("req-a"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trajectories-2026-10-12.jsonl").write_text(
        json.dumps(make_trajectory("req-b"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index = TrajectoryReader(tmp_path).load_index()
    assert set(index) == {"req-a", "req-b"}
    assert isinstance(index["req-a"], Trajectory)
    assert index["req-a"].meta["intent"] == "medication_interaction"


def test_duplicate_request_id_keeps_latest_timestamp(tmp_path):
    """同一 request_id 重跑多次：跨文件、同文件都只留 timestamp 最新的一条。"""
    lines = [
        make_trajectory("req-1", timestamp="2026-10-12T10:00:00"),
        make_trajectory("req-1", timestamp="2026-10-12T11:00:00"),  # 更晚 → 应胜出
    ]
    (tmp_path / "trajectories-2026-10-12.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in lines),
        encoding="utf-8",
    )
    (tmp_path / "trajectories-2026-10-13.jsonl").write_text(
        json.dumps(make_trajectory("req-1", timestamp="2026-10-12T09:00:00"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index = TrajectoryReader(tmp_path).load_index()
    assert index["req-1"].timestamp == "2026-10-12T11:00:00"


def test_bad_json_line_skipped_with_warning(tmp_path, caplog):
    """坏 JSON 行告警跳过，不影响同一文件里其余好行。"""
    (tmp_path / "trajectories-2026-10-12.jsonl").write_text(
        "这不是JSON\n" + json.dumps(make_trajectory("req-ok"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        index = TrajectoryReader(tmp_path).load_index()
    assert set(index) == {"req-ok"}
    assert "坏行" in caplog.text


def test_missing_required_key_skipped(tmp_path, caplog):
    """缺必填键（request_id 等）的行告警跳过，不崩溃。"""
    broken = make_trajectory("req-x")
    del broken["request_id"]
    (tmp_path / "trajectories-2026-10-12.jsonl").write_text(
        json.dumps(broken, ensure_ascii=False) + "\n" +
        json.dumps(make_trajectory("req-ok"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        index = TrajectoryReader(tmp_path).load_index()
    assert set(index) == {"req-ok"}
    assert "缺必填键" in caplog.text


def test_from_dict_missing_key_raises_value_error():
    """from_dict 对缺键数据抛 ValueError（读取端据此决定跳过）。"""
    with pytest.raises(ValueError):
        Trajectory.from_dict({"trajectory_id": "t-1", "request_id": "r"})  # 缺 timestamp/meta 等


# ── 检索结果提取 ──────────────────────────────────────────────────────────────

def test_parse_search_results_extracts_items():
    """正常链路：tool_result 块里的 JSON 字符串被解析成 SearchResultItem。"""
    items = parse_search_results(make_messages(make_search_payload(
        make_result_item("chunk-1", "金文档"), make_result_item("chunk-2", "银文档"),
    )))
    assert [i.id for i in items] == ["chunk-1", "chunk-2"]
    assert items[0].title == "金文档"
    assert items[0].score == 0.9
    assert items[0].chunk == 0


def test_parse_skips_sdk_object_blocks():
    """anthropic 协议下 assistant 消息的 content 是 SDK 对象，落盘成字符串——
    解析必须跳过这些块而不是报错。"""
    messages = make_messages(make_search_payload(make_result_item("chunk-1")))
    # 模拟 SDK 对象被 default=str 序列化后的形态：块不是 dict 而是字符串
    messages[0]["content"] = ["TextBlock(text='我要调用工具', type='text')"]
    messages[-1]["content"] = ["TextBlock(text='最终结论', type='text')"]
    items = parse_search_results(messages)
    assert [i.id for i in items] == ["chunk-1"]


def test_parse_handles_dict_payload():
    """块 content 已是 dict（而非 JSON 字符串）时同样能解析。"""
    messages = [{
        "role": "user", "agent_type": "medication", "kind": "tool_result",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_0",
                     "content": {"success": True, "results": [make_result_item("chunk-1")]}}],
    }]
    items = parse_search_results(messages)
    assert [i.id for i in items] == ["chunk-1"]


def test_parse_skips_non_search_tool_result():
    """非检索工具的结果（没有 results 键，如 inspect_request_context）被跳过。"""
    messages = [{
        "role": "user", "agent_type": "symptom_triage", "kind": "tool_result",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_0",
                     "content": json.dumps({"intent": "symptom_check", "urgency": "LOW"})}],
    }]
    assert parse_search_results(messages) == []


def test_parse_skips_bad_json_with_warning(caplog):
    """块 content 不是合法 JSON 时告警跳过，其余块继续解析。"""
    # 两条 tool_result：第一条坏 JSON（跳过），第二条正常（应解析出 chunk-2）
    messages = make_messages(
        make_search_payload(make_result_item("chunk-1")),
        make_search_payload(make_result_item("chunk-2")),
    )
    messages[1]["content"] = [{"type": "tool_result", "tool_use_id": "toolu_0", "content": "不是JSON"}]
    with caplog.at_level("WARNING"):
        items = parse_search_results(messages)
    assert [i.id for i in items] == ["chunk-2"]
    assert "JSON 解析失败" in caplog.text


def test_parse_dedups_by_chunk_id():
    """两次搜索命中同一 chunk：合并结果只保留一次（覆盖判分只算一次）。"""
    items = parse_search_results(make_messages(
        make_search_payload(make_result_item("chunk-1")),
        make_search_payload(make_result_item("chunk-1"), make_result_item("chunk-2")),
    ))
    assert [i.id for i in items] == ["chunk-1", "chunk-2"]


def test_parse_by_call_groups_by_tool_use_id():
    """按 tool_use_id 分组：两次搜索各自的结果分开存放。"""
    by_call = parse_search_results_by_call(make_messages(
        make_search_payload(make_result_item("chunk-1")),
        make_search_payload(make_result_item("chunk-2")),
    ))
    assert set(by_call) == {"toolu_0", "toolu_1"}
    assert [i.id for i in by_call["toolu_0"]] == ["chunk-1"]
    assert [i.id for i in by_call["toolu_1"]] == ["chunk-2"]


# ── 检索步数 ──────────────────────────────────────────────────────────────────

def test_search_calls_counts_only_search_tool():
    """步数只数 search_knowledge_base：其他工具的调用不算检索步数。"""
    traj = Trajectory.from_dict(make_trajectory(steps=[
        make_step(
            tool_calls=[
                {"tool_name": "search_knowledge_base", "args": {"query": "第一跳"}, "tool_use_id": "toolu_0"},
                {"tool_name": "build_medication_plan",
                 "args": {"current_medications": "华法林"}, "tool_use_id": "toolu_1"},
            ],
            tool_results=[],
        ),
        make_step(
            tool_calls=[{"tool_name": "search_knowledge_base", "args": {"query": "第二跳"}, "tool_use_id": "toolu_2"}],
            tool_results=[],
        ),
    ]))
    calls = traj.search_calls()
    assert len(calls) == 2
    assert [c["args"]["query"] for c in calls] == ["第一跳", "第二跳"]


# ── 最终回答提取 ──────────────────────────────────────────────────────────────

def test_final_answer_extracts_text_blocks():
    """final 消息的 text 块按序拼接；非 text 块忽略。"""
    traj = Trajectory.from_dict(make_trajectory())
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": [
            {"type": "text", "text": "不建议同服。"},
            {"type": "thinking", "text": "（内部思考，不是给用户的）"},
            {"type": "text", "text": "出血风险增加。"},
        ],
    })
    assert traj.final_answer() == "不建议同服。出血风险增加。"


def test_final_answer_skips_sdk_object_blocks():
    """SDK 对象落盘成字符串的块跳过，其余 text 块照常提取。"""
    traj = Trajectory.from_dict(make_trajectory())
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": [
            "TextBlock(text='垃圾串', type='text')",
            {"type": "text", "text": "正常内容"},
        ],
    })
    assert traj.final_answer() == "正常内容"


def test_final_answer_no_final_message_returns_empty():
    """没有 final 消息（轨迹被截断）→ 空串，判分方记失败原因。"""
    traj = Trajectory.from_dict(make_trajectory(messages=[]))
    assert traj.final_answer() == ""


def test_final_answer_keeps_last_final_only():
    """多条 final（重试残留）时只取最后一条，不拼接旧内容。"""
    traj = Trajectory.from_dict(make_trajectory())
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": [{"type": "text", "text": "旧结论"}],
    })
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": [{"type": "text", "text": "新结论"}],
    })
    assert traj.final_answer() == "新结论"


def test_final_answer_accepts_plain_string():
    """生产轨迹的 final content 是纯字符串（2026-10-13 pilot 实测），原样返回。"""
    traj = Trajectory.from_dict(make_trajectory())
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": "不建议同服，出血风险增加。",
    })
    assert traj.final_answer() == "不建议同服，出血风险增加。"


def test_final_answer_plain_string_keeps_last_final_only():
    """字符串形态同样遵守「只取最后一条 final」（与块列表形态口径一致）。"""
    traj = Trajectory.from_dict(make_trajectory())
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": "旧结论",
    })
    traj.messages.append({
        "role": "assistant", "agent_type": "medication", "kind": "final",
        "content": "新结论",
    })
    assert traj.final_answer() == "新结论"


# ── 数据质量断言 ──────────────────────────────────────────────────────────────

def test_assert_quality_passes_on_clean_trajectory():
    """干净轨迹（cached=False、rerank_degraded=False、检索成功）不抛异常。"""
    traj = Trajectory.from_dict(make_trajectory(steps=[make_clean_search_step()]))
    assert_eval_data_quality(traj)  # 不抛即通过


@pytest.mark.parametrize("violation", [
    {"cached": True},
    {"rerank_degraded": True},
    {"result_success": False},
    {"success": False},
])
def test_assert_quality_raises_on_violations(violation):
    """四种违规（缓存回放/精排降级/检索失败/调用失败）逐一拦截。"""
    step = make_clean_search_step()
    step["tool_results"][0].update(violation)
    traj = Trajectory.from_dict(make_trajectory(steps=[step]))
    with pytest.raises(AssertionError, match="数据质量违规"):
        assert_eval_data_quality(traj)


def test_assert_quality_ignores_non_search_tools():
    """非检索工具没有 cached/rerank 语义，断言只查 search_knowledge_base。"""
    step = make_step(
        tool_calls=[{"tool_name": "build_medication_plan", "args": {}, "tool_use_id": "toolu_0"}],
        tool_results=[{"tool_name": "build_medication_plan", "success": True,
                       "result_success": True, "cached": True,  # 非检索工具的键不被检查
                       "reranked": False, "rerank_degraded": True, "error": ""}],
    )
    traj = Trajectory.from_dict(make_trajectory(steps=[step]))
    assert_eval_data_quality(traj)  # 不抛即通过
