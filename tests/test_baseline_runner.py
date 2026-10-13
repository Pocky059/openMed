"""
evaluation/baseline_runner.py 的离线单测（v2 阶段 1 第 7 步配套）。

锁定五件事（其余判分器模块已有各自测试）：
  1. 急症告警模拟收件箱：POST 落网、退出落盘 jsonl；run_emergency 默认
     分支显式注入类属性指向 mock、跑完恢复 None，告警条数进
     _webhook_payloads（v2-plan 1.3 做法 2，用户拍板）
  2. 真实 webhook 安全门：env 配了真实 URL 且未 --allow-emergency-webhook
     → EvalAbortError 中止；显式 allow 才放行
  3. 断点续跑/部分重跑：--resume 跳过已完成行（不把全部题重跑一遍白花钱）；
     --merge 指定题强制重跑、新行覆盖旧行
  4. 生成质量层（方案 C）：空回答短路零 LLM 调用；正常路径 faithfulness +
     answer_relevancy 落 scores.generation_quality；long_session / emergency
     不评生成质量（2026-10-13 用户拍板），旧产物重聚合时统一清理
  5. 轨迹质量口径：只有走过 Agent 编排的题预期产轨迹——intent / qa single
     不产轨迹，缺轨迹不算缺失（2026-10-13 pilot 误报修复）
  6. --rejudge-from 重聚合：旧 run 产物读入、指定题重判、其余行保留、
     口径清理后聚合落盘新 run_id（2026-10-13 用户指令）
"""
import asyncio
import json
import http.client
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from evaluation.baseline_runner import (
    BaselineRunner, EmergencyWebhookInbox, EvalAbortError,
    case_conv_id, render_webhook_summary,
)
from evaluation.trajectory_reader import Trajectory


# ── 假组件（沿用 test_chat_service.py 的假编排器/假记忆风格）────────────────

class FakeClient:
    """统计 create 调用次数的假 LLM 客户端（本测试里 judge 不该被调到）。"""

    def __init__(self):
        self.messages = SimpleNamespace(calls=[])

        async def create(**kwargs):
            self.messages.calls.append(kwargs)
            return SimpleNamespace(content=[{"type": "text", "text": '{"score": 0.5}'}])

        self.messages.create = create


class FakeMemory:
    """假记忆管理器（get_context/add_message/update_profile + get_summary）。"""

    def __init__(self):
        self.calls = []

    async def get_context(self, user_id, conv_id, query=""):
        self.calls.append(("get_context", user_id, conv_id))
        return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "")

    async def add_message(self, user_id, conv_id, role, content):
        self.calls.append(("add_message", role))

    async def update_profile(self, user_id, conv_id):
        self.calls.append(("update_profile", user_id, conv_id))

    async def get_summary(self, user_id, conv_id):
        return "summary 原文"


def _post(port: int, path: str, payload: dict):
    """同步 POST（供收件箱单测与假编排器的告警推送共用）。

    注意：http.client 对 str body 默认按 latin-1 编码，中文会炸，
    必须先编码成 UTF-8 字节（收件箱那端按 utf-8 解码）。
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


class WebhookFakeOrchestrator:
    """模拟急症升级链路：run 时若 EmergencyAgent._webhook_url 已注入，
    就 POST 一条告警过去——与真实 _notify_webhook 同构的落网路径，
    用来验证「收件箱真的收到告警」而不是只有空文件。"""

    async def recognize_intent(self, message, history=None):
        return SimpleNamespace(
            entities={},
            intent=SimpleNamespace(value="emergency"),
            intent_group="emergency",
            urgency=SimpleNamespace(name="CRITICAL", value=4),
            confidence=1.0,
            source_scores={},
        )

    async def run(self, req):
        from urllib.parse import urlparse

        from agents.agent_orchestrator import EmergencyAgent
        url = EmergencyAgent._webhook_url
        if url:
            payload = {"user_id": req.user_id, "message": req.message,
                       "urgency": "CRITICAL", "intent": "emergency", "entities": {}}
            await asyncio.get_running_loop().run_in_executor(
                None, _post, urlparse(url).port, "/emergency", payload)
        return SimpleNamespace(
            request_id="req-1",
            response="已为你转接人工",
            agent_type=SimpleNamespace(value="emergency"),
            intent=SimpleNamespace(value="emergency"),
            agent_types=[SimpleNamespace(value="emergency")],
            primary_agent=None,
            supporting_agents=[],
            tools_used=[],
            routing_reason="急症门控",
            routing_confidence=1.0,
            escalated=True,
            latency_ms=1.0,
        )


class FakeCtx:
    """BaselineRunner 用得到的 ctx 字段最小集（judge_client 显式传入）。"""

    def __init__(self, orchestrator=None, memory=None):
        self.cfg = {"api_key": "test", "model": "test-model"}
        self.orchestrator = orchestrator or WebhookFakeOrchestrator()
        self.memory = memory or FakeMemory()
        self.tool_manager = SimpleNamespace()
        self.recognizer = SimpleNamespace()


@pytest.fixture
def no_llm_network(monkeypatch):
    """构造 BaselineRunner 时强制 LLM 失败识别器内部会 new 客户端，换成假的。"""
    monkeypatch.setattr("core.intent_recognizer.create_llm_client",
                        lambda api_key, base_url: FakeClient())


def _make_runner(tmp_path, monkeypatch, judge_client=None, allow_webhook=False,
                 orchestrator=None, memory=None):
    runner = BaselineRunner(
        FakeCtx(orchestrator, memory), results_root=tmp_path,
        judge_client=judge_client or FakeClient(),
        allow_webhook=allow_webhook,
    )
    monkeypatch.setattr("evaluation.baseline_runner._WEBHOOK_DRAIN_S", 0)
    return runner


# ── 1. 模拟收件箱 ──────────────────────────────────────────────────────────────

def test_inbox_receives_and_persists(tmp_path, monkeypatch):
    """POST 落网（带 received_at），退出后落盘 jsonl 一行一个告警。"""
    monkeypatch.setattr("evaluation.baseline_runner._WEBHOOK_DRAIN_S", 0)
    out = tmp_path / "emergency_webhooks.jsonl"
    with EmergencyWebhookInbox(out) as inbox:
        assert inbox.port > 0  # 随机端口，不抢固定端口
        payload = {"user_id": "eval-ESC-001", "message": "胸口剧痛喘不上气",
                   "urgency": "CRITICAL"}
        status, _ = _post(inbox.port, "/emergency", payload)
        assert status == 200
        assert len(inbox.received) == 1
        assert inbox.received[0]["user_id"] == "eval-ESC-001"
        assert "received_at" in inbox.received[0]

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["urgency"] == "CRITICAL"
    assert saved["message"] == "胸口剧痛喘不上气"


def test_render_webhook_summary_template():
    """人话摘要模板：user_id/message/urgency/intent 都进文案。"""
    text = render_webhook_summary({
        "user_id": "eval-ESC-001", "message": "胸痛伴出汗",
        "urgency": "CRITICAL", "intent": "emergency",
        "entities": {"drug": ["硝酸甘油"]},
    })
    assert "eval-ESC-001" in text
    assert "胸痛伴出汗" in text
    assert "CRITICAL" in text
    assert "emergency" in text
    assert "请人工客服跟进" in text


# ── 2. run_emergency：注入/恢复/安全门 ─────────────────────────────────────────

def _emergency_case():
    return {"id": "ESC-001", "message": "胸口剧痛喘不上气",
            "expected_escalated": True, "expected_urgency": "CRITICAL",
            "split": "train"}


def test_run_emergency_default_inbox(tmp_path, monkeypatch, no_llm_network):
    """默认分支：跑题期间类属性指向 mock，告警落网落盘，跑完恢复 None。"""
    monkeypatch.delenv("OPENMED_EMERGENCY_WEBHOOK_URL", raising=False)
    from agents.agent_orchestrator import EmergencyAgent
    EmergencyAgent._webhook_url = None  # 测试间隔离

    runner = _make_runner(tmp_path, monkeypatch)

    rows = asyncio.run(runner.run_emergency([_emergency_case()]))

    assert EmergencyAgent._webhook_url is None  # finally 恢复，不残留注入
    assert rows[0]["passed"] is True
    assert rows[0]["scores"]["actual_escalated"] is True
    assert rows[0]["scores"]["actual_urgency"] == "CRITICAL"
    # 告警真的落网（假编排器在跑题时 POST 到注入地址）
    assert len(runner._webhook_payloads) == 1
    assert runner._webhook_payloads[0]["user_id"] == case_conv_id("ESC-001")
    out = tmp_path / "emergency_webhooks.jsonl"
    saved = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 1
    assert saved[0]["urgency"] == "CRITICAL"


def test_run_emergency_gate_blocks_real_url(tmp_path, monkeypatch, no_llm_network):
    """env 配了真实 URL 且未 allow → 中止；显式 allow 才放行（不进收件箱）。"""
    monkeypatch.setenv("OPENMED_EMERGENCY_WEBHOOK_URL", "https://hooks.example.com/x")
    runner = _make_runner(tmp_path, monkeypatch)
    with pytest.raises(EvalAbortError):
        asyncio.run(runner.run_emergency([_emergency_case()]))
    assert not (tmp_path / "emergency_webhooks.jsonl").exists()  # 中止时不落盘

    runner2 = _make_runner(tmp_path, monkeypatch, allow_webhook=True)
    rows = asyncio.run(runner2.run_emergency([_emergency_case()]))
    assert rows[0]["passed"] is True
    assert runner2._webhook_payloads == []  # allow 分支不接收件箱


def test_run_emergency_empty_cases_short_circuits(tmp_path, monkeypatch, no_llm_network):
    """空考卷直接返回：不起 server、不落盘、不碰 webhook 门控。"""
    runner = _make_runner(tmp_path, monkeypatch)
    rows = asyncio.run(runner.run_emergency([]))
    assert rows == []
    assert not (tmp_path / "emergency_webhooks.jsonl").exists()


# ── 3. 断点续跑 / 部分重跑 ─────────────────────────────────────────────────────

def _intent_exam():
    return {"intent": [
        {"id": "INT-1", "message": "感冒了怎么办", "expected_intent": "query", "split": "train"},
        {"id": "INT-2", "message": "头疼挂哪科", "expected_intent": "department_guide", "split": "train"},
    ], "qa": [], "multihop": [], "long_session": [], "emergency": []}


# ── 5. 轨迹质量口径 ───────────────────────────────────────────────────────────

def _row_for_quality(case_id: str, exam: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    """_assert_quality 用的一行最小输入（跑题阶段其余字段判分不读）。"""
    return {"case_id": case_id, "exam": exam, "conv_id": case_conv_id(case_id),
            "scores": {}, "detail": detail}


def test_expects_trajectory_by_exam():
    """只有走过 Agent 编排的题预期产轨迹：intent / qa single 不产。"""
    runner = BaselineRunner.__new__(BaselineRunner)  # 纯静态判断，不需要构造
    assert runner._expects_trajectory(_row_for_quality("INT-1", "intent", {})) is False
    assert runner._expects_trajectory(_row_for_quality("INT-1", "intent_llm_failure", {})) is False
    assert runner._expects_trajectory(
        _row_for_quality("QA-1", "qa", {"gold_chunks": ["c1"]})) is False  # single 档直测检索
    assert runner._expects_trajectory(
        _row_for_quality("QA-1", "qa", {"gold_answer": "挂内科"})) is True  # multi 档走编排
    for exam in ("multihop", "long_session", "emergency"):
        assert runner._expects_trajectory(_row_for_quality("X-1", exam, {})) is True


def test_assert_quality_skips_trajectoryless_exams(tmp_path, monkeypatch):
    """intent / qa single 不产轨迹：缺轨迹不记 missing、不进 total（pilot 误报修复）。"""
    runner = _make_runner(tmp_path, monkeypatch)
    rows = [
        _row_for_quality("INT-1", "intent", {}),
        _row_for_quality("INT-2", "intent_llm_failure", {}),
        _row_for_quality("QA-1", "qa", {"gold_chunks": ["c1"]}),
    ]
    report = runner._assert_quality(rows, traj_index={})
    assert report == {"total": 0, "violations": [], "missing": []}


def test_assert_quality_reports_missing_for_trajectory_exams(tmp_path, monkeypatch):
    """预期产轨迹的题（长会话/急症/qa multi）缺轨迹记 missing。"""
    runner = _make_runner(tmp_path, monkeypatch)
    rows = [
        _row_for_quality("LS-1", "long_session", {}),
        _row_for_quality("QA-2", "qa", {"gold_answer": "挂内科"}),
    ]
    report = runner._assert_quality(rows, traj_index={})
    assert sorted(report["missing"]) == ["LS-1", "QA-2"]
    assert report["total"] == 0


def test_assert_quality_multihop_missing_aborts(tmp_path, monkeypatch):
    """multihop 缺轨迹直接中止整个 run（判分依赖轨迹，宁可重跑不产脏基线）。"""
    runner = _make_runner(tmp_path, monkeypatch)
    with pytest.raises(EvalAbortError):
        runner._assert_quality([_row_for_quality("MH-1", "multihop", {})], traj_index={})


def _prepare_run_all(runner, monkeypatch, seen, captured):
    """把 run_all 里与跑题/判分/落盘无关的部分全部换成记录型假实现。"""

    async def fake_warmup():
        pass

    async def fake_collect(rows):
        return {}

    def fake_assert(rows, traj_index):
        return {"total": 0, "violations": [], "missing": []}

    async def fake_judge(rows, traj_index):
        pass

    def fake_build(rows, manifest):
        captured["rows"] = list(rows)
        return {"headline": {}, "config": {}, "exams": {}, "safety": {},
                "emergency": {}, "retrieval": {}, "results": list(rows)}

    def fake_save_outputs(baseline, rows):
        pass

    monkeypatch.setattr(runner, "_warmup", fake_warmup)
    monkeypatch.setattr(runner, "_collect_trajectories", fake_collect)
    monkeypatch.setattr(runner, "_assert_quality", fake_assert)
    monkeypatch.setattr(runner, "_judge_rows", fake_judge)
    monkeypatch.setattr(runner, "_build_baseline", fake_build)
    monkeypatch.setattr(runner, "_save_outputs", fake_save_outputs)


def test_run_all_resume_skips_completed(tmp_path, monkeypatch, no_llm_network):
    """--resume：cases.jsonl 已有的行跳过跑题（不把全部题重跑一遍白花钱）。"""
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "cases.jsonl").write_text(json.dumps(
        {"case_id": "INT-1", "exam": "intent", "passed": True,
         "scores": {"confidence": 0.9}, "split": "train"}) + "\n",
        encoding="utf-8")
    runner = _make_runner(tmp_path, monkeypatch)
    seen, captured = [], {}
    _prepare_run_all(runner, monkeypatch, seen, captured)

    async def fake_run_intent(cases):
        seen.extend(c["id"] for c in cases)
        return [runner._row(c, "intent", passed=True, scores={"confidence": 0.5},
                            detail={"expected": c["expected_intent"],
                                    "actual": c["expected_intent"]}) for c in cases]

    monkeypatch.setattr(runner, "run_intent", fake_run_intent)

    asyncio.run(runner.run_all(_intent_exam(), manifest={}, resume=True))

    assert seen == ["INT-2"]  # INT-1 已完成被跳过
    assert sorted(r["case_id"] for r in captured["rows"]) == ["INT-1", "INT-2"]


def test_run_all_merge_overrides_existing(tmp_path, monkeypatch, no_llm_network):
    """--cases + --merge：指定题强制重跑，新行覆盖旧行，其余保留。"""
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "cases.jsonl").write_text("\n".join([
        json.dumps({"case_id": "INT-1", "exam": "intent", "passed": True,
                    "scores": {"confidence": 0.9}, "split": "train"}),
        json.dumps({"case_id": "INT-2", "exam": "intent", "passed": True,
                    "scores": {"confidence": 0.8}, "split": "train"}),
    ]) + "\n", encoding="utf-8")
    runner = _make_runner(tmp_path, monkeypatch)
    seen, captured = [], {}
    _prepare_run_all(runner, monkeypatch, seen, captured)

    async def fake_run_intent(cases):
        seen.extend(c["id"] for c in cases)
        return [runner._row(c, "intent", passed=True, scores={"confidence": 0.1},
                            detail={"expected": c["expected_intent"],
                                    "actual": c["expected_intent"]}) for c in cases]

    monkeypatch.setattr(runner, "run_intent", fake_run_intent)

    asyncio.run(runner.run_all(_intent_exam(), manifest={},
                               case_ids=["INT-1"], merge=True))

    assert seen == ["INT-1"]  # 只重跑指定题
    merged = {r["case_id"]: r for r in captured["rows"]}
    assert sorted(merged) == ["INT-1", "INT-2"]  # 未重跑的 INT-2 保留
    assert merged["INT-1"]["scores"]["confidence"] == 0.1  # 新行覆盖旧行
    assert merged["INT-2"]["scores"]["confidence"] == 0.8


# ── 4. 生成质量层（方案 C）─────────────────────────────────────────────────────

def _traj_with_answer(answer: str):
    """一条带最终回答的轨迹（tool_result 供 contexts 提取）。

    轨迹落盘结构：tool_result 消息的 content 是块列表，每块
    {"type": "tool_result", "content": "<JSON 字符串>"}（与
    parse_search_results 的解析路径一致）。
    """
    return Trajectory.from_dict({
        "trajectory_id": "t1", "request_id": "r1", "timestamp": "2026-01-01T00:00:00",
        "meta": {"conv_id": "eval-QA-001"},
        "messages": [
            {"kind": "final", "content": [{"type": "text", "text": answer}]},
            {"kind": "tool_result", "content": [{"type": "tool_result",
                                                 "tool_use_id": "tu1",
                                                 "content": json.dumps(
                {"results": [{"id": "c1", "title": "布洛芬用法", "content": "一次一片",
                              "score": 0.9, "chunk": 0}]})}]},
        ],
        "steps": [],
    })


def test_generation_quality_empty_answer_skips(tmp_path, monkeypatch, no_llm_network):
    """空回答短路：不调 LLM，记 skipped 标记。"""
    client = FakeClient()
    runner = _make_runner(tmp_path, monkeypatch, judge_client=client)
    row = {"case_id": "QA-001", "scores": {}, "detail": {"answer": "  "},
           "judge_failed": False}
    asyncio.run(runner._judge_generation_quality(
        row, _traj_with_answer(""), question="布洛芬怎么吃"))
    assert row["scores"]["generation_quality"] == {"skipped": "回答为空"}
    assert client.messages.calls == []


def test_generation_quality_populates_scores(tmp_path, monkeypatch, no_llm_network):
    """正常路径：faithfulness + answer_relevancy 落 generation_quality。"""
    calls = {}

    async def fake_faithfulness(question, answer, contexts, client, model):
        calls["faith"] = (question, answer, contexts)
        return SimpleNamespace(score=1.0, supported=2, total=2, judge_failed=False)

    async def fake_relevancy(question, answer, client, model, embedder=None):
        calls["relevancy"] = (question, answer)
        return SimpleNamespace(score=0.8, judge_failed=False)

    monkeypatch.setattr("evaluation.ragas_metrics.judge_faithfulness", fake_faithfulness)
    monkeypatch.setattr("evaluation.ragas_metrics.judge_answer_relevancy", fake_relevancy)

    runner = _make_runner(tmp_path, monkeypatch)
    row = {"case_id": "QA-001", "scores": {}, "detail": {"answer": ""},
           "judge_failed": False}
    asyncio.run(runner._judge_generation_quality(
        row, _traj_with_answer("布洛芬一次一片，饭后服用。"), question="布洛芬怎么吃"))

    quality = row["scores"]["generation_quality"]
    assert quality["faithfulness"] == {"score": 1.0, "supported": 2, "total": 2,
                                       "judge_failed": False}
    assert quality["answer_relevancy"] == {"score": 0.8, "judge_failed": False}
    # faithfulness 的上下文来自轨迹检索结果（一次一片），问题原样传入
    assert calls["faith"][0] == "布洛芬怎么吃"
    assert calls["faith"][2] == ["一次一片"]


def test_strip_generation_quality_only_ls_and_emergency():
    """口径清理：只删 long_session / emergency 行的 generation_quality，
    qa / multihop 的保留（2026-10-13 用户拍板：这两集不评生成质量）。"""
    rows = [
        {"exam": "long_session", "scores": {"recall": 1.0, "generation_quality": {"x": 1}}},
        {"exam": "emergency", "scores": {"escalated": True, "generation_quality": {"x": 1}}},
        {"exam": "qa", "scores": {"mean": 0.8, "generation_quality": {"x": 1}}},
        {"exam": "multihop", "scores": {"success": True, "generation_quality": {"x": 1}}},
    ]
    BaselineRunner._strip_generation_quality(rows)
    assert "generation_quality" not in rows[0]["scores"]
    assert "generation_quality" not in rows[1]["scores"]
    assert rows[2]["scores"]["generation_quality"] == {"x": 1}
    assert rows[3]["scores"]["generation_quality"] == {"x": 1}


def test_judge_rows_skips_emergency_entirely(tmp_path, monkeypatch, no_llm_network):
    """急症判分无任务：升级与否跑题时已确定性判完，judge 阶段零 LLM 调用、
    不产生 generation_quality（2026-10-13 用户拍板：急症不评生成质量）。"""
    client = FakeClient()
    runner = _make_runner(tmp_path, monkeypatch, judge_client=client)
    rows = [{"case_id": "ESC-001", "exam": "emergency", "scores": {},
             "detail": {"expected_escalated": True, "actual_escalated": True},
             "judge_failed": False}]
    asyncio.run(runner._judge_rows(rows, {}))
    assert client.messages.calls == []
    assert rows[0]["scores"] == {}  # 没有任何判分产出


def test_judge_long_session_no_generation_quality(tmp_path, monkeypatch, no_llm_network):
    """长会话判分只产 summary 四字段结果，不产 generation_quality。"""
    from evaluation.multihop_scorer import LongSessionResult

    async def fake_score(summary, case, judge=None):
        return LongSessionResult(
            case_id="LS-006", total=3, hit=3, judge_hit=0, recall=1.0,
            loss_rate=0.0, judge_failed=0, items=[])

    monkeypatch.setattr("evaluation.multihop_scorer.score_long_session", fake_score)
    client = FakeClient()
    runner = _make_runner(tmp_path, monkeypatch, judge_client=client)
    runner._cases_by_id = {"LS-006": {"id": "LS-006", "turns": [], "gold_items": []}}
    row = {"case_id": "LS-006", "exam": "long_session", "scores": {},
           "detail": {"summary": '{"allergies": []}'}, "judge_failed": False}
    asyncio.run(runner._judge_long_session(row, None))
    assert row["scores"]["recall"] == 1.0
    assert "generation_quality" not in row["scores"]


def test_judge_rows_resets_stale_judge_failed(tmp_path, monkeypatch, no_llm_network):
    """重判路径：旧行的 judge_failed=True 进判分前归零，结果由本次判分决定
    （2026-10-13 重聚合实测：三场 LS 旧兜底标记残留）。"""
    from evaluation.multihop_scorer import LongSessionResult

    async def fake_score(summary, case, judge=None):
        return LongSessionResult(
            case_id="LS-006", total=3, hit=3, judge_hit=0, recall=1.0,
            loss_rate=0.0, judge_failed=0, items=[])

    monkeypatch.setattr("evaluation.multihop_scorer.score_long_session", fake_score)
    runner = _make_runner(tmp_path, monkeypatch)
    runner._cases_by_id = {"LS-006": {"id": "LS-006", "turns": [], "gold_items": []}}
    row = {"case_id": "LS-006", "exam": "long_session", "scores": {},
           "detail": {"summary": '{"allergies": []}'},
           "judge_failed": True}  # 旧行残留（faithfulness 兜底时代留下）
    asyncio.run(runner._judge_rows([row], {}))
    assert row["judge_failed"] is False  # 本次 summary 判分零失败 → 归零


# ── 6. --rejudge-from 重聚合 ───────────────────────────────────────────────────

def test_run_rejudge_rejudges_specified_cases(tmp_path, monkeypatch, no_llm_network):
    """从旧 run 产物重聚合：指定题重判覆盖、其余行保留、口径清理、聚合落盘。"""
    results = tmp_path / "results"
    prev = results / "baseline-20261013"
    prev.mkdir(parents=True)
    (prev / "cases.jsonl").write_text("\n".join([
        json.dumps({"case_id": "LS-006", "exam": "long_session", "split": "train",
                    "conv_id": "eval-LS-006", "passed": True, "judge_failed": True,
                    "scores": {"recall": 0.5, "generation_quality": {"x": 1}},
                    "detail": {"summary": '{"allergies": []}'},
                    "failure_reason": "", "trajectory_id": "", "request_id": "",
                    "retrieved_chunk_ids": []}),
        json.dumps({"case_id": "ESC-001", "exam": "emergency", "split": "train",
                    "conv_id": "eval-ESC-001", "passed": True, "judge_failed": False,
                    "scores": {"escalated": True, "generation_quality": {"x": 1}},
                    "detail": {"expected_escalated": True, "actual_escalated": True},
                    "failure_reason": "", "trajectory_id": "", "request_id": "",
                    "retrieved_chunk_ids": []}),
        json.dumps({"case_id": "QA-001", "exam": "qa", "split": "train",
                    "conv_id": "eval-QA-001", "passed": True, "judge_failed": False,
                    "scores": {"mean": 0.8, "generation_quality": {"x": 1}},
                    "detail": {"gold_answer": "挂内科"},
                    "failure_reason": "", "trajectory_id": "", "request_id": "",
                    "retrieved_chunk_ids": []}),
    ]) + "\n", encoding="utf-8")
    (prev / "trajectory_quality_report.json").write_text(
        json.dumps({"total": 2, "violations": [], "missing": []}), encoding="utf-8")
    (prev / "emergency_webhooks.jsonl").write_text(
        json.dumps({"user_id": "eval-ESC-001", "urgency": "CRITICAL"}) + "\n",
        encoding="utf-8")

    runner = _make_runner(results / "baseline-20261013-rev2", monkeypatch)
    judged, built = [], {}

    async def fake_collect(rows):
        return {}

    async def fake_judge(rows, traj_index):
        judged.extend(r["case_id"] for r in rows)
        for r in rows:
            r["scores"] = {"recall": 1.0, "judge_failed": False}
            r["judge_failed"] = False  # 判分前归零的语义在 _judge_rows 里，此处模拟其结果

    def fake_build(rows, manifest):
        built["rows"] = list(rows)
        return {"headline": {}, "config": {}, "exams": {}, "safety": {},
                "emergency": {}, "retrieval": {}, "results": list(rows)}

    async def fake_save(baseline, rows):
        pass

    monkeypatch.setattr(runner, "_collect_trajectories", fake_collect)
    monkeypatch.setattr(runner, "_judge_rows", fake_judge)
    monkeypatch.setattr(runner, "_build_baseline", fake_build)
    monkeypatch.setattr(runner, "_save_outputs", fake_save)

    exams = {"long_session": [{"id": "LS-006"}], "emergency": [{"id": "ESC-001"}],
             "qa": [{"id": "QA-001"}]}
    asyncio.run(runner.run_rejudge("baseline-20261013", exams, manifest={},
                                   case_ids=["LS-006"]))

    assert judged == ["LS-006"]  # 只有指定题重判
    by_id = {r["case_id"]: r for r in built["rows"]}
    assert sorted(by_id) == ["ESC-001", "LS-006", "QA-001"]  # 其余行保留
    assert by_id["LS-006"]["scores"]["recall"] == 1.0        # 新行覆盖
    assert by_id["LS-006"]["judge_failed"] is False
    assert by_id["QA-001"]["scores"]["mean"] == 0.8          # 旧行不动
    # 口径清理：LS/ESC 的 generation_quality 删除，QA 保留
    assert "generation_quality" not in by_id["LS-006"]["scores"]
    assert "generation_quality" not in by_id["ESC-001"]["scores"]
    assert by_id["QA-001"]["scores"]["generation_quality"] == {"x": 1}
    # 旧 run 的质量报告与收件箱已带入（_save_outputs 用的状态）
    assert runner._quality_report == {"total": 2, "violations": [], "missing": []}
    assert len(runner._webhook_payloads) == 1
    # 新 run 目录已落盘 cases.jsonl（_save_rows 真实路径）
    new_rows = [json.loads(l) for l in
                (results / "baseline-20261013-rev2" / "cases.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(new_rows) == 3


def test_run_rejudge_missing_prev_run_raises(tmp_path, monkeypatch, no_llm_network):
    """旧 run 产物不存在 → FileNotFoundError（run_id 对不上的直接报错，别静默产出空基线）。"""
    runner = _make_runner(tmp_path / "results" / "baseline-20261013-rev2", monkeypatch)
    with pytest.raises(FileNotFoundError):
        asyncio.run(runner.run_rejudge("no-such-run", {}, manifest={}))


# ── 5. 进度条 ─────────────────────────────────────────────────────────────────

def test_progress_disabled_is_silent(capsys):
    """默认关（单测/日志重定向场景）：start/tick/note 全部不输出。"""
    from evaluation.baseline_runner import _Progress
    p = _Progress(enabled=False)
    p.note("预热中")
    p.start("跑题", 10)
    for _ in range(10):
        p.tick("x 1.0s")
    assert capsys.readouterr().err == ""


def test_progress_renders_and_ends_with_newline(capsys):
    """开：每 tick 刷新一行（\r 覆盖），跑完补换行不糊下一段输出。"""
    from evaluation.baseline_runner import _Progress
    p = _Progress(enabled=True)
    p.start("跑题", 3)
    p.tick("INT-001 1.0s")
    p.tick("INT-002 2.0s")
    p.tick("INT-003 3.0s")
    err = capsys.readouterr().err
    assert "跑题" in err
    assert "[3/3]" in err          # 计数到满
    assert "INT-003 3.0s" in err   # 最后一题的备注可见
    assert err.endswith("\n")      # 收尾换行
    assert err.count("[1/3]") == 1 and err.count("[3/3]") == 1


def test_progress_zero_total_is_silent(capsys):
    """total=0（resume 后没题可跑）：start/tick 都不输出、不除零。"""
    from evaluation.baseline_runner import _Progress
    p = _Progress(enabled=True)
    p.start("跑题", 0)
    p.tick("不该出现")
    assert capsys.readouterr().err == ""
