"""
evaluation/multihop_scorer.py 的离线单测（v2 阶段 1 第 5 步配套）。

轨迹用 test_trajectory_reader 同款构造件（检索 payload / messages / steps），
方向 judge 与长会话复核 judge 都注入假函数——全程不联网。锁定要点：
判分纪律「对照行为不对照知识」——实体链命中只看检索结果，长会话
只看 summary 四字段（回答里提到不算）。
"""
import asyncio
import json

import pytest

from evaluation.multihop_scorer import (
    LongSessionResult,
    MultihopResult,
    score_long_session,
    score_multihop,
)
from evaluation.trajectory_reader import Trajectory

# 复用 test_trajectory_reader 的构造件（同仓库测试间共享的假轨迹形态）
from tests.test_trajectory_reader import (
    make_messages,
    make_result_item,
    make_search_payload,
    make_trajectory,
)

# ── 假 judge ──────────────────────────────────────────────────────────────────

async def judge_correct(question, answer, case):
    """方向 judge：恒判 correct_avoid。"""
    return "correct_avoid"


async def judge_wrong(question, answer, case):
    return "wrong_direction"


async def judge_boom(question, answer, case):
    raise RuntimeError("judge 挂了")


async def long_judge_true(field, field_text, item):
    """长会话复核 judge：恒判「摘要里有同义改写」。"""
    return True


async def long_judge_false(field, field_text, item):
    return False


# ── 多跳考卷构造 ──────────────────────────────────────────────────────────────

def make_multihop_case(case_id="MH-001", expected_steps=2, chain=None, safety=None):
    """一题多跳考卷（与 multihop_v1.json 同构的最小形态）。"""
    case = {
        "id": case_id,
        "question": "华法林和圣约翰草能一起吃吗？",
        "expected_retrieval_steps": expected_steps,
        "entity_chain": chain or [
            {"node": "herb-1", "doc_id": "HERB-015", "title": "圣约翰草概述", "role": "herb"},
            {"node": "interaction-1", "doc_id": "INTERACTION-014", "title": "圣约翰草×华法林", "role": "interaction"},
        ],
        "split": "train",
    }
    if safety is not None:
        case["safety"] = safety
    return case


def make_search_trajectory(*chunk_specs, steps=2, with_answer=True):
    """按 chunk 规格造轨迹：chunk_specs 每项 (chunk_id, title)。

    检索次数 = steps（steps 视图），检索结果 = 每次一条 payload 合并。
    """
    payloads = [
        make_search_payload(*[make_result_item(cid, title) for cid, title in chunk_specs])
    ] if chunk_specs else [make_search_payload()]
    messages = make_messages(*payloads)
    if not with_answer:
        messages = messages[:-1]  # 去掉 final 消息
    step_list = [{
        "step_id": i + 1, "agent_type": "medication", "llm_latency_ms": 100.0,
        "tool_calls": [{"tool_name": "search_knowledge_base",
                        "args": {"query": f"第{i + 1}跳"}, "tool_use_id": f"toolu_{i}"}],
        "tool_results": [{"tool_name": "search_knowledge_base", "success": True,
                          "result_success": True, "latency_ms": 300.0, "cached": False,
                          "reranked": True, "rerank_degraded": False, "error": ""}],
    } for i in range(steps)]
    return Trajectory.from_dict(make_trajectory(messages=messages, steps=step_list))


# ── 多跳：实体链覆盖 ──────────────────────────────────────────────────────────

def test_chain_full_hit_no_safety_succeeds():
    """常规档（无 safety）：链全命中即成功，方向不判（None）。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_multihop_case()))
    assert isinstance(result, MultihopResult)
    assert result.entity_chain_hit is True
    assert result.chain_hit == 2
    assert result.success is True
    assert result.direction is None


def test_chain_partial_miss_lists_missing_in_reason():
    """一个节点没检索到：失败且 failure_reason 点名未命中的节点。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"))
    result = asyncio.run(score_multihop(traj, make_multihop_case()))
    assert result.entity_chain_hit is False
    assert result.success is False
    assert "圣约翰草×华法林" in result.failure_reason


def test_chain_hit_by_title_fallback():
    """chunk id 漂移但标题命中：标题兜底，链仍算全命中。"""
    traj = make_search_trajectory(
        ("new-chunk-1", "圣约翰草概述"),
        ("new-chunk-2", "圣约翰草×华法林"),
    )
    result = asyncio.run(score_multihop(traj, make_multihop_case()))
    assert result.entity_chain_hit is True


def test_ordered_true_when_sequence_matches_chain():
    """检索顺序与链顺序一致 → ordered=True。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_multihop_case()))
    assert result.ordered is True


def test_ordered_false_when_sequence_reversed():
    """检索顺序与链顺序相反（链全命中但乱序）→ ordered=False。"""
    traj = make_search_trajectory(("INTERACTION-014", "圣约翰草×华法林"), ("HERB-015", "圣约翰草概述"))
    result = asyncio.run(score_multihop(traj, make_multihop_case()))
    assert result.entity_chain_hit is True
    assert result.ordered is False


def test_ordered_with_title_fallback_positions():
    """标题兜底命中的节点也参与有序投影。"""
    traj = make_search_trajectory(("new-1", "圣约翰草概述"), ("new-2", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_multihop_case()))
    assert result.ordered is True


# ── 多跳：步数与捷径 ──────────────────────────────────────────────────────────

def test_actual_steps_counts_search_calls():
    """步数 = steps 视图里 search_knowledge_base 的调用数。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"), steps=3)
    result = asyncio.run(score_multihop(traj, make_multihop_case(expected_steps=3)))
    assert result.actual_steps == 3


def test_understep_diagnostic_bucket():
    """比预期步数少就解决（1 步拿到 2 节点）：understep=True。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"), steps=1)
    result = asyncio.run(score_multihop(traj, make_multihop_case(expected_steps=2)))
    assert result.understep is True
    assert result.actual_steps == 1


def test_understep_requires_full_chain():
    """少步但链没全命中不算捷径（只是没搜够）。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), steps=1)
    result = asyncio.run(score_multihop(traj, make_multihop_case(expected_steps=2)))
    assert result.understep is False


# ── 多跳：结论方向（安全档）───────────────────────────────────────────────────

def make_safety_case():
    """安全档考卷：带 safety 字段。"""
    return make_multihop_case(safety={
        "drug_a": "圣约翰草", "drug_b": "华法林",
        "expected_direction": "avoid", "severity": "major",
        "evidence_level": "guideline_label",
    })


def test_safety_direction_correct_succeeds():
    """安全档：链全命中 + 方向 correct_avoid → success。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_safety_case(), direction_judge=judge_correct))
    assert result.direction == "correct_avoid"
    assert result.success is True


def test_safety_wrong_direction_fails():
    """安全档：链全命中但方向相反 → success=False，failure_reason 点名。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_safety_case(), direction_judge=judge_wrong))
    assert result.success is False
    assert "结论方向错误" in result.failure_reason


def test_safety_judge_crash_marks_unclear():
    """judge 挂掉：按 unclear 计（不是 wrong，别冤枉被测系统）+ judge_failed 标记。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_safety_case(), direction_judge=judge_boom))
    assert result.direction == "unclear"
    assert result.judge_failed is True
    assert result.success is False
    assert "结论方向不明" in result.failure_reason


def test_safety_no_judge_injected_marks_unclear():
    """没注入 judge：方向按 unclear 计（保守），不假装有判卷能力。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    result = asyncio.run(score_multihop(traj, make_safety_case()))
    assert result.direction == "unclear"
    assert result.success is False


def test_safety_empty_answer_unclear_without_judge_call():
    """无最终回答：方向直接 unclear，不浪费 judge 调用。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"), with_answer=False)
    calls = []

    async def counting_judge(question, answer, case):
        calls.append(1)
        return "correct_avoid"

    result = asyncio.run(score_multihop(traj, make_safety_case(), direction_judge=counting_judge))
    assert result.direction == "unclear"
    assert calls == []


def test_safety_judge_receives_question_answer_case():
    """judge 拿到的是题目、最终回答与考卷原文（判方向的三要素）。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    received = {}

    async def spy_judge(question, answer, case):
        received["question"] = question
        received["answer"] = answer
        received["case"] = case
        return "correct_avoid"

    asyncio.run(score_multihop(traj, make_safety_case(), direction_judge=spy_judge))
    assert "华法林和圣约翰草" in received["question"]
    assert "最终结论" in received["answer"]
    assert received["case"]["safety"]["expected_direction"] == "avoid"


# ── 长会话：summary 四字段判分 ────────────────────────────────────────────────

def make_long_session_case():
    """一场长会话考卷（gold_items 三节点，与 long_session_v1.json 同构）。"""
    return {
        "id": "LS-001",
        "title": "华法林用药咨询",
        "gold_items": [
            {"item_id": "med_wafarin", "type": "medication", "label": "华法林",
             "aliases": ["华法令", "抗凝药"]},
            {"item_id": "med_ginkgo", "type": "medication", "label": "银杏叶提取物",
             "aliases": ["银杏叶", "银杏叶片"]},
            {"item_id": "sym_bleed", "type": "symptom", "label": "牙龈出血",
             "aliases": ["刷牙出血", "牙龈渗血"]},
        ],
        "turns": [],  # 判分不读剧本，只读 summary
    }


def make_summary(medications=("正在服用华法林 3mg 每日一次",), symptoms=("牙龈渗血三天",)):
    """构造四字段摘要 JSON（与 conversation_memory 落盘形态一致）。"""
    return json.dumps({
        "allergies": [],
        "medications": list(medications),
        "symptoms": list(symptoms),
        "history": [],
    }, ensure_ascii=False)


def test_long_session_exact_label_match():
    """label 归一化子串命中：华法林/牙龈出血直接命中，无 judge 也全中。"""
    result = asyncio.run(score_long_session(make_summary(), make_long_session_case()))
    assert isinstance(result, LongSessionResult)
    assert result.total == 3
    assert result.hit == 2  # 华法林 + 牙龈出血
    assert result.judge_hit == 0
    assert result.recall == 0.6667
    assert result.loss_rate == 0.3333


def test_long_session_alias_match():
    """别名命中：摘要写「华法令」/「刷牙出血」，label 没出现也算命中。"""
    summary = make_summary(medications=("正在服用华法令", "新加银杏叶片"), symptoms=("刷牙出血",))
    result = asyncio.run(score_long_session(summary, make_long_session_case()))
    assert result.hit == 3
    assert result.recall == 1.0
    assert result.loss_rate == 0.0


def test_long_session_normalization_ignores_punctuation_spaces():
    """归一化后再匹配：「华法林3mg/日」与「华法林 3mg 日」等价。"""
    summary = make_summary(medications=("华法林 3mg/日",))
    result = asyncio.run(score_long_session(summary, make_long_session_case()))
    assert any(i["item_id"] == "med_wafarin" and i["matched"] for i in result.items)


def test_long_session_field_isolation():
    """字段隔离：药物名只搜 medications 字段——history 里提到不算命中。"""
    summary = json.dumps({
        "allergies": [],
        "medications": [],
        "symptoms": [],
        "history": ["三年前曾用华法林抗凝治疗"],  # 出现在错字段
    }, ensure_ascii=False)
    result = asyncio.run(score_long_session(summary, make_long_session_case()))
    assert result.hit == 0
    assert result.recall == 0.0


def test_long_session_judge_rescues_miss():
    """确定性没命中 → judge 复核判「同义改写」救回。

    注意「抗凝剂」不是任何别名（别名是「抗凝药」），确保华法林确实
    走 judge 路径而不是被别名子串命中。
    """
    summary = make_summary(medications=("目前在用抗凝剂",), symptoms=("牙龈渗血三天",))
    result = asyncio.run(score_long_session(summary, make_long_session_case(), judge=long_judge_true))
    assert result.hit == 1        # 牙龈出血确定性命中（别名「牙龈渗血」）
    assert result.judge_hit == 2  # 华法林、银杏叶由 judge 救回
    assert result.recall == 1.0


def test_long_session_judge_rejects_miss():
    """judge 复核判不中 → 保持未命中。"""
    summary = make_summary(medications=("目前在用抗凝剂",))
    result = asyncio.run(score_long_session(summary, make_long_session_case(), judge=long_judge_false))
    assert result.recall == 0.3333  # 只有牙龈出血命中
    assert result.items[0]["matched"] is False


def test_long_session_judge_crash_counts_as_miss():
    """judge 挂掉：按未命中计 + judge_failed 计数（不中断整场判分）。"""
    async def judge_boom(field, field_text, item):
        raise RuntimeError("复核挂了")

    summary = make_summary(medications=(), symptoms=("牙龈渗血三天",))
    result = asyncio.run(score_long_session(summary, make_long_session_case(), judge=judge_boom))
    assert result.recall == 0.3333
    assert result.judge_failed == 2  # 华法林、银杏叶两次复核都挂了


def test_long_session_corrupt_summary_all_miss():
    """summary 不是合法四字段 JSON：节点全按未命中，judge 仍可用原始文本复核。"""
    result = asyncio.run(score_long_session("这不是JSON", make_long_session_case(), judge=long_judge_true))
    assert result.recall == 1.0  # judge 用原始文本复核全部救回
    assert result.judge_hit == 3

    no_judge = asyncio.run(score_long_session("这不是JSON", make_long_session_case()))
    assert no_judge.recall == 0.0


def test_long_session_unknown_type_skipped():
    """未知 gold_item.type：告警跳过该节点，不影响其他节点。"""
    case = make_long_session_case()
    case["gold_items"].append({"item_id": "x", "type": "whatever", "label": "奇怪"})
    result = asyncio.run(score_long_session(make_summary(), case))
    assert result.total == 3  # 未知 type 的节点不计入总数


def test_long_session_empty_gold_items():
    """gold_items 为空：total 0、recall 0（防除零）。"""
    case = {"id": "LS-X", "gold_items": []}
    result = asyncio.run(score_long_session(make_summary(), case))
    assert result.total == 0 and result.recall == 0.0


def test_to_dict_shapes():
    """两类结果都能扁平化落盘（baseline.json 的 scores 子对象）。"""
    traj = make_search_trajectory(("HERB-015", "圣约翰草概述"), ("INTERACTION-014", "圣约翰草×华法林"))
    mh = asyncio.run(score_multihop(traj, make_multihop_case()))
    d = mh.to_dict()
    assert d["chain_hit"] == "2/2" and d["success"] is True
    ls = asyncio.run(score_long_session(make_summary(), make_long_session_case()))
    assert "recall" in ls.to_dict() and len(ls.to_dict()["items"]) == 3


# ── SummaryRecallJudge（长会话复核 judge 的实现，第 7 步 CLI 组装用）────────────

def make_summary_judge(responses, model="test-model"):
    """用 test_ragas_metrics 的假客户端构造真 SummaryRecallJudge。"""
    from evaluation.multihop_scorer import SummaryRecallJudge
    from tests.test_ragas_metrics import FakeClient
    return SummaryRecallJudge(FakeClient(responses), model)


GOLD_ITEM = {"item_id": "med_wafarin", "type": "medication",
             "label": "华法林", "aliases": ["华法令", "抗凝药"]}


def test_summary_judge_matched_true():
    from tests.test_ragas_metrics import json_resp
    judge = make_summary_judge([json_resp({"matched": True, "reasoning": "第 1 条提到华法林"})])
    assert asyncio.run(judge("medications", '["目前在用抗凝剂"]', GOLD_ITEM)) is True


def test_summary_judge_matched_false():
    from tests.test_ragas_metrics import json_resp
    judge = make_summary_judge([json_resp({"matched": False, "reasoning": "没提到"})])
    assert asyncio.run(judge("medications", '["仅在用阿司匹林"]', GOLD_ITEM)) is False


def test_summary_judge_retries_on_bad_structure():
    """matched 不是布尔值 = 结构错，与调用失败同样走重试（同 _judge_json_call 约定）。"""
    from tests.test_ragas_metrics import json_resp
    judge = make_summary_judge([
        json_resp({"matched": "yes"}),                          # 结构错 → 重试
        json_resp({"matched": True, "reasoning": "ok"}),
    ])
    assert asyncio.run(judge("medications", "[]", GOLD_ITEM)) is True
    assert len(judge._client.messages.calls) == 2


def test_summary_judge_retries_on_exception():
    from tests.test_ragas_metrics import json_resp
    judge = make_summary_judge([
        RuntimeError("网络炸了"),                                # 调用失败 → 重试
        json_resp({"matched": True}),
    ])
    assert asyncio.run(judge("medications", "[]", GOLD_ITEM)) is True


def test_summary_judge_raises_after_retries():
    """重试 2 次仍失败 → 抛异常，score_long_session 按未命中计 + judge_failed 计数。"""
    judge = make_summary_judge([RuntimeError("炸"), RuntimeError("炸"), RuntimeError("炸")])
    with pytest.raises(RuntimeError):
        asyncio.run(judge("medications", "[]", GOLD_ITEM))


def test_summary_judge_prompt_contains_label_aliases_field_no_rule_id():
    """判分纪律：prompt 有字段人话/label/别名，无规则编号类答案册内容。"""
    from tests.test_ragas_metrics import json_resp
    judge = make_summary_judge([json_resp({"matched": True})])
    asyncio.run(judge("medications", '["x"]', GOLD_ITEM))
    prompt = judge._client.messages.calls[0]["messages"][0]["content"]
    assert "用药史" in prompt and "华法林" in prompt and "华法令" in prompt
    assert "DDI-" not in prompt and "rule_id" not in prompt


def test_score_long_session_judge_rescues_synonym_rewrite():
    """端到端：摘要同义改写「抗凝剂」确定性 miss → SummaryRecallJudge 救回 → recall 1.0。"""
    from tests.test_ragas_metrics import json_resp
    # 「抗凝剂」让华法林 miss；银杏叶也不在摘要里 → 两个 medication 节点都走 judge
    summary = make_summary(medications=("目前在用抗凝剂",), symptoms=("牙龈渗血三天",))
    judge = make_summary_judge([
        json_resp({"matched": True, "reasoning": "抗凝剂即华法林"}),   # 同义改写救回
        json_resp({"matched": False, "reasoning": "摘要没提银杏叶"}),  # 真丢失
    ])
    result = asyncio.run(score_long_session(summary, make_long_session_case(), judge=judge))
    assert result.hit == 1 and result.judge_hit == 1   # 牙龈渗血确定性命中
    assert result.recall == 0.6667 and result.loss_rate == 0.3333
    assert result.items[0]["via"] == "judge"   # 华法林被 judge 救回
    assert result.items[1]["via"] is None       # med_ginkgo judge 判 False → 仍算 miss
