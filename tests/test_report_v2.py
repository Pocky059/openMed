"""
evaluation/report_v2.py 的离线单测（v2 阶段 1 第 5 步配套）。

聚合函数都是确定性纯函数：造小规模结果对象喂进去，核对数字；
空输入全 0 不炸（防除零）；build_baseline 的 headline 从各集指标
抽出（同一数字不维护两份）；save/load 往返 + 版本校验。
"""
import json

import pytest

from evaluation.report_v2 import (
    SCHEMA_VERSION,
    aggregate_dialog,
    aggregate_dialog_single,
    aggregate_emergency,
    aggregate_intent,
    aggregate_long_session,
    aggregate_multihop,
    aggregate_safety,
    build_baseline,
    load_baseline,
    save_baseline,
)


# ── 意图集聚合 ────────────────────────────────────────────────────────────────

def test_aggregate_intent_accuracy_and_macro_f1():
    """两类别各一半：一个全对、一个全错 → accuracy 0.5，per_class 数字可核。"""
    pairs = [
        {"case_id": "1", "expected": "symptom_check", "actual": "symptom_check"},
        {"case_id": "2", "expected": "symptom_check", "actual": "symptom_check"},
        {"case_id": "3", "expected": "appointment", "actual": "emergency"},
        {"case_id": "4", "expected": "appointment", "actual": "symptom_check"},
    ]
    out = aggregate_intent(pairs)
    assert out["accuracy"] == 0.5
    assert out["total"] == 4
    sc = out["per_class"]["symptom_check"]
    assert sc["expected"] == 2 and sc["actual"] == 3 and sc["correct"] == 2
    assert sc["precision"] == 0.6667 and sc["recall"] == 1.0
    # appointment：recall 0、emergency：precision 0
    assert out["per_class"]["appointment"]["recall"] == 0.0
    assert out["per_class"]["emergency"]["precision"] == 0.0
    # macro_f1 = 三类 F1 均值：symptom_check 0.8、appointment 0、emergency 0
    assert out["macro_f1"] == round(0.8 / 3, 4)


def test_aggregate_intent_empty():
    """空输入：全 0，不炸。"""
    out = aggregate_intent([])
    assert out["accuracy"] == 0.0 and out["macro_f1"] == 0.0 and out["per_class"] == {}


# ── 对话（问答）集聚合 ─────────────────────────────────────────────────────────

def test_aggregate_dialog_single_means():
    """MRR / hit@1 / hit@5 逐项取均值。"""
    stats = [
        {"mrr": 1.0, "hit_at_1": 1, "hit_at_5": 1},
        {"mrr": 0.5, "hit_at_1": 0, "hit_at_5": 1},
    ]
    out = aggregate_dialog_single(stats)
    assert out["mrr"] == 0.75
    assert out["hit_at_1"] == 0.5
    assert out["hit_at_5"] == 1.0
    assert out["cases"] == 2
    assert "f1_at_5" not in out  # rev3 起 F1@5 不再统计（上限锁死在 1/3）


def test_aggregate_dialog_empty():
    assert aggregate_dialog_single([])["mrr"] == 0.0


def test_aggregate_dialog_wraps_single_hop_mrr():
    """dialog 组 = single_hop_mrr（=MRR 均值）+ 可选 judge 分均值。"""
    stats = [{"mrr": 1.0, "hit_at_1": 1, "hit_at_5": 1}]
    out = aggregate_dialog(stats, judge_scores=[3.0, 4.0])
    assert out["single_hop_mrr"] == 1.0
    assert out["avg_judge_scores"] == 3.5
    # 不传 judge 分：键不存在（legacy 兼容字段可选）
    assert "avg_judge_scores" not in aggregate_dialog(stats)


def test_aggregate_dialog_single_skipped_is_none():
    """single 档本次没跑（换被测模型时跳过）：headline 记 None 而不是 0，
    0 会被误读成「检索全挂」。"""
    out = aggregate_dialog([], judge_scores=[0.9], single_skipped=True)
    assert out["single_hop_mrr"] is None
    assert out["retrieval"] == {"skipped": True}
    assert out["avg_judge_scores"] == 0.9


# ── 多跳集聚合 ────────────────────────────────────────────────────────────────

def _mh(case_id="MH-1", success=True, chain=True, ordered=True, expected=2, actual=2):
    """造一个 MultihopResult（SimpleNamespace 同构即可，聚合只看字段）。"""
    from types import SimpleNamespace
    return SimpleNamespace(case_id=case_id, success=success, entity_chain_hit=chain,
                           ordered=ordered, expected_steps=expected,
                           actual_steps=actual, understep=False)


def test_aggregate_multihop_rates():
    """4 题：2 成功 / 3 链命中 / 2 有序。"""
    results = [
        _mh("a", success=True, chain=True, ordered=True),
        _mh("b", success=False, chain=False, ordered=False),
        _mh("c", success=True, chain=True, ordered=False),   # 链全中但乱序
        _mh("d", success=False, chain=True, ordered=True),   # 链中但方向错
    ]
    out = aggregate_multihop(results)
    assert out["success_rate"] == 0.5
    assert out["entity_chain_hit_rate"] == 0.75
    assert out["ordered_chain_rate"] == 0.5
    assert out["cases"] == 4


def test_aggregate_multihop_step_efficiency():
    """步数效率 = mean(min(1, expected/actual))；actual=0 记 0。"""
    from types import SimpleNamespace
    results = [
        SimpleNamespace(actual_steps=2, expected_steps=2, success=True,
                        entity_chain_hit=True, ordered=True, understep=False),  # 1.0
        SimpleNamespace(actual_steps=4, expected_steps=2, success=True,
                        entity_chain_hit=True, ordered=True, understep=False),  # 0.5
        SimpleNamespace(actual_steps=0, expected_steps=2, success=False,
                        entity_chain_hit=False, ordered=False, understep=False),  # 0.0
    ]
    out = aggregate_multihop(results)
    assert out["step_efficiency"] == 0.5
    assert out["avg_actual_steps"] == 2.0


def test_aggregate_multihop_empty():
    out = aggregate_multihop([])
    assert out["success_rate"] == 0.0 and out["cases"] == 0


# ── 长会话聚合 ────────────────────────────────────────────────────────────────

def test_aggregate_long_session_means():
    from types import SimpleNamespace
    results = [
        SimpleNamespace(recall=1.0, loss_rate=0.0),
        SimpleNamespace(recall=0.5, loss_rate=0.5),
    ]
    out = aggregate_long_session(results)
    assert out["recall"] == 0.75 and out["loss_rate"] == 0.25


def test_aggregate_long_session_empty():
    assert aggregate_long_session([])["loss_rate"] == 0.0


# ── 安全聚合 ──────────────────────────────────────────────────────────────────

def _safety(case_id, rule_found=True, hard=True, violation=False, severity="major"):
    from types import SimpleNamespace
    return SimpleNamespace(case_id=case_id, rule_found=rule_found,
                           hard_eligible=hard, violation=violation, severity=severity)


def test_aggregate_safety_denominator_filters():
    """违规率分母只算 hard_eligible：4 题里 2 够格 1 违规 → 0.5；
    未覆盖（规则查不到）单独算 uncovered_rate。"""
    results = [
        _safety("a", violation=True),                       # 够格违规
        _safety("b"),                                       # 够格不违规
        _safety("c", hard=False, violation=True),           # 非 hard：不进分母
        _safety("d", rule_found=False, hard=False, violation=True),  # 未覆盖：不进分母
    ]
    out = aggregate_safety(results)
    assert out["overall_violation_rate"] == 0.5
    assert out["denominator"] == 2
    assert out["violations"] == 1
    assert out["uncovered_rate"] == 0.25
    assert out["by_severity"]["major"]["cases"] == 2
    assert out["by_severity"]["major"]["violations"] == 1


def test_aggregate_safety_empty_and_zero_denominator():
    """空输入与「有题但全不够格」都返回 0 违规率（防除零）。"""
    assert aggregate_safety([])["overall_violation_rate"] == 0.0
    out = aggregate_safety([_safety("x", hard=False)])
    assert out["overall_violation_rate"] == 0.0


# ── 急症聚合 ──────────────────────────────────────────────────────────────────

def test_aggregate_emergency_rates():
    """红旗 3 题漏 1、阴性 2 题误报 1。"""
    rows = [
        {"case_id": "r1", "expected_escalated": True, "actual_escalated": True},
        {"case_id": "r2", "expected_escalated": True, "actual_escalated": True},
        {"case_id": "r3", "expected_escalated": True, "actual_escalated": False},  # 漏检
        {"case_id": "n1", "expected_escalated": False, "actual_escalated": False},
        {"case_id": "n2", "expected_escalated": False, "actual_escalated": True},  # 误报
    ]
    out = aggregate_emergency(rows)
    assert out["missed_escalation_rate"] == 0.3333
    assert out["false_escalation_rate"] == 0.5
    assert out["red_flag_cases"] == 3 and out["negative_cases"] == 2


def test_aggregate_emergency_empty():
    out = aggregate_emergency([])
    assert out["missed_escalation_rate"] == 0.0 and out["false_escalation_rate"] == 0.0


# ── baseline v2 组装与落盘 ────────────────────────────────────────────────────

def _make_parts():
    """组装 build_baseline 的各个输入（全零聚合即可，只验结构）。"""
    exam_metrics = {
        "intent": aggregate_intent([]),
        "dialog": aggregate_dialog([]),
        "multihop": aggregate_multihop([]),
        "long_session": aggregate_long_session([]),
    }
    return exam_metrics, aggregate_safety([]), aggregate_emergency([])


def test_build_baseline_schema_complete():
    """schema 关键键齐备，headline 从各集指标抽出（同一数字一份）。"""
    exams, safety, emergency = _make_parts()
    data = build_baseline(
        run_id="baseline-20261012",
        config={"model": "test", "eval_mode": True},
        exam_metrics=exams, safety=safety, emergency=emergency,
        retrieval={"trajectory_quality": {"total": 0, "violations": 0}},
        results=[],
    )
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["run_id"] == "baseline-20261012"
    assert set(data["headline"]) == {"single_hop_mrr", "multihop_success_rate",
                                     "long_session_loss_rate", "safety_violation_rate",
                                     "emergency_miss_rate"}
    assert set(data["exams"]) == {"intent", "dialog", "multihop", "long_session"}
    assert "overall_violation_rate" in data["safety"]
    assert "missed_escalation_rate" in data["emergency"]
    assert data["legacy"] == {}
    assert isinstance(data["generated_at"], str)


def test_build_baseline_headline_pulls_from_exams():
    """headline 数字与各集指标一致（改口径只改聚合函数一处）。"""
    exams, safety, emergency = _make_parts()
    exams["multihop"] = aggregate_multihop([_mh("a")])          # success_rate 1.0
    exams["long_session"] = {"recall": 1.0, "loss_rate": 0.0, "cases": 1}
    safety = {"overall_violation_rate": 0.25}
    data = build_baseline(
        run_id="r", config={}, exam_metrics=exams, safety=safety,
        emergency=emergency, retrieval={}, results=[],
    )
    assert data["headline"]["multihop_success_rate"] == 1.0
    assert data["headline"]["safety_violation_rate"] == 0.25


def test_save_and_load_roundtrip(tmp_path):
    """落盘读回一致（中文不转义）。"""
    exams, safety, emergency = _make_parts()
    data = build_baseline(
        run_id="baseline-20261012", config={"model": "深寻"},
        exam_metrics=exams, safety=safety, emergency=emergency,
        retrieval={}, results=[],
    )
    path = tmp_path / "baseline.json"
    save_baseline(path, data)
    raw = path.read_text(encoding="utf-8")
    assert "深寻" in raw  # ensure_ascii=False 生效
    loaded = load_baseline(path)
    assert loaded["run_id"] == "baseline-20261012"
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_load_baseline_rejects_wrong_version(tmp_path):
    """版本不认识：拒绝处理（防旧 schema 静默混入）。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="版本不匹配"):
        load_baseline(path)
