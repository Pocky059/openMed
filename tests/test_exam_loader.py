"""
v2 阶段 1 第 2 步 考卷加载器的单元测试。

覆盖三块：
1. LEGACY_INTENT_MAP 与现枚举的 identity 对应（第 3 步导入 archive 素材的依据）；
2. 五套合法样本的正常加载与字段解析；
3. schema 校验对坏数据的拦截（防止日后手改考卷 JSON 改坏）。

全部离线，不联网、不依赖模型文件。坏数据用 tmp_path 写临时 JSON
（同 test_drug_rules 惯例）。
"""
import json

import pytest

from core.intent_recognizer import IntentCategory
from evaluation.exam_loader import (
    INTENT_VALUES,
    LEGACY_INTENT_MAP,
    load_all_exams,
    load_emergency_exam,
    load_intent_exam,
    load_long_session_exam,
    load_manifest,
    load_multihop_exam,
    load_qa_exam,
)


def _write(tmp_path, name: str, data) -> str:
    """把数据写成 UTF-8 JSON 文件（ensure_ascii=False 保证中文可读），返回路径。"""
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(p)


# ── 合法样本 ──────────────────────────────────────────────────────────────────

VALID_INTENT = {
    "id": "intent-001", "message": "在吗", "expected_intent": "greeting",
    "source": "archive", "split": "train", "llm_failure": False,
    "audited": "2026-10-13",
}

VALID_QA_SINGLE = {
    "id": "qa-001", "kind": "single",
    "question": "本院皮肤科周日开门吗？",
    "source_docs": ["hospital_dermatology.md"],
    "gold_chunks": ["chunk-abc123"],
    "gold_answer": "皮肤科周日休息，工作日 8:00-17:00。",
    "split": "train",
}

VALID_QA_MULTI = {
    "id": "qa-002", "kind": "multi",
    "turns": ["我想挂皮肤科", "请问您要看什么问题？"],
    "source_docs": ["hospital_dermatology.md"],
    "gold_answer": "应先确认具体皮肤问题再推荐科室。",
    "split": "held_out",
}

VALID_MULTIHOP_SAFETY = {
    "id": "multihop-001",
    "question": "我正在服用华法林，最近朋友送了三七粉，能吃吗？",
    "expected_retrieval_steps": 2,
    "entity_chain": [
        {"node": "herb-1", "doc_id": "HERB-012", "title": "三七", "role": "herb"},
        {"node": "interaction-1", "doc_id": "INTERACTION-002", "title": "三七×华法林", "role": "interaction"},
    ],
    "rule_ref": "DDI-091",
    "safety": {
        "drug_a": "华法林", "drug_b": "三七",
        "expected_direction": "avoid", "severity": "major", "evidence_level": "clinical_trial",
    },
    "split": "train",
}

VALID_MULTIHOP_COMPONENT = {
    "id": "multihop-002",
    "question": "三七为什么会影响凝血功能？",
    "expected_retrieval_steps": 2,
    "entity_chain": [
        {"node": "herb-1", "doc_id": "HERB-012", "title": "三七", "role": "herb"},
        {"node": "component-1", "doc_id": "COMPONENT-004", "title": "三七总皂苷", "role": "component"},
    ],
    "split": "held_out",
}


def _long_session_turns():
    """10 轮剧本（TURN_RANGE 下限）：2 个记忆项各被 reveal 一次，最终轮 requires 两项。

    reveal 放在 turns[0]/turns[2]（last reveal ≤ 轮数-8 = 2），满足
    「reveal 及其回显都落在 5 消息窗口外」的距离约束；中间轮 requires 全空
    （requires 只允许出现在最终轮）。
    """
    turns = [
        {"speaker": "user" if i % 2 == 0 else "assistant", "text": f"第{i}轮", "reveals": [], "requires": []}
        for i in range(10)
    ]
    turns[0]["reveals"] = ["allergy-penicillin"]
    turns[2]["reveals"] = ["med-warfarin"]
    turns[-1] = {"speaker": "user", "text": "第9轮（最终轮）", "reveals": [], "requires": ["allergy-penicillin", "med-warfarin"]}
    return turns


VALID_LONG_SESSION = {
    "id": "long-001",
    "title": "华法林复诊 + 新增过敏史",
    "gold_items": [
        {"item_id": "allergy-penicillin", "type": "allergy", "label": "青霉素过敏", "aliases": ["青霉素", "盘尼西林"]},
        {"item_id": "med-warfarin", "type": "medication", "label": "正在服用华法林", "aliases": ["华法林"]},
    ],
    "turns": _long_session_turns(),
    "split": "train",
}

VALID_EMERGENCY_RED = {
    "id": "ESC-001", "message": "我突然胸痛，呼吸困难，怎么办？",
    "expected_escalated": True, "expected_urgency": "CRITICAL", "split": "train",
}

VALID_EMERGENCY_NEG = {
    "id": "ESC-002", "message": "感冒了，流鼻涕打喷嚏，吃什么药好？",
    "expected_escalated": False, "expected_urgency": "LOW", "split": "held_out",
}

VALID_MANIFEST = {
    "schema_version": "1.0",
    "generated_at": "2026-10-13",
    "corpus_version": "multihop_corpus.jsonl 217 篇（弱化后 265 片段）",
    "rules_version": "drug_interactions_v1.json 117 组",
    "exams": {
        "intent": {"file": "intent_v1.json", "count": 150, "held_out": 38},
        "qa": {"file": "qa_v1.json", "count": 90, "held_out": 23},
        "multihop": {"file": "multihop_v1.json", "count": 60, "held_out": 15},
        "long_session": {"file": "long_session_v1.json", "count": 18, "held_out": 5},
        "emergency": {"file": "emergency_v1.json", "count": 30, "held_out": 8},
    },
}


# ── 1. LEGACY_INTENT_MAP 与现枚举的 identity 对应 ─────────────────────────────

def test_legacy_intent_map_matches_enum():
    """旧素材 19 类标签与 IntentCategory 枚举必须逐一同名（第 3 步导入的依据）。"""
    enum_values = {c.value for c in IntentCategory}
    assert set(INTENT_VALUES) == enum_values, "INTENT_VALUES 与枚举不一致"
    assert LEGACY_INTENT_MAP == {v: v for v in enum_values}, "LEGACY_INTENT_MAP 必须是 identity 映射"
    assert len(INTENT_VALUES) == 19


# ── 2. 合法样本正常加载 ───────────────────────────────────────────────────────

def test_load_valid_intent_exam(tmp_path):
    path = _write(tmp_path, "intent_v1.json", [VALID_INTENT])
    cases = load_intent_exam(path)
    assert len(cases) == 1
    c = cases[0]
    assert c.id == "intent-001" and c.expected_intent == "greeting"
    assert c.source == "archive" and c.split == "train"
    assert c.llm_failure is False and c.audited == "2026-10-13"


def test_load_valid_qa_exam(tmp_path):
    path = _write(tmp_path, "qa_v1.json", [VALID_QA_SINGLE, VALID_QA_MULTI])
    cases = load_qa_exam(path)
    assert len(cases) == 2
    single, multi = cases
    assert single.kind == "single" and single.gold_chunks == ["chunk-abc123"]
    assert multi.kind == "multi" and len(multi.turns) == 2 and multi.split == "held_out"


def test_load_valid_multihop_exam(tmp_path):
    path = _write(tmp_path, "multihop_v1.json", [VALID_MULTIHOP_SAFETY, VALID_MULTIHOP_COMPONENT])
    cases = load_multihop_exam(path)
    assert len(cases) == 2
    safety, component = cases
    # 安全档：safety 解析完整
    assert safety.rule_ref == "DDI-091"
    assert safety.safety.drug_a == "华法林" and safety.safety.expected_direction == "avoid"
    assert safety.entity_chain[0].role == "herb"
    # 组件跳档：rule_ref/safety 均为 None（无规则对应）
    assert component.rule_ref is None and component.safety is None


def test_load_valid_long_session_exam(tmp_path):
    path = _write(tmp_path, "long_session_v1.json", [VALID_LONG_SESSION])
    cases = load_long_session_exam(path)
    assert len(cases) == 1
    c = cases[0]
    assert len(c.turns) == 10 and len(c.gold_items) == 2
    assert c.turns[-1].requires == ["allergy-penicillin", "med-warfarin"]


def test_load_valid_manifest(tmp_path):
    path = _write(tmp_path, "MANIFEST.json", VALID_MANIFEST)
    m = load_manifest(path)
    assert m.schema_version == "1.0"
    assert m.exams["multihop"]["held_out"] == 15


def test_load_valid_emergency_exam(tmp_path):
    """急症集合法样本：红旗题 + 阴性题各一条正常加载。"""
    path = _write(tmp_path, "emergency_v1.json", [VALID_EMERGENCY_RED, VALID_EMERGENCY_NEG])
    cases = load_emergency_exam(path)
    assert len(cases) == 2
    red, neg = cases
    assert red.expected_escalated is True and red.expected_urgency == "CRITICAL"
    assert neg.expected_escalated is False and neg.expected_urgency == "LOW"
    assert neg.split == "held_out"


def test_load_all_exams(tmp_path):
    """五套考卷 + MANIFEST 一次性加载（第 7 步基线 CLI 的入口）。"""
    _write(tmp_path, "intent_v1.json", [VALID_INTENT])
    _write(tmp_path, "qa_v1.json", [VALID_QA_SINGLE])
    _write(tmp_path, "multihop_v1.json", [VALID_MULTIHOP_SAFETY])
    _write(tmp_path, "long_session_v1.json", [VALID_LONG_SESSION])
    _write(tmp_path, "emergency_v1.json", [VALID_EMERGENCY_RED])
    exams = load_all_exams(tmp_path)
    assert set(exams.keys()) == {"intent", "qa", "multihop", "long_session", "emergency"}
    assert all(len(v) == 1 for v in exams.values())


# ── 3. 坏数据拦截 ─────────────────────────────────────────────────────────────

def test_manifest_missing_field_raises(tmp_path):
    bad = {k: v for k, v in VALID_MANIFEST.items() if k != "corpus_version"}
    path = _write(tmp_path, "MANIFEST.json", bad)
    with pytest.raises(ValueError, match="缺少字段: corpus_version"):
        load_manifest(path)


def test_intent_invalid_expected_intent_raises(tmp_path):
    bad = dict(VALID_INTENT, expected_intent="not_a_real_intent")
    path = _write(tmp_path, "intent_v1.json", [bad])
    with pytest.raises(ValueError, match="expected_intent 非法"):
        load_intent_exam(path)


def test_intent_invalid_source_raises(tmp_path):
    bad = dict(VALID_INTENT, source="handmade")
    path = _write(tmp_path, "intent_v1.json", [bad])
    with pytest.raises(ValueError, match="source 非法"):
        load_intent_exam(path)


def test_intent_duplicate_id_raises(tmp_path):
    path = _write(tmp_path, "intent_v1.json", [VALID_INTENT, VALID_INTENT])
    with pytest.raises(ValueError, match="id 重复"):
        load_intent_exam(path)


def test_intent_bad_audited_format_raises(tmp_path):
    bad = dict(VALID_INTENT, audited="13/10/2026")
    path = _write(tmp_path, "intent_v1.json", [bad])
    with pytest.raises(ValueError, match="audited 格式非法"):
        load_intent_exam(path)


def test_qa_single_missing_gold_chunks_raises(tmp_path):
    bad = dict(VALID_QA_SINGLE, gold_chunks=[])
    path = _write(tmp_path, "qa_v1.json", [bad])
    with pytest.raises(ValueError, match="gold_chunks"):
        load_qa_exam(path)


def test_qa_single_alt_gold_chunks_parsed(tmp_path):
    """补标的其他含答案片段（rev3）读进 alt_gold_chunks；不写该字段的老题默认空列表。"""
    good = dict(VALID_QA_SINGLE, alt_gold_chunks=["chunk-faq001"])
    path = _write(tmp_path, "qa_v1.json", [good, dict(VALID_QA_SINGLE, id="qa-003")])
    with_alt, without_alt = load_qa_exam(path)
    assert with_alt.alt_gold_chunks == ["chunk-faq001"]
    assert without_alt.alt_gold_chunks == []


def test_qa_single_alt_overlapping_gold_raises(tmp_path):
    """补标片段与出题依据重复 = 手改考卷写错了。"""
    bad = dict(VALID_QA_SINGLE, alt_gold_chunks=["chunk-abc123"])
    path = _write(tmp_path, "qa_v1.json", [bad])
    with pytest.raises(ValueError, match="alt_gold_chunks 与 gold_chunks 重复"):
        load_qa_exam(path)


def test_qa_multi_with_alt_gold_raises(tmp_path):
    """multi 档不判检索命中，带补标金标准没有意义（防字段写错题）。"""
    bad = dict(VALID_QA_MULTI, alt_gold_chunks=["chunk-faq001"])
    path = _write(tmp_path, "qa_v1.json", [bad])
    with pytest.raises(ValueError, match="kind=multi"):
        load_qa_exam(path)


def test_qa_multi_missing_turns_raises(tmp_path):
    bad = dict(VALID_QA_MULTI, turns=[])
    path = _write(tmp_path, "qa_v1.json", [bad])
    with pytest.raises(ValueError, match="turns"):
        load_qa_exam(path)


def test_qa_missing_source_docs_raises(tmp_path):
    """「语料有据」纪律：题目事实必须能定位到语料文件。"""
    bad = dict(VALID_QA_SINGLE, source_docs=[])
    path = _write(tmp_path, "qa_v1.json", [bad])
    with pytest.raises(ValueError, match="source_docs"):
        load_qa_exam(path)


def test_multihop_empty_chain_raises(tmp_path):
    bad = dict(VALID_MULTIHOP_SAFETY, entity_chain=[])
    path = _write(tmp_path, "multihop_v1.json", [bad])
    with pytest.raises(ValueError, match="entity_chain"):
        load_multihop_exam(path)


def test_multihop_invalid_chain_role_raises(tmp_path):
    bad = dict(VALID_MULTIHOP_SAFETY)
    bad["entity_chain"] = [dict(VALID_MULTIHOP_SAFETY["entity_chain"][0], role="drug")]
    path = _write(tmp_path, "multihop_v1.json", [bad])
    with pytest.raises(ValueError, match="role 非法"):
        load_multihop_exam(path)


def test_multihop_rule_ref_without_safety_raises(tmp_path):
    """rule_ref 与 safety 必须成对（安全档）或都不出现（组件跳档）。"""
    bad = dict(VALID_MULTIHOP_SAFETY)
    del bad["safety"]
    path = _write(tmp_path, "multihop_v1.json", [bad])
    with pytest.raises(ValueError, match="rule_ref 与 safety 必须成对"):
        load_multihop_exam(path)


def test_multihop_invalid_safety_enum_raises(tmp_path):
    bad = dict(VALID_MULTIHOP_SAFETY)
    bad["safety"] = dict(VALID_MULTIHOP_SAFETY["safety"], expected_direction="neutral")
    path = _write(tmp_path, "multihop_v1.json", [bad])
    with pytest.raises(ValueError, match="expected_direction 非法"):
        load_multihop_exam(path)


def test_multihop_invalid_evidence_raises(tmp_path):
    """severity/evidence_level 复用规则库枚举（drug_rules.VALID_*），非法值拦截。"""
    bad = dict(VALID_MULTIHOP_SAFETY)
    bad["safety"] = dict(VALID_MULTIHOP_SAFETY["safety"], evidence_level="very_strong")
    path = _write(tmp_path, "multihop_v1.json", [bad])
    with pytest.raises(ValueError, match="evidence_level 非法"):
        load_multihop_exam(path)


def test_long_session_turn_count_out_of_range_raises(tmp_path):
    bad = dict(VALID_LONG_SESSION, turns=_long_session_turns()[:5])
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="turns 条数"):
        load_long_session_exam(path)


def test_long_session_reveals_unknown_item_raises(tmp_path):
    turns = _long_session_turns()
    turns[2]["reveals"] = ["ghost-item"]
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="不存在的 gold_item"):
        load_long_session_exam(path)


def test_long_session_gold_never_revealed_raises(tmp_path):
    """剧情完整性：每个 gold_item 至少被 reveal 一次（否则测不出记忆）。"""
    turns = _long_session_turns()
    turns[0]["reveals"] = []   # allergy-penicillin 从此无人 reveal
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="从未被任何 turn reveal"):
        load_long_session_exam(path)


def test_long_session_final_turn_no_requires_raises(tmp_path):
    """信息差构造：最终轮必须依赖历史记忆（否则最后一题不需要历史也能答）。"""
    turns = _long_session_turns()
    turns[-1]["requires"] = []
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="最终轮 requires 不能为空"):
        load_long_session_exam(path)


def test_long_session_final_turn_names_label_raises(tmp_path):
    """信息差强度：最终轮题干具名 requires 项的 label = 确认问，测不了记忆。"""
    turns = _long_session_turns()
    turns[-1] = {"speaker": "user", "text": "我现在还能吃青霉素吗？",
                 "reveals": [], "requires": ["allergy-penicillin", "med-warfarin"]}
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="最终轮题干具名"):
        load_long_session_exam(path)


def test_long_session_final_turn_names_alias_raises(tmp_path):
    """别名同样拦截：题干写「华法林」等于具名（label 是「正在服用华法林」）。"""
    turns = _long_session_turns()
    turns[-1] = {"speaker": "user", "text": "那我这个华法林还要不要吃？",
                 "reveals": [], "requires": ["allergy-penicillin", "med-warfarin"]}
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="最终轮题干具名"):
        load_long_session_exam(path)


def test_long_session_invalid_speaker_raises(tmp_path):
    turns = _long_session_turns()
    turns[1]["speaker"] = "system"
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="speaker 非法"):
        load_long_session_exam(path)


def test_long_session_assistant_turn_annotations_raises(tmp_path):
    """标注只能落在 user 轮：回放只注入 user 台词，assistant 台词上的标注是死标注。"""
    turns = _long_session_turns()
    turns[1]["reveals"] = ["allergy-penicillin"]
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="assistant 轮"):
        load_long_session_exam(path)


def test_long_session_mid_turn_requires_raises(tmp_path):
    """requires 只能标注在最终轮：中间轮点名记忆项会把答案泄漏进 5 消息窗口。"""
    turns = _long_session_turns()
    turns[6]["requires"] = ["med-warfarin"]
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="只能标注在最终轮"):
        load_long_session_exam(path)


def test_long_session_final_requires_must_cover_all_raises(tmp_path):
    """最终轮 requires 必须覆盖全部 gold_items：漏标项进不了 recall 分母，丢失率失真。"""
    turns = _long_session_turns()
    turns[-1]["requires"] = ["allergy-penicillin"]   # 漏了 med-warfarin
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="必须覆盖全部"):
        load_long_session_exam(path)


def test_long_session_final_turn_bare_drug_name_raises(tmp_path):
    """LS-005 漏洞回归：别名带剂型后缀（「二甲双胍片」），题干写裸名（「二甲双胍」），
    raw substring 检查漏过——归一化 + 剂型剥离双路匹配必须拦住。"""
    turns = _long_session_turns()
    turns[-1] = {"speaker": "user", "text": "检查完恢复二甲双胍之前，我要先确认什么？",
                 "reveals": [], "requires": ["allergy-penicillin", "med-warfarin"]}
    # gold_items 直接构造新列表，不能改 VALID_LONG_SESSION 里的共享常量
    # （dict() 浅拷贝会带上原 gold_items list，原地改一项会污染后续测试）
    case = dict(VALID_LONG_SESSION, turns=turns,
                gold_items=[
                    {"item_id": "allergy-penicillin", "type": "allergy",
                     "label": "青霉素过敏", "aliases": ["青霉素", "盘尼西林"]},
                    {"item_id": "med-warfarin", "type": "medication",
                     "label": "正在服用二甲双胍", "aliases": ["二甲双胍片"]},
                ])
    path = _write(tmp_path, "long_session_v1.json", [case])
    with pytest.raises(ValueError, match="最终轮题干具名"):
        load_long_session_exam(path)


def test_long_session_window_leak_raises(tmp_path):
    """最终轮前 5 消息窗口（系统 history）内点名 requires 项 = 抄写题，必须拦截。"""
    turns = _long_session_turns()
    turns[7]["text"] = "我现在吃的华法林是不是要复查？"   # 窗口 index ∈ [4,8]
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="5 消息窗口内点名"):
        load_long_session_exam(path)


def test_long_session_reveal_too_close_raises(tmp_path):
    """最后一条 reveal 距最终轮太近（回显落进 5 消息窗口）必须拦截。"""
    turns = _long_session_turns()
    turns[8]["reveals"] = ["allergy-penicillin"]   # index 8 > 轮数-8 = 2
    bad = dict(VALID_LONG_SESSION, turns=turns)
    path = _write(tmp_path, "long_session_v1.json", [bad])
    with pytest.raises(ValueError, match="距最终轮太近"):
        load_long_session_exam(path)


def test_emergency_missing_field_raises(tmp_path):
    """急症集缺字段必须拦截（expected_escalated 是判分核心字段）。"""
    bad = {k: v for k, v in VALID_EMERGENCY_RED.items() if k != "expected_escalated"}
    path = _write(tmp_path, "emergency_v1.json", [bad])
    with pytest.raises(ValueError, match="缺少字段"):
        load_emergency_exam(path)


def test_emergency_non_bool_escalated_raises(tmp_path):
    """expected_escalated 必须是布尔值：字符串「true」是手改 JSON 的常见坑。"""
    bad = dict(VALID_EMERGENCY_RED, expected_escalated="true")
    path = _write(tmp_path, "emergency_v1.json", [bad])
    with pytest.raises(ValueError, match="布尔值"):
        load_emergency_exam(path)


def test_emergency_invalid_urgency_raises(tmp_path):
    """expected_urgency 必须是 UrgencyLevel 枚举值。"""
    bad = dict(VALID_EMERGENCY_RED, expected_urgency="SEVERE")
    path = _write(tmp_path, "emergency_v1.json", [bad])
    with pytest.raises(ValueError, match="expected_urgency 非法"):
        load_emergency_exam(path)


def test_emergency_red_non_critical_urgency_raises(tmp_path):
    """红旗题（升级题）的期望级别必须是 CRITICAL：升级门控只认 CRITICAL。"""
    bad = dict(VALID_EMERGENCY_RED, expected_urgency="HIGH")
    path = _write(tmp_path, "emergency_v1.json", [bad])
    with pytest.raises(ValueError, match="必须为 CRITICAL"):
        load_emergency_exam(path)


def test_emergency_neg_critical_urgency_raises(tmp_path):
    """阴性题期望不升级却标 CRITICAL：自相矛盾，必须拦截。"""
    bad = dict(VALID_EMERGENCY_NEG, expected_urgency="CRITICAL")
    path = _write(tmp_path, "emergency_v1.json", [bad])
    with pytest.raises(ValueError, match="不能为 CRITICAL"):
        load_emergency_exam(path)


def test_emergency_duplicate_id_raises(tmp_path):
    """急症集 id 重复必须拦截。"""
    dup = dict(VALID_EMERGENCY_RED, message="老人突然昏迷了，怎么叫都不醒")
    path = _write(tmp_path, "emergency_v1.json", [VALID_EMERGENCY_RED, dup])
    with pytest.raises(ValueError, match="id 重复"):
        load_emergency_exam(path)
