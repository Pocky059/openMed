"""
v2 阶段 0.4 药物禁忌规则库的单元测试。

覆盖三块：
1. 真实规则库的加载与全库健康检查（数量、枚举、无重复——数据质量的门）；
2. find_interactions 的别名/双向匹配（阶段 1 安全评测的核心断言能力）；
3. schema 校验对坏数据的拦截（防止日后手改 JSON 改坏）。

全部离线，不联网、不依赖模型文件。
"""
import json
from pathlib import Path

import pytest

from core.drug_rules import (
    DEFAULT_RULES_PATH,
    VALID_EVIDENCE,
    VALID_SEVERITY,
    find_interactions,
    hard_constraint_rules,
    load_rules,
)


# ── 真实规则库健康检查 ────────────────────────────────────────────────────────

def test_real_rules_load_successfully():
    """默认路径的规则库必须能加载（schema 校验通过即视为数据格式正确）。"""
    rules = load_rules()
    assert len(rules) >= 50, "v1 规则库规模不应低于 50 组"


def test_all_enums_valid():
    """全库扫描：severity / evidence_level / type 全部落在合法枚举内。"""
    for rule in load_rules():
        assert rule["severity"] in VALID_SEVERITY
        assert rule["evidence_level"] in VALID_EVIDENCE
        for drug_key in ("drug_a", "drug_b"):
            assert rule[drug_key]["type"] in ("西药", "草药/保健品")


def test_no_duplicate_ids_or_pairs():
    """id 与「组合」都不得重复（组合重复由 load_rules 拦截，这里再保险一层）。"""
    rules = load_rules()
    ids = [r["id"] for r in rules]
    assert len(ids) == len(set(ids)), "存在重复 id"

    def pair_key(r):
        return frozenset((r["drug_a"]["name"], r["drug_b"]["name"]))

    pairs = [pair_key(r) for r in rules]
    assert len(pairs) == len(set(pairs)), "存在重复药物组合"


def test_hard_constraint_subset_exists():
    """hard 奖励子集不能为空（阶段 5 依赖它做硬约束）。

    标准：major + 证据等级为 guideline_label 或 clinical_trial
    （个案/理论/mixed 组合也提示，但不做硬性判据）。
    """
    subset = hard_constraint_rules()
    assert len(subset) >= 20
    for rule in subset:
        assert rule["severity"] == "major"
        assert rule["evidence_level"] in ("guideline_label", "clinical_trial")


def test_new_evidence_levels_in_use():
    """六级证据等级里，新引入的级别必须实际被用到（防止迁移不彻底）。

    mixed 尤其重要：病例阳性+受控研究阴性的组合如果被塞进
    case_report 或 theoretical，会让 benchmark 金标准失真。
    """
    levels = {r["evidence_level"] for r in load_rules()}
    for level in ("mixed", "clinical_trial", "observational"):
        assert level in levels, f"evidence_level={level} 在库中未被使用"


# ── 查询：别名 / 双向匹配 ─────────────────────────────────────────────────────

def test_find_by_alias_ginkgo_tea_and_warfarin():
    """阶段 0 验收场景：「银杏叶茶」×「华法林」必须命中（别名匹配）。"""
    hits = find_interactions("银杏叶茶", "华法林")
    assert len(hits) == 1
    assert hits[0]["id"] == "DDI-071"
    # 第二轮审核修订后：受控研究总体不支持确定性出血相互作用，降为 moderate+mixed
    assert hits[0]["severity"] == "moderate"
    assert hits[0]["evidence_level"] == "mixed"


def test_find_is_bidirectional():
    """查询方向无关：A×B 和 B×A 结果一致。"""
    forward = find_interactions("华法林", "银杏叶茶")
    backward = find_interactions("银杏叶茶", "华法林")
    assert forward == backward


def test_find_by_english_alias():
    """英文别名也可命中（warfarin → 华法林规则）。"""
    hits = find_interactions("warfarin", "ginkgo biloba")
    assert any(r["id"] == "DDI-071" for r in hits)


def test_find_unknown_drug_returns_empty():
    """未收录的药物返回空列表（调用方当作「无禁忌」处理）。"""
    assert find_interactions("华法林", "某种不存在的药") == []
    assert find_interactions("不存在的药A", "不存在的药B") == []


def test_find_class_rule_via_member_alias():
    """类规则（如 NSAIDs）可通过成员别名命中：「布洛芬」×「华法林」。"""
    hits = find_interactions("布洛芬", "华法林")
    assert any(r["id"] == "DDI-002" for r in hits)


def test_find_herbal_western_pair():
    """草×西组合：「丹参片」×「华法林」应命中。"""
    hits = find_interactions("丹参片", "华法林")
    assert any(r["id"] == "DDI-089" for r in hits)


def test_advice_and_mechanism_nonempty_throughout():
    """机制与处置建议不得为空——这两项是规则库的语义核心。"""
    for rule in load_rules():
        assert rule["mechanism"].strip()
        assert rule["advice"].strip()
        assert rule["source"].strip()


# ── schema 校验对坏数据的拦截 ────────────────────────────────────────────────

def _write_temp_rules(tmp_path: Path, rules) -> Path:
    """把规则列表写成临时 JSON 文件，供坏数据测试使用。"""
    path = tmp_path / "bad_rules.json"
    path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    return path


def test_missing_field_rejected(tmp_path):
    """缺字段必须抛 ValueError。"""
    rules = load_rules()
    bad = [dict(r) for r in rules[:1]]
    del bad[0]["mechanism"]
    path = _write_temp_rules(tmp_path, bad)
    with pytest.raises(ValueError, match="mechanism"):
        load_rules(path)


def test_bad_enum_rejected(tmp_path):
    """severity 写错枚举必须抛 ValueError。"""
    rules = load_rules()
    bad = [dict(r) for r in rules[:1]]
    bad[0]["severity"] = "严重"   # 不是 major/moderate/minor
    path = _write_temp_rules(tmp_path, bad)
    with pytest.raises(ValueError, match="severity"):
        load_rules(path)


def test_empty_aliases_rejected(tmp_path):
    """aliases 为空必须抛 ValueError（否则别名匹配静默失效）。"""
    rules = load_rules()
    bad = [dict(r) for r in rules[:1]]
    bad[0]["drug_a"] = dict(bad[0]["drug_a"])
    bad[0]["drug_a"]["aliases"] = []
    path = _write_temp_rules(tmp_path, bad)
    with pytest.raises(ValueError, match="aliases"):
        load_rules(path)


def test_empty_condition_rejected(tmp_path):
    """可选字段 condition 出现时必须是非空字符串（空串等于白写，拦住）。"""
    rules = load_rules()
    bad = [dict(r) for r in rules[:1]]
    bad[0]["condition"] = "   "
    path = _write_temp_rules(tmp_path, bad)
    with pytest.raises(ValueError, match="condition"):
        load_rules(path)


def test_duplicate_pair_rejected(tmp_path):
    """同组合写两条必须抛 ValueError（load_rules 会拦）。"""
    rules = load_rules()
    dup = dict(rules[0])          # 深拷贝不够，重新构造
    dup["id"] = "DDI-999"         # id 改掉，让「组合重复」成为唯一的错误
    path = _write_temp_rules(tmp_path, rules[:1] + [dup])
    with pytest.raises(ValueError, match="重复"):
        load_rules(path)


def test_default_path_exists():
    """默认路径指向真实存在的文件（防止路径重构时静默丢失）。"""
    assert DEFAULT_RULES_PATH.is_file()
