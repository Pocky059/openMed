"""药物禁忌规则库的加载与查询（v2 阶段 0.4）。

规则数据在 data/eval/rules/drug_interactions_v1.json：
    {组合, 机制, 严重度, 处置建议} + 别名/证据等级/来源 + 可选 condition/species_formulation
    （v1 完整 schema 见 v2-plan 0.4）。

定位：规则库是评测的「评分标准/答案册」（不是评测集本身），与考题同放在
data/eval/ 下；生产链路（agent 工具）不读它。

消费方（在对应阶段接入）：
    - 阶段 1 安全评测：构造禁忌场景测试集，用 find_interactions 做确定性断言；
    - 阶段 5 hard 奖励：只有 severity=major 且证据等级为 guideline_label 或
      clinical_trial 的组合才做硬约束，避免给模型引入争议组合的噪声。

本模块只做「加载 + 校验 + 查询」，不做违规判定（那是阶段 5 的事）。
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 默认路径：core/ 的上级目录下的 data/eval/rules/
DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "rules" / "drug_interactions_v1.json"

# 合法枚举值：手改 JSON 时最容易改坏的地方，加载时强制校验
VALID_SEVERITY = {"major", "moderate", "minor"}
# 证据等级六级：回答「这个相互作用的结论有多硬」，与 severity 正交
# （guideline_label 说明书/指南明确记载 > clinical_trial 人体试验 > observational
#   队列观察 > case_report 个案 > theoretical 仅机制推测；mixed = 证据互相矛盾）
VALID_EVIDENCE = {
    "guideline_label", "clinical_trial", "observational",
    "case_report", "theoretical", "mixed",
}
VALID_TYPES = {"西药", "草药/保健品"}

# 每条规则必须包含的字段（drug_a/drug_b 的子字段另查）
REQUIRED_FIELDS = (
    "id", "drug_a", "drug_b", "mechanism", "severity", "advice",
    "evidence_level", "source",
)
REQUIRED_DRUG_FIELDS = ("name", "type", "aliases")

# 可选字段（草药/保健品条目常用）：condition 限定风险成立的条件
# （如「仅 eGFR<30 时」），species_formulation 区分物种/制剂
# （如「美洲人参 Panax quinquefolius」），避免同名不同物被错误合并
OPTIONAL_FIELDS = ("condition", "species_formulation")


def _normalize(name: str) -> str:
    """归一化药物名：转小写、去掉空白与常见分隔符/括号。

    目的：让「银杏叶茶」「银杏叶 茶」「银杏叶(茶)」都命中同一条规则。
    中文名不受 lower() 影响，英文名/拼音统一成小写。
    """
    return re.sub(r"[\s\-_（）()·,，/、]", "", str(name)).lower()


def load_rules(path: Optional[Any] = None) -> List[Dict[str, Any]]:
    """加载规则库并做 schema 校验。

    校验失败抛 ValueError——这是数据质量的门：后续任何人手改 JSON
    改坏了（漏字段、写错枚举、重复组合），测试会在这里拦住。
    """
    rules_path = Path(path) if path else DEFAULT_RULES_PATH
    with open(rules_path, "r", encoding="utf-8") as f:
        rules: List[Dict[str, Any]] = json.load(f)

    seen_ids: Set[str] = set()
    seen_pairs: Set[frozenset] = set()

    for rule in rules:
        # 1. 必填字段
        for field in REQUIRED_FIELDS:
            if field not in rule:
                raise ValueError(f"规则 {rule.get('id', '?')} 缺少字段: {field}")
        for drug_key in ("drug_a", "drug_b"):
            drug = rule[drug_key]
            if not isinstance(drug, dict):
                raise ValueError(f"规则 {rule['id']} 的 {drug_key} 必须是对象")
            for field in REQUIRED_DRUG_FIELDS:
                if field not in drug:
                    raise ValueError(f"规则 {rule['id']} 的 {drug_key} 缺少字段: {field}")
            if not isinstance(drug["aliases"], list) or not all(isinstance(a, str) for a in drug["aliases"]):
                raise ValueError(f"规则 {rule['id']} 的 {drug_key}.aliases 必须是非空字符串列表")
            if not drug["aliases"]:
                raise ValueError(f"规则 {rule['id']} 的 {drug_key}.aliases 不能为空（否则别名匹配失效）")
            if drug["type"] not in VALID_TYPES:
                raise ValueError(f"规则 {rule['id']} 的 {drug_key}.type 非法: {drug['type']!r}")

        # 2. 枚举字段
        if rule["severity"] not in VALID_SEVERITY:
            raise ValueError(f"规则 {rule['id']} 的 severity 非法: {rule['severity']!r}")
        if rule["evidence_level"] not in VALID_EVIDENCE:
            raise ValueError(f"规则 {rule['id']} 的 evidence_level 非法: {rule['evidence_level']!r}")

        # 2.5 可选字段：出现就必须是非空字符串（写空串等于白写，同样拦住）
        for field in OPTIONAL_FIELDS:
            if field in rule and (not isinstance(rule[field], str) or not rule[field].strip()):
                raise ValueError(f"规则 {rule['id']} 的 {field} 必须是非空字符串")

        # 3. id 唯一
        if rule["id"] in seen_ids:
            raise ValueError(f"规则 id 重复: {rule['id']}")
        seen_ids.add(rule["id"])

        # 4. 组合无重复（规范化排序后比较，A×B 与 B×A 视为同一组合）
        pair = frozenset((
            _normalize(rule["drug_a"]["name"]),
            _normalize(rule["drug_b"]["name"]),
        ))
        if pair in seen_pairs:
            raise ValueError(f"规则 {rule['id']} 的组合与已有规则重复")
        seen_pairs.add(pair)

    return rules


def _drug_matches(drug: Dict[str, Any], query: str) -> bool:
    """查询词是否命中某个 drug 对象（正名或任一别名，归一化后比较）。"""
    q = _normalize(query)
    return _normalize(drug["name"]) == q or any(_normalize(a) == q for a in drug["aliases"])


def find_interactions(drug_a: str, drug_b: str, rules: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """查询两个药物的禁忌规则。

    - 双向匹配：「银杏叶茶 × 华法林」和「华法林 × 银杏叶茶」都会命中；
    - 别名匹配：「银杏叶茶」能命中 name 为「银杏叶提取物」的规则；
    - 未命中返回空列表（调用方把它当「无禁忌」处理）。
    """
    loaded = rules if rules is not None else load_rules()
    hits = []
    for rule in loaded:
        a_hit = _drug_matches(rule["drug_a"], drug_a) or _drug_matches(rule["drug_b"], drug_a)
        b_hit = _drug_matches(rule["drug_a"], drug_b) or _drug_matches(rule["drug_b"], drug_b)
        if a_hit and b_hit:
            hits.append(rule)
    return hits


def hard_constraint_rules(rules: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """返回可用于 hard 奖励硬约束的子集。

    标准：severity=major 且 evidence_level ∈ {guideline_label, clinical_trial}。
    理由：硬约束是训练奖励的硬性判据，只有「后果严重 + 证据够硬」才配；
    个案/理论/mixed 的组合照样要提示，但方向可能有误，判违规会放大噪声。
    """
    loaded = rules if rules is not None else load_rules()
    return [r for r in loaded
            if r["severity"] == "major"
            and r["evidence_level"] in ("guideline_label", "clinical_trial")]
