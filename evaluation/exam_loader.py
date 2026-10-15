"""
考卷加载器（v2 阶段 1 第 2 步）。

五套评测集的 schema 定义、加载与全量校验——与 core/drug_rules.load_rules
同一风格：校验失败抛 ValueError（数据质量的门，手改 JSON 改坏会在测试里拦住）。

目录布局（data/eval/）：
    rules/                    规则库答案册（已有，加载见 core/drug_rules.py）
    exams/                    五套考卷 + MANIFEST（本模块加载；第 3 步生成流水线产出）
    held_out/held_out_ids.txt held-out 审计快照（第 3 步流水线写，双保险）
    baseline.json             运行时生成（v2 schema，第 7 步产出）
    results/<run_id>/         每次基线的产物

判分纪律（重要）：考卷本身在 data/eval/ 下，是「答案册」的一部分——
rule_ref/safety/gold_* 等字段只供判分器读；被测系统（agent 链路）的视野里
只有 message/question/turns 等题干，绝无答案字段（v2-plan 判分纪律）。

held-out：case 的 split 字段在生成流水线里按 id 的 SHA256 哈希当场分配
（~25% 落 held_out）并固化进 JSON——loader 只校验合法性，不重新分配；
阶段 2.1 捞取失败样本时硬编码拒绝 split=="held_out"。
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.drug_rules import VALID_EVIDENCE, VALID_SEVERITY, _normalize
from core.intent_recognizer import UrgencyLevel

# 默认考卷目录：evaluation/ 的上级目录下的 data/eval/exams/
DEFAULT_EXAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "exams"

# ── 通用枚举 ──────────────────────────────────────────────────────────────────

SPLIT_VALUES = ("train", "held_out")

# 意图集来源：archive 旧素材导入 / rewritten LLM 改写措辞 / extended 补生成弱类
INTENT_SOURCES = ("archive", "rewritten", "extended")

# 意图集期望标签 = IntentCategory 枚举的 19 个值（与旧素材 19 类标签逐一同名，
# 实测 identity 映射，见 LEGACY_INTENT_MAP 留档）
INTENT_VALUES = (
    "query", "complaint", "request", "greeting", "emergency",
    "medication", "appointment", "feedback", "symptom_check",
    "department_guide", "appointment_manage", "appointment_reschedule",
    "appointment_cancel", "medical_receipt", "appointment_payment_issue",
    "medication_dosage", "medication_interaction", "human_handoff", "other",
)

# 旧项目医疗评测素材（data/knowledge/_archive/medical_eval_dataset.json）的
# 19 类意图标签 → 现 IntentCategory 枚举名。identity 映射（逐一同名）留档：
# 第 3 步生成流水线导入 archive 素材时据此校验，发现旧标签改名/新增时在这里挡下。
LEGACY_INTENT_MAP: Dict[str, str] = {label: label for label in INTENT_VALUES}

# 多跳集实体链节点角色：herb 草药篇 / component 成分篇 / interaction 相互作用篇
CHAIN_ROLES = ("herb", "component", "interaction")

# 安全档期望方向（对照规则 advice 语义归类）：
#   avoid   —— 禁止/避免同服（如「不建议同用」）
#   caution —— 需谨慎、医生评估后决定（如「避免同服；确需同用须加强监测」）
#   monitor —— 可同用但需监测指标（如「可同用，建议监测 INR」）
#
# 「避免同服」类 advice 的 avoid/caution 判别规则（子 agent B 审核 DDI-077 时定案，
# 2026-10-12）：fallback 是医疗监测/医生评估（「必须同用时监测…」）→ caution；
# fallback 是患者侧措施（如「加用屏障避孕法」）或没有 fallback 的禁止/避免 → avoid。
# 例：DDI-077「避免同服；同服期间加用屏障避孕法」= avoid（患者侧措施）；
#     DDI-075/079/092「避免同服；必须同用时监测…」= caution（医疗监测）。
SAFETY_DIRECTIONS = ("avoid", "caution", "monitor")

# 长会话集 gold_item 类型：allergy 过敏史 / medication 用药史 / symptom 症状 / history 其他病史
GOLD_ITEM_TYPES = ("allergy", "medication", "symptom", "history")

# 急症集期望紧急级别 = UrgencyLevel 枚举值。判分读实测 urgency 与期望对比
# （红旗题的升级走 CRITICAL 关键词门控，是确定性的；词表匹配断言在
# scripts/validate_exams.py，本处只校验枚举合法与 expected_escalated 配对）
URGENCY_VALUES = tuple(level.name for level in UrgencyLevel)

# 长会话集每场戏的轮数范围（10-20 轮 = 真实轮 8-12 + 干扰轮 2-6）
TURN_RANGE = (10, 20)

# 剂型后缀（按长度降序，先剥长的再剥短的）：「二甲双胍片」→「二甲双胍」。
# 来历（LS-005 教训，2026-10-12）：最终轮自己点名裸名「二甲双胍」，而 gold_item
# 的 aliases 只有「二甲双胍片」——raw substring 检查漏过，考卷带着泄漏上了线。
# 此后具名检查统一走「归一化 + 剂型剥离」双路匹配（见 _named_in_text）。
_DOSAGE_FORM_SUFFIXES = (
    "薄膜衣片", "缓释胶囊", "肠溶胶囊", "缓释片", "肠溶片", "控释片", "分散片",
    "咀嚼片", "泡腾片", "舌下片", "软胶囊", "颗粒剂", "口服液", "注射液",
    "滴眼液", "喷雾剂", "混悬液", "片剂", "软膏", "乳膏", "凝胶", "贴剂",
    "糖浆", "胶囊", "滴丸", "颗粒", "片",
)


def _strip_dosage_form(name: str) -> str:
    """剥离剂型后缀：别名「二甲双胍片」→「二甲双胍」，让题干裸名也能命中。

    只对别名/题干匹配用，不改判分逻辑（判分器仍是别名全量归一化匹配）。
    """
    for suffix in _DOSAGE_FORM_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _named_in_text(text: str, term: str) -> bool:
    """term（label/alias）是否在 text 里具名出现。

    两路匹配：
    1. 原文子串——覆盖「华法林」这类无剂型后缀的别名；
    2. 归一化子串（复用 drug_rules._normalize，与判分器同口径）+ 剂型后缀剥离——
       别名「二甲双胍片」剥离后命中题干裸名「二甲双胍」（LS-005 漏洞的堵法）。

    注意 label 是完整表述句（如「正在服用华法林」），归一化后仍是长串，
    子串匹配方向决定了它只会在题干完整复述 label 时才命中——保守无副作用。
    """
    if term in text:
        return True
    n_text = _normalize(text)
    n_term = _normalize(term)
    for candidate in (n_term, _strip_dosage_form(n_term)):
        if candidate and candidate in n_text:
            return True
    return False


_AUDITED_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── schema dataclass ──────────────────────────────────────────────────────────

@dataclass
class IntentCase:
    id: str
    message: str
    expected_intent: str
    source: str                                  # archive / rewritten / extended
    split: str
    llm_failure: bool = False                    # 是否属于「LLM 路失败降级」子集（第 6 步 forced-fail 跑这批）
    audited: Optional[str] = None                # 人工审核日期 YYYY-MM-DD（第 4 步填写）


@dataclass
class QaCase:
    id: str
    kind: str                                    # single 单轮问答走 /search 直测检索链路；multi 多轮问答走 agent 链路
    question: str = ""                           # kind=single 的题干
    turns: List[str] = field(default_factory=list)      # kind=multi 的对话文本轮
    source_docs: List[str] = field(default_factory=list)  # 题目事实出处的语料文件名（「语料有据」纪律）
    gold_chunks: List[str] = field(default_factory=list)  # 金标准 chunk id = 出题依据的那一篇（单跳 MRR 判分依据；multi 可空）
    # 补标金标准（2026-10-15 rev3）：库里其他同样完整包含答案的片段 id。
    # 医院知识库有内容重叠的文档（如《科室挂号指引》与各篇《科室介绍》），出题时
    # 只标了出题依据那一篇，检索排到别的含答案片段会被误判成错。判分时
    # gold_chunks ∪ alt_gold_chunks 都算对；分两个字段存，是为了保留「哪些是事后补的」
    alt_gold_chunks: List[str] = field(default_factory=list)
    gold_answer: str = ""                        # 标准答案要点（judge 校准用）
    split: str = "train"


@dataclass
class EntityChainNode:
    node: str                                    # 链上节点引用名（如 herb-1），报告哪一环断链用
    doc_id: str                                  # 语料 doc_id（实录复制，validate_exams 断言实存）
    title: str                                   # 语料标题（实录复制）
    role: str                                    # herb / component / interaction


@dataclass
class SafetySpec:
    drug_a: str                                  # 问法中的两个实体名（validate_exams 用规则库别名归一化命中校验）
    drug_b: str
    expected_direction: str                      # avoid / caution / monitor
    severity: str                                # major / moderate / minor（复用规则库枚举）
    evidence_level: str                          # 六级证据（复用规则库枚举）


@dataclass
class MultihopCase:
    id: str
    question: str
    expected_retrieval_steps: int                # 预期最少检索步数（阶段 5 搜索完整性奖励的 ground truth）
    entity_chain: List[EntityChainNode]
    rule_ref: Optional[str] = None               # 对应规则库条目 id（判分答案册；组件跳档无规则 → None）
    safety: Optional[SafetySpec] = None          # 安全档判分标准（组件跳档无规则 → None）
    split: str = "train"


@dataclass
class GoldItem:
    item_id: str                                 # 记忆项引用名（reveals/requires 里引用）
    type: str                                    # allergy / medication / symptom / history
    label: str                                   # 记忆项的标准表述（判分对照）
    aliases: List[str]                           # 别名（判分时归一化匹配模型输出）


@dataclass
class SessionTurn:
    speaker: str                                 # user / assistant（assistant turn 是预写剧本，评测按剧本推进）
    text: str
    reveals: List[str] = field(default_factory=list)    # 本轮用户透露了哪些 gold_item（item_id）
    requires: List[str] = field(default_factory=list)   # 本轮回答需要记住哪些 gold_item 才能答对


@dataclass
class LongSessionCase:
    id: str
    title: str                                   # 剧本主题（如「华法林复诊 + 新增过敏史」）
    gold_items: List[GoldItem]
    turns: List[SessionTurn]
    split: str = "train"


@dataclass
class EmergencyCase:
    """急症集单题：一句用户消息 + 期望的升级行为。

    判分纯确定性（第 5 步判分器读轨迹/ChatResponse 的 escalated 与 urgency
    字段对比期望），不用 LLM judge——升级与否是系统状态，不是语义判断。

    两个核心指标：
      漏检率 = 红旗题（expected_escalated=True）实测未升级的比例
      误报率 = 阴性题（expected_escalated=False）实测升级的比例
    红旗题的升级走 CRITICAL 关键词门控（message 含 URGENCY_KEYWORDS 的
    CRITICAL 词 → _urgency 确定性返回 CRITICAL → 路由升级），与 LLM 意图
    无关，漏检率理论上应为 0（非 0 说明急症链路本身坏了）；阴性题的升级
    只能来自 LLM 意图误判（emergency/human_handoff），误报率测的是
    「LLM 路对吓人但非红旗的主诉会不会过度反应」。
    """
    id: str
    message: str
    expected_escalated: bool                     # 期望是否升级转人工（红旗题 True / 阴性题 False）
    expected_urgency: str                        # 期望紧急级别（UrgencyLevel 枚举值；升级题必为 CRITICAL）
    split: str = "train"
    audited: Optional[str] = None                # 人工审核日期 YYYY-MM-DD（未填合法）


@dataclass
class Manifest:
    """考卷清单：记录这批题是在哪版语料/规则上生成的（基线可复现性的根）。"""
    schema_version: str
    generated_at: str                            # 生成日期 YYYY-MM-DD
    corpus_version: str                          # 语料版本说明（如「multihop_corpus.jsonl 217 篇，弱化后 265 片段」）
    rules_version: str                           # 规则库版本说明（如「drug_interactions_v1.json 117 组」）
    exams: Dict[str, Dict[str, Any]]             # {套名: {file, count, held_out}}


# ── 通用校验 ──────────────────────────────────────────────────────────────────

def _fail(case_id: str, msg: str) -> None:
    raise ValueError(f"考卷 {case_id}: {msg}")


def _check_common(case_id: str, split: str) -> None:
    if split not in SPLIT_VALUES:
        _fail(case_id, f"split 非法: {split!r}（必须 {SPLIT_VALUES}）")


def _check_unique_ids(cases: List[Any], exam_name: str) -> None:
    seen: Set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"{exam_name} id 重复: {case.id}")
        seen.add(case.id)


def _check_audited(case_id: str, audited: Optional[str]) -> None:
    """audited 若已填必须是 YYYY-MM-DD 格式（第 4 步人工审核后填写；未填合法）。"""
    if audited is not None and not _AUDITED_PAT.match(audited):
        _fail(case_id, f"audited 格式非法: {audited!r}（应为 YYYY-MM-DD）")


# ── 加载 + 校验 ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Optional[Any] = None) -> Manifest:
    """加载考卷清单并校验结构。"""
    m = _load_json(Path(path) if path else DEFAULT_EXAMS_DIR / "MANIFEST.json")
    if not isinstance(m, dict):
        raise ValueError("MANIFEST 必须是 JSON 对象")
    for field in ("schema_version", "generated_at", "corpus_version", "rules_version", "exams"):
        if field not in m:
            raise ValueError(f"MANIFEST 缺少字段: {field}")
    if not isinstance(m["exams"], dict) or not m["exams"]:
        raise ValueError("MANIFEST 的 exams 必须是非空对象")
    for name, info in m["exams"].items():
        if name not in ("intent", "qa", "multihop", "long_session", "emergency"):
            raise ValueError(f"MANIFEST 未知考卷: {name}")
        for field in ("file", "count", "held_out"):
            if field not in info:
                raise ValueError(f"MANIFEST exams.{name} 缺少字段: {field}")
    return Manifest(
        schema_version=m["schema_version"],
        generated_at=m["generated_at"],
        corpus_version=m["corpus_version"],
        rules_version=m["rules_version"],
        exams=m["exams"],
    )


def load_intent_exam(path: Optional[Any] = None) -> List[IntentCase]:
    data = _load_json(Path(path) if path else DEFAULT_EXAMS_DIR / "intent_v1.json")
    if not isinstance(data, list):
        raise ValueError("意图集考卷必须是 JSON 数组")
    cases: List[IntentCase] = []
    for item in data:
        case_id = item.get("id", "?")
        for field in ("id", "message", "expected_intent", "source", "split"):
            if field not in item:
                _fail(case_id, f"缺少字段: {field}")
        if item["expected_intent"] not in INTENT_VALUES:
            _fail(case_id, f"expected_intent 非法: {item['expected_intent']!r}")
        if item["source"] not in INTENT_SOURCES:
            _fail(case_id, f"source 非法: {item['source']!r}")
        _check_common(case_id, item["split"])
        _check_audited(case_id, item.get("audited"))
        cases.append(IntentCase(
            id=item["id"], message=item["message"], expected_intent=item["expected_intent"],
            source=item["source"], split=item["split"],
            llm_failure=bool(item.get("llm_failure", False)),
            audited=item.get("audited"),
        ))
    _check_unique_ids(cases, "意图集")
    return cases


def load_qa_exam(path: Optional[Any] = None) -> List[QaCase]:
    data = _load_json(Path(path) if path else DEFAULT_EXAMS_DIR / "qa_v1.json")
    if not isinstance(data, list):
        raise ValueError("问答集考卷必须是 JSON 数组")
    cases: List[QaCase] = []
    for item in data:
        case_id = item.get("id", "?")
        for field in ("id", "kind", "split"):
            if field not in item:
                _fail(case_id, f"缺少字段: {field}")
        kind = item["kind"]
        if kind not in ("single", "multi"):
            _fail(case_id, f"kind 非法: {kind!r}（必须 single/multi）")
        _check_common(case_id, item["split"])
        if not item.get("source_docs"):
            _fail(case_id, "source_docs 不能为空（「语料有据」纪律：题目事实必须能在语料中定位）")
        if kind == "single":
            if not item.get("question"):
                _fail(case_id, "kind=single 必须给出 question（题干）")
            if not item.get("gold_chunks"):
                _fail(case_id, "kind=single 必须给出 gold_chunks（单跳 MRR 的金标准）")
            # 补标片段与出题依据重复 = 手改考卷时写错了（同一片段不该记两次）
            overlap = set(item.get("alt_gold_chunks", [])) & set(item["gold_chunks"])
            if overlap:
                _fail(case_id, f"alt_gold_chunks 与 gold_chunks 重复: {sorted(overlap)}")
        else:  # multi
            if not item.get("turns"):
                _fail(case_id, "kind=multi 必须给出 turns（多轮对话）")
            # multi 档走 agent 链路、不判检索命中，补标金标准对它没有意义
            if item.get("alt_gold_chunks"):
                _fail(case_id, "kind=multi 不判检索命中，不应给出 alt_gold_chunks")
        cases.append(QaCase(
            id=item["id"], kind=kind,
            question=item.get("question", ""),
            turns=list(item.get("turns", [])),
            source_docs=list(item.get("source_docs", [])),
            gold_chunks=list(item.get("gold_chunks", [])),
            alt_gold_chunks=list(item.get("alt_gold_chunks", [])),
            gold_answer=item.get("gold_answer", ""),
            split=item["split"],
        ))
    _check_unique_ids(cases, "问答集")
    return cases


def load_multihop_exam(path: Optional[Any] = None) -> List[MultihopCase]:
    data = _load_json(Path(path) if path else DEFAULT_EXAMS_DIR / "multihop_v1.json")
    if not isinstance(data, list):
        raise ValueError("多跳集考卷必须是 JSON 数组")
    cases: List[MultihopCase] = []
    for item in data:
        case_id = item.get("id", "?")
        for field in ("id", "question", "expected_retrieval_steps", "entity_chain", "split"):
            if field not in item:
                _fail(case_id, f"缺少字段: {field}")
        _check_common(case_id, item["split"])
        if not isinstance(item["expected_retrieval_steps"], int) or item["expected_retrieval_steps"] < 1:
            _fail(case_id, f"expected_retrieval_steps 必须是 ≥1 的整数: {item['expected_retrieval_steps']!r}")
        chain: List[EntityChainNode] = []
        if not isinstance(item["entity_chain"], list) or not item["entity_chain"]:
            _fail(case_id, "entity_chain 必须是非空数组")
        for node in item["entity_chain"]:
            for field in ("node", "doc_id", "title", "role"):
                if field not in node:
                    _fail(case_id, f"entity_chain 节点缺少字段: {field}")
            if node["role"] not in CHAIN_ROLES:
                _fail(case_id, f"entity_chain 节点 {node.get('node', '?')} role 非法: {node['role']!r}")
            chain.append(EntityChainNode(node=node["node"], doc_id=node["doc_id"],
                                         title=node["title"], role=node["role"]))
        # rule_ref 与 safety 必须成对出现（安全档）或都不出现（组件跳档）
        has_ref = item.get("rule_ref") is not None
        has_safety = item.get("safety") is not None
        if has_ref != has_safety:
            _fail(case_id, "rule_ref 与 safety 必须成对出现（安全档）/ 或都不出现（组件跳档）")
        safety: Optional[SafetySpec] = None
        if has_safety:
            s = item["safety"]
            for field in ("drug_a", "drug_b", "expected_direction", "severity", "evidence_level"):
                if field not in s:
                    _fail(case_id, f"safety 缺少字段: {field}")
            if s["expected_direction"] not in SAFETY_DIRECTIONS:
                _fail(case_id, f"safety.expected_direction 非法: {s['expected_direction']!r}")
            if s["severity"] not in VALID_SEVERITY:
                _fail(case_id, f"safety.severity 非法: {s['severity']!r}")
            if s["evidence_level"] not in VALID_EVIDENCE:
                _fail(case_id, f"safety.evidence_level 非法: {s['evidence_level']!r}")
            safety = SafetySpec(drug_a=s["drug_a"], drug_b=s["drug_b"],
                                expected_direction=s["expected_direction"],
                                severity=s["severity"], evidence_level=s["evidence_level"])
        cases.append(MultihopCase(
            id=item["id"], question=item["question"],
            expected_retrieval_steps=item["expected_retrieval_steps"],
            entity_chain=chain,
            rule_ref=item.get("rule_ref"),
            safety=safety,
            split=item["split"],
        ))
    _check_unique_ids(cases, "多跳集")
    return cases


def load_long_session_exam(path: Optional[Any] = None) -> List[LongSessionCase]:
    data = _load_json(Path(path) if path else DEFAULT_EXAMS_DIR / "long_session_v1.json")
    if not isinstance(data, list):
        raise ValueError("长会话集考卷必须是 JSON 数组")
    cases: List[LongSessionCase] = []
    for item in data:
        case_id = item.get("id", "?")
        for field in ("id", "title", "gold_items", "turns", "split"):
            if field not in item:
                _fail(case_id, f"缺少字段: {field}")
        _check_common(case_id, item["split"])

        # gold_items：item_id 全剧本唯一；type 枚举合法；aliases 非空字符串列表
        gold_items: List[GoldItem] = []
        item_ids: Set[str] = set()
        for g in item["gold_items"]:
            for field in ("item_id", "type", "label", "aliases"):
                if field not in g:
                    _fail(case_id, f"gold_items 缺少字段: {field}")
            if g["type"] not in GOLD_ITEM_TYPES:
                _fail(case_id, f"gold_item {g.get('item_id', '?')} type 非法: {g['type']!r}")
            if not isinstance(g["aliases"], list) or not g["aliases"]:
                _fail(case_id, f"gold_item {g.get('item_id', '?')} 的 aliases 必须是非空列表")
            if g["item_id"] in item_ids:
                _fail(case_id, f"gold_item item_id 重复: {g['item_id']}")
            item_ids.add(g["item_id"])
            gold_items.append(GoldItem(item_id=g["item_id"], type=g["type"],
                                       label=g["label"], aliases=list(g["aliases"])))
        if not gold_items:
            _fail(case_id, "gold_items 不能为空")

        # turns：条数在 [10,20]；speaker 合法；reveals/requires 引用的 item_id 必须存在；
        # 标注只能落在 user 轮（assistant 台词是预写的「模拟输出」，回放不注入系统，
        # 标在上面是死标注；且 assistant 回显正是历史泄漏路径之一，锁死标注位置防漂移）
        turns: List[SessionTurn] = []
        lo, hi = TURN_RANGE
        if not isinstance(item["turns"], list) or not (lo <= len(item["turns"]) <= hi):
            _fail(case_id, f"turns 条数必须在 [{lo},{hi}]，实际 {len(item.get('turns', []))}")
        for i, t in enumerate(item["turns"]):
            for field in ("speaker", "text"):
                if field not in t:
                    _fail(case_id, f"turns[{i}] 缺少字段: {field}")
            if t["speaker"] not in ("user", "assistant"):
                _fail(case_id, f"turns[{i}] speaker 非法: {t['speaker']!r}")
            for ref in t.get("reveals", []) + t.get("requires", []):
                if ref not in item_ids:
                    _fail(case_id, f"turns[{i}] 引用了不存在的 gold_item: {ref!r}")
            if t["speaker"] == "assistant" and (t.get("reveals") or t.get("requires")):
                _fail(case_id, f"turns[{i}] 是 assistant 轮，不得标注 reveals/requires"
                               f"（剧本回放只注入 user 台词，assistant 台词上的标注是死标注）")
            turns.append(SessionTurn(speaker=t["speaker"], text=t["text"],
                                     reveals=list(t.get("reveals", [])),
                                     requires=list(t.get("requires", []))))

        # requires 只能出现在最终轮：中间轮的 requires 是泄漏源——2026-10-12 全库
        # 排查发现 16/16 场的「串讲轮」把 requires 项全部点名、落在 5 消息窗口内，
        # 最终轮变成抄写题。重排后中间轮 requires 一律清空，此处锁死形态。
        for i, t in enumerate(turns[:-1]):
            if t.requires:
                _fail(case_id, f"turns[{i}] 标注了 requires（只能标注在最终轮）："
                               f"中间轮点名记忆项会把答案泄漏进 5 消息窗口，最终轮变抄写题")

        # 剧情完整性：每个 gold_item 至少被 reveal 一次；最终轮 requires 必须覆盖
        # 全部 gold_items（集合相等——漏标一项，该项就进不了 recall 分母，丢失率失真）
        revealed: Set[str] = set()
        for t in turns:
            revealed.update(t.reveals)
        for gid in item_ids:
            if gid not in revealed:
                _fail(case_id, f"gold_item {gid} 从未被任何 turn reveal（剧情不完整）")
        if not turns[-1].requires:
            _fail(case_id, "最终轮 requires 不能为空（必须构造信息差：最后一题依赖历史记忆）")
        if set(turns[-1].requires) != item_ids:
            _fail(case_id,
                  f"最终轮 requires 必须覆盖全部 gold_items"
                  f"（实际 {sorted(turns[-1].requires)}，全集 {sorted(item_ids)}）"
                  f"——漏标项进不了 recall 分母，丢失率失真")

        # 信息差强度：最终轮题干不得具名 requires 项的 label/aliases——
        # 「银杏叶保健品是不是先别吃了？」这类实体具名的确认问，不看历史也能答，
        # 测的是浅记忆；「该怎么跟医生说」型（不具名）才真正依赖记忆提取。
        # 判定 = 原文子串 + 归一化/剂型剥离双路匹配（_named_in_text，堵 LS-005 裸名漏洞）。
        by_id = {g.item_id: g for g in gold_items}
        final_text = turns[-1].text
        for gid in turns[-1].requires:
            g = by_id[gid]
            for term in [g.label] + g.aliases:
                if _named_in_text(final_text, term):
                    _fail(case_id,
                          f"最终轮题干具名了 requires 项 {gid} 的 label/别名「{term}」"
                          f"（确认问测不了记忆——把确认问降为中间轮，最终轮改成不具名的回忆型问题）")

        # 窗口泄漏：最终轮发起时系统 history = 最近 5 条消息（turns[-6:-1]）。
        # 这 5 条里任何人点名最终轮 requires 项 = 抄写题——user 轮台词直接注入
        # history（直接泄漏）；assistant 回显虽不注入，但真实系统回复大概率引用
        # 记忆库（间接泄漏）。两路都拦：窗口内任何轮点名 label/别名即报错。
        for i in range(max(0, len(turns) - 6), len(turns) - 1):
            t = turns[i]
            for gid in turns[-1].requires:
                g = by_id[gid]
                for term in [g.label] + g.aliases:
                    if _named_in_text(t.text, term):
                        _fail(case_id,
                              f"turns[{i}] 在最终轮前 5 消息窗口内点名了 {gid} 的"
                              f"label/别名「{term}」——窗口内点名 = 抄写题，"
                              f"把该轮的点名挪到窗口之外")

        # 记忆熬过窗口：最后一条 reveal 及其紧随的回显轮都必须落在 5 消息窗口外。
        # 窗口起点 = len-6，回显轮 = reveal+1（剧本 user/assistant 交替）⇒ reveal ≤ len-8。
        # 若 reveal 的回显落在窗口内，最终轮可以直接抄回显，测的不是「记忆」。
        last_reveal_idx = max((i for i, t in enumerate(turns) if t.reveals), default=-1)
        if last_reveal_idx > len(turns) - 8:
            _fail(case_id,
                  f"最后一条 reveal 在 turns[{last_reveal_idx}]，距最终轮太近："
                  f"reveal 及其回显必须都落在 5 消息窗口外（要求 index ≤ 轮数-8），"
                  f"reveal 太晚要在其后补干扰轮把它冲出窗口")

        cases.append(LongSessionCase(
            id=item["id"], title=item["title"],
            gold_items=gold_items, turns=turns, split=item["split"],
        ))
    _check_unique_ids(cases, "长会话集")
    return cases


def load_emergency_exam(path: Optional[Any] = None) -> List[EmergencyCase]:
    """急症集考卷：红旗题（期望升级）+ 阴性题（期望不升级）。"""
    data = _load_json(Path(path) if path else DEFAULT_EXAMS_DIR / "emergency_v1.json")
    if not isinstance(data, list):
        raise ValueError("急症集考卷必须是 JSON 数组")
    cases: List[EmergencyCase] = []
    for item in data:
        case_id = item.get("id", "?")
        for field in ("id", "message", "expected_escalated", "expected_urgency", "split"):
            if field not in item:
                _fail(case_id, f"缺少字段: {field}")
        escalated = item["expected_escalated"]
        urgency = item["expected_urgency"]
        if not isinstance(escalated, bool):
            _fail(case_id, f"expected_escalated 必须是布尔值: {escalated!r}")
        if urgency not in URGENCY_VALUES:
            _fail(case_id, f"expected_urgency 非法: {urgency!r}（必须 {URGENCY_VALUES}）")
        # 配对铁律：升级题（红旗）的升级走 CRITICAL 关键词门控，期望级别必为
        # CRITICAL；阴性题期望不升级，级别绝不能是 CRITICAL（写反了就是废题）
        if escalated and urgency != "CRITICAL":
            _fail(case_id, f"expected_escalated=True 时 expected_urgency 必须为 CRITICAL，实际 {urgency!r}")
        if not escalated and urgency == "CRITICAL":
            _fail(case_id, "expected_escalated=False 时 expected_urgency 不能为 CRITICAL"
                           "（期望不升级却标危重级别，自相矛盾）")
        _check_common(case_id, item["split"])
        _check_audited(case_id, item.get("audited"))
        cases.append(EmergencyCase(
            id=item["id"], message=item["message"],
            expected_escalated=escalated, expected_urgency=urgency,
            split=item["split"], audited=item.get("audited"),
        ))
    _check_unique_ids(cases, "急症集")
    return cases


def load_all_exams(exams_dir: Optional[Any] = None) -> Dict[str, List[Any]]:
    """一次性加载五套考卷（第 7 步基线 CLI 用）。任一套校验失败即抛 ValueError。

    参数语义：目录（默认 data/eval/exams/）；单个 load_* 的参数语义是文件路径。
    """
    d = Path(exams_dir) if exams_dir else DEFAULT_EXAMS_DIR
    return {
        "intent": load_intent_exam(d / "intent_v1.json"),
        "qa": load_qa_exam(d / "qa_v1.json"),
        "multihop": load_multihop_exam(d / "multihop_v1.json"),
        "long_session": load_long_session_exam(d / "long_session_v1.json"),
        "emergency": load_emergency_exam(d / "emergency_v1.json"),
    }
