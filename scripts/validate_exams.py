"""
v2 阶段 1 第 3 步：考卷确定性校验脚本。

校验分两层（与用户确认的分工）：
  - 本脚本 = 确定性层：只做机器可证的断言（schema/枚举/语料实存/别名归一化
    命中/字面重叠/held-out 占比）——LLM 会「幻觉实存」，这类检查绝不给 LLM；
  - 子 agent B = 语义层：医学事实准确性、题干歧义、答案提示、gold_answer
    质量等需要读懂的检查，产出审核报告由主 agent 修复。

断言清单（全部 error 级，任一失败退出码非 0）：
  1. schema/枚举：直接调 evaluation/exam_loader.py 的 load_all_exams()
     （loader 抛 ValueError 即拦截，与 drug_rules.load_rules 同风格的门）
  2. 语料有据（多跳集）：entity_chain 每节点 doc_id 在语料实存、role 与
     doc_type 一致、title 逐字一致；相邻节点 entities 交集非空（实体共现
     校验——线索句已弱化，多跳可达性只能靠「前篇 entities 与后篇标题/
     entities 有交集」代理验证，无交集意味着这一跳检索不可达）
  3. 语料有据（安全档）：safety.drug_a/drug_b 经规则库别名归一化命中同一条
     规则；rule_ref 指向的规则存在且两侧药物归一化匹配；severity/evidence_level
     与规则逐字一致；规则必须 ∈ hard_constraint_rules()（hard 69 组铁律）
  4. 语料有据（问答集）：source_docs 文件名实存；gold_chunks 每个 chunk id
     在 chunk_index.json 实存
  5. 字面重叠（多跳集）：question 的 5 字窗口落在链上语料正文中的比例 >60%
     即 error（题目几乎照抄语料 = 字面检索命中，测不出检索规划）
  6. held_out ≥20%：每套考卷的 held-out 占比下限
  7. MANIFEST 一致性：exams.{套}.{file,count,held_out} 与五套 JSON 实测一致
  8. 急症集确定性一致：红旗题必须含 CRITICAL 词（urgency 门控确定性升级）、
     阴性题必须不含 CRITICAL 词与转人工关键词（否则必升级=废题）、
     expected_urgency 必须与词表确定性结果一致（判分期望可达）

expected_direction 与规则 advice 的语义一致性不做脆启发式——advice→
三分类是语义判定，由出题者实录推导 + 子 agent B 交叉审核两层把关。

用法：
    .venv/Scripts/python.exe scripts/validate_exams.py            # 校验正式考卷
    .venv/Scripts/python.exe scripts/validate_exams.py --report txt  # 报告输出位置（默认 _archive/）
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.drug_rules import _normalize, find_interactions, hard_constraint_rules, load_rules  # noqa: E402
from core.intent_recognizer import HUMAN_HANDOFF_KEYWORDS, URGENCY_KEYWORDS  # noqa: E402
from evaluation.exam_loader import load_all_exams, load_manifest  # noqa: E402

EXAMS_DIR = ROOT / "data" / "eval" / "exams"
RULES_FILE = ROOT / "data" / "eval" / "rules" / "drug_interactions_v1.json"
CHUNK_INDEX_FILE = ROOT / "data" / "eval" / "chunk_index.json"
CORPUS_FILE = ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"
HOSPITAL_DIR = ROOT / "data" / "knowledge" / "hospital"
# 旧意图素材（意图集 archive/rewritten 的原文来源）。_archive 不进公开仓库，
# 文件缺失时 rewritten 断言跳过（生成环境必有，公开仓库跑 validate 不挡路）。
LEGACY_INTENT_FILE = ROOT / "data" / "knowledge" / "_archive" / "medical_eval_dataset.json"
DEFAULT_REPORT = ROOT / "data" / "knowledge" / "_archive" / "exam_validate_report.txt"

HELD_OUT_MIN_RATIO = 0.20      # held-out 占比下限（考卷数据分区纪律）
OVERLAP_MAX_RATIO = 0.60       # 题干 vs 语料字面重叠上限（防字面检索命中）

# MANIFEST 里每套考卷必须指向的文件名（load_all_exams 的固定约定）
EXAM_FILES = {
    "intent": "intent_v1.json",
    "qa": "qa_v1.json",
    "multihop": "multihop_v1.json",
    "long_session": "long_session_v1.json",
    "emergency": "emergency_v1.json",
}


def _load_corpus() -> dict:
    """多跳语料按 doc_id 索引（{doc_id: {title, doc_type, entities, content}}）。"""
    corpus = {}
    for line in CORPUS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        corpus[row["doc_id"]] = row
    return corpus


def _load_chunk_index() -> dict:
    return json.loads(CHUNK_INDEX_FILE.read_text(encoding="utf-8"))


def _hospital_filenames() -> set:
    return {p.name for p in HOSPITAL_DIR.glob("*.json")}


def _find_rule_by_ref(rule_ref: str, rules: list) -> dict:
    for r in rules:
        if r["id"] == rule_ref:
            return r
    return None


def _check_multihop(cases: list, corpus: dict, rules: list, problems: List[str]) -> None:
    """断言 2/3/5：实体链语料实存、安全档规则命中、字面重叠。"""
    hard_ids = {r["id"] for r in hard_constraint_rules(rules)}
    for c in cases:
        where = f"多跳集 {c.id}"
        # ── 实体链每节点语料实存 + role/doc_type/title 一致 ──
        for node in c.entity_chain:
            doc = corpus.get(node.doc_id)
            if doc is None:
                problems.append(f"{where}: entity_chain 节点 {node.node} 引用了不存在的 doc_id {node.doc_id}")
                continue
            if doc["doc_type"] != node.role:
                problems.append(
                    f"{where}: 节点 {node.node} 的 role={node.role} 与语料 doc_type={doc['doc_type']} 不一致")
            if doc["title"] != node.title:
                problems.append(
                    f"{where}: 节点 {node.node} 的 title 与语料不一致: 考卷「{node.title}」vs 语料「{doc['title']}」")

        # ── 实体共现校验（线索句弱化后链闭合的机器可验证代理）──
        # 链上相邻节点必须在实体层面相连：前一篇的 entities 与后一篇的
        # entities∪{标题} 有交集（归一化后比较，容忍别名变体），否则模型即使
        # 检索到前一篇，也找不到跳向后一篇的实体线索（「链」检索不可达）。
        # 例：HERB-001 entities 含「5-羟色氨酸」→ INTERACTION-001 标题「5-羟
        # 色氨酸×SSRI类抗抑郁药」→ 交集非空，这一跳可达。
        docs_in_chain = [corpus[n.doc_id] for n in c.entity_chain if n.doc_id in corpus]
        for a, b in zip(docs_in_chain, docs_in_chain[1:]):
            a_terms = {_normalize(t) for t in a.get("entities", [])}
            b_terms = {_normalize(t) for t in b.get("entities", [])} | {_normalize(b["title"])}
            if not (a_terms & b_terms):
                problems.append(
                    f"{where}: 链上 {a['doc_id']} → {b['doc_id']} 实体共现为空"
                    f"（前篇 entities={a.get('entities', [])}，后篇标题/entities={b['title']}"
                    f"——线索句已弱化，无实体交集意味着这一跳检索不可达）")

        # ── 安全档：规则库别名归一化命中 + 等级一致 + hard 铁律 ──
        if c.safety is not None:
            s = c.safety
            matched = find_interactions(s.drug_a, s.drug_b, rules)
            if not matched:
                problems.append(
                    f"{where}: safety 药物对 {s.drug_a}×{s.drug_b} 在规则库中未命中任何规则（别名归一化失败）")
                continue
            ref_rule = _find_rule_by_ref(c.rule_ref, rules)
            if ref_rule is None:
                problems.append(f"{where}: rule_ref {c.rule_ref} 在规则库中不存在")
            elif c.rule_ref not in {r["id"] for r in matched}:
                problems.append(
                    f"{where}: rule_ref {c.rule_ref} 与 safety 药物对 {s.drug_a}×{s.drug_b} 不匹配"
                    f"（该药物对命中的规则是 {[r['id'] for r in matched]}）")
            if ref_rule is not None:
                for field in ("severity", "evidence_level"):
                    if getattr(s, field) != ref_rule[field]:
                        problems.append(
                            f"{where}: safety.{field}={getattr(s, field)} 与规则 {c.rule_ref} "
                            f"的 {ref_rule[field]} 不一致")
                if c.rule_ref not in hard_ids:
                    problems.append(
                        f"{where}: 安全档题必须出自 hard 约束规则（{c.rule_ref} 不在 hard 69 组内）")

        # ── 字面重叠：5 字窗口落在链上语料正文中的比例（防字面检索命中）──
        overlap = _literal_overlap(c.question, [corpus[n.doc_id]["content"] for n in c.entity_chain
                                                if n.doc_id in corpus])
        if overlap > OVERLAP_MAX_RATIO:
            problems.append(
                f"{where}: 题干与链上语料的字面重叠率 {overlap:.0%} > {OVERLAP_MAX_RATIO:.0%}"
                f"（题目接近照抄语料，字面检索即可命中——请改写问法）")


def _literal_overlap(question: str, contents: List[str]) -> float:
    """question 的 5 字窗口中有多大比例出现在任一语料正文中。

    用字符窗口而非词窗口：中文无空格分词，5 字窗口对「连续照抄」敏感，
    对正常措辞（窗口跨句/换字）几乎不命中，天然容忍合法改写。
    """
    if len(question) < 5:
        return 0.0
    corpus_text = "\n".join(contents)
    windows = [question[i:i + 5] for i in range(len(question) - 4)]
    hit = sum(1 for w in windows if w in corpus_text)
    return hit / len(windows)


def _check_qa(cases: list, chunk_index: dict, problems: List[str]) -> None:
    """断言 4：source_docs 文件实存 + gold_chunks 每 id 在 chunk_index 实存。"""
    hospital_files = _hospital_filenames()
    all_chunk_ids = {ch["id"] for doc in chunk_index["docs"].values() for ch in doc["chunks"]}
    for c in cases:
        where = f"问答集 {c.id}"
        for src in c.source_docs:
            if src not in hospital_files:
                problems.append(f"{where}: source_docs 引用了不存在的 hospital 文件 {src}")
        for cid in c.gold_chunks:
            if cid not in all_chunk_ids:
                problems.append(f"{where}: gold_chunks 引用了 chunk_index 中不存在的 chunk id {cid}")


def _check_intent_rewritten(cases: list, problems: List[str]) -> None:
    """断言：rewritten 档的 message 不得是 legacy 原文（改写必须生效）。

    防回归：流水线初版 bug 把「source 标 rewritten、message 却取原文」的
    47 条混进了考卷（子 agent B 逐条比对抓到）。这道断言是第二道防线——
    改写缓存丢/取错字段都会在这里拦下，而不是污染意图准确率数字。
    legacy 素材文件缺失（公开仓库无 _archive）时跳过。
    """
    if not LEGACY_INTENT_FILE.exists():
        return
    legacy = json.loads(LEGACY_INTENT_FILE.read_text(encoding="utf-8"))
    legacy_msgs = {c["message"] for c in legacy["intent_cases"]}
    for c in cases:
        if c.source == "rewritten" and c.message in legacy_msgs:
            problems.append(
                f"意图集 {c.id}: source=rewritten 但 message 与 legacy 原文逐字相同"
                f"「{c.message[:30]}...」（改写未生效，流水线取错了字段）")


def _deterministic_urgency(message: str) -> str:
    """消息按 URGENCY_KEYWORDS 词表的确定性匹配结果（与 recognizer._urgency 的
    词表段同序同口径）。

    刻意不模拟 intent 补偿段（EMERGENCY/HUMAN_HANDOFF intent 会把无词表命中
    的消息抬到 HIGH）：那是 LLM/embedding 波动项，不在题目可锁定的范围内。
    考卷 expected_urgency 写的是「词表确定性部分」的期望——红旗题含 CRITICAL
    词，实测必 CRITICAL（词表匹配优先于 intent，与 LLM 无关）；阴性题判分
    主看 escalated，urgency 期望只作诊断信息。
    """
    msg = message.lower()
    for level, kws in URGENCY_KEYWORDS.items():
        if any(kw in msg for kw in kws):
            return level.name
    return "LOW"


def _check_emergency(cases: list, problems: List[str]) -> None:
    """断言 8：急症集期望与系统确定性行为一致（红旗词匹配 + urgency 期望可达）。

    升级门控（agent_orchestrator._route_decision）：urgency==CRITICAL 或
    intent∈{EMERGENCY, HUMAN_HANDOFF} 才升级转人工。其中 urgency 的词表
    匹配是纯消息函数（词表优先于 intent）、human_handoff 的关键词路也
    是纯消息函数，两条链路机器可证：
      - 红旗题（escalated=True）不含 CRITICAL 词 = urgency 门控不会触发、
        升级只能靠 LLM 判 emergency（波动）= 漏检率失去确定性保证 → 废题；
      - 阴性题（escalated=False）含 CRITICAL 词 = 系统必升级 = 期望恒失败 → 废题；
      - 阴性题含转人工关键词 = 关键词路必判 human_handoff、LLM/embedding
        路几乎必随 → 升级接近必然 = 废题。
    expected_urgency 必须与词表确定性结果一致——消息写「持续高烧」却标
    LOW 这类手误在这里拦下，保证判分对比 urgency 时期望值可达。
    """
    for c in cases:
        where = f"急症集 {c.id}"
        det = _deterministic_urgency(c.message)
        if c.expected_escalated and det != "CRITICAL":
            problems.append(
                f"{where}: 红旗题消息不含任何 CRITICAL 词（词表确定性结果={det}），"
                f"urgency 门控不会升级，漏检率失去确定性保证。消息:「{c.message}」")
        if not c.expected_escalated:
            if det == "CRITICAL":
                problems.append(
                    f"{where}: 阴性题消息含 CRITICAL 词，系统确定性升级，"
                    f"expected_escalated=False 恒失败。消息:「{c.message}」")
            if any(kw in c.message for kw in HUMAN_HANDOFF_KEYWORDS):
                problems.append(
                    f"{where}: 阴性题消息含转人工关键词，意图路必判 human_handoff 升级，"
                    f"expected_escalated=False 恒失败。消息:「{c.message}」")
        if c.expected_urgency != det:
            problems.append(
                f"{where}: expected_urgency={c.expected_urgency} 与词表确定性结果 {det} 不一致"
                f"（消息:「{c.message}」）")


def _check_held_out(exam_name: str, cases: list, problems: List[str]) -> None:
    """断言 6：每套考卷 held-out ≥20%（数据分区纪律的下限）。"""
    held = sum(1 for c in cases if c.split == "held_out")
    ratio = held / len(cases) if cases else 0
    if ratio < HELD_OUT_MIN_RATIO:
        problems.append(
            f"{exam_name}: held_out 占比 {ratio:.1%} < {HELD_OUT_MIN_RATIO:.0%}（held {held}/{len(cases)}）")


def _check_manifest(manifest, loaded: dict, problems: List[str]) -> None:
    """断言 7：MANIFEST 的 file/count/held_out 与四套 JSON 实测一致。"""
    for name, info in manifest.exams.items():
        if info["file"] != EXAM_FILES[name]:
            problems.append(f"MANIFEST.exams.{name}.file={info['file']} 与约定 {EXAM_FILES[name]} 不一致")
        cases = loaded[name]
        held = sum(1 for c in cases if c.split == "held_out")
        if info["count"] != len(cases):
            problems.append(
                f"MANIFEST.exams.{name}.count={info['count']} 与实测 {len(cases)} 不一致")
        if info["held_out"] != held:
            problems.append(
                f"MANIFEST.exams.{name}.held_out={info['held_out']} 与实测 {held} 不一致")


def validate() -> Tuple[int, List[str]]:
    """跑全部确定性断言。返回 (error 数, 报告行列表)。"""
    problems: List[str] = []
    lines: List[str] = []

    # 断言 1：loader 全量 schema 校验（抛 ValueError 即在这里拦下，不进 problems 列表）
    lines.append("== 考卷 schema 校验 ==")
    try:
        loaded = load_all_exams(EXAMS_DIR)
    except ValueError as e:
        lines.append(f"schema 校验失败: {e}")
        return 1, lines
    except FileNotFoundError as e:
        # 流水线中途自检的合法状态：某套考卷还没落盘（出题进行中）。
        # 转成可读报告而不是 traceback，定稿时四套齐了自然消失。
        lines.append(f"考卷文件缺失: {e}")
        return 1, lines
    for name, cases in loaded.items():
        held = sum(1 for c in cases if c.split == "held_out")
        lines.append(f"{name}: {len(cases)} 条（held_out {held}）")

    # 断言 2-7
    corpus = _load_corpus()
    chunk_index = _load_chunk_index()
    rules = load_rules(RULES_FILE)

    _check_multihop(loaded["multihop"], corpus, rules, problems)
    _check_qa(loaded["qa"], chunk_index, problems)
    _check_intent_rewritten(loaded["intent"], problems)
    _check_emergency(loaded["emergency"], problems)
    for name in loaded:
        _check_held_out(name, loaded[name], problems)

    try:
        manifest = load_manifest(EXAMS_DIR / "MANIFEST.json")
        _check_manifest(manifest, loaded, problems)
    except (ValueError, FileNotFoundError) as e:
        problems.append(f"MANIFEST 校验失败: {e}")

    lines.append("")
    lines.append("== 语料有据 / 分区 / 一致性断言 ==")
    if problems:
        lines.append(f"发现 error {len(problems)} 个:")
        lines += [f"  - {p}" for p in problems]
    else:
        lines.append("全部通过 ✓")
    return len(problems), lines


def main() -> None:
    parser = argparse.ArgumentParser(description="考卷确定性校验（断言全部 error 级，任一失败退出码 1）")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="报告输出位置（UTF-8，默认 _archive/）")
    args = parser.parse_args()

    n_errors, lines = validate()
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"validation: {n_errors} errors (report -> {Path(args.report).name})")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
