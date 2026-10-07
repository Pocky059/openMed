"""
v2 阶段 0.3 多跳知识语料批量扩写脚本。

流水线：herb_list_v1.json（写哪些）+ drug_interactions_v1.json（A 类转写来源）
        + multihop_seed.jsonl（few-shot 范文）→ LLM 批量扩写 → 全量校验 → 合并输出。

三类任务（对应清单的 category）：
- A 类：规则库条目全文作为输入，模型原样转写方向/严重度/证据等级；
- B 类：只给调研线索（note），模型按真实文献知识写候选相互作用篇（待人工审核升级入规则库）；
- C 类：只写草药篇（成分并入正文），明确禁止写相互作用篇。

用法（在项目根目录）：
    .venv/Scripts/python.exe scripts/build_multihop_corpus.py --category A --limit 3   # 小规模试跑
    .venv/Scripts/python.exe scripts/build_multihop_corpus.py --category B            # B 类全量
    .venv/Scripts/python.exe scripts/build_multihop_corpus.py --merge                 # 合并+全量校验
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# 脚本在 scripts/ 下运行，把项目根加进导入路径才能 import core
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.drug_rules import _normalize  # noqa: E402  复用规则库的归一化，保证匹配逻辑同源
from core.llm_utils import create_llm_client, extract_text_content  # noqa: E402

HERB_LIST = ROOT / "data" / "knowledge" / "_archive" / "herb_list_v1.json"
RULES_FILE = ROOT / "data" / "eval" / "rules" / "drug_interactions_v1.json"
SEED_FILE = ROOT / "data" / "knowledge" / "_archive" / "multihop_seed.jsonl"
CORPUS_DIR = ROOT / "data" / "knowledge" / "_archive" / "corpus"   # 每味草药一个 JSONL（便于断点续跑+人工审）
FINAL_OUT = ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"  # 合并后的全量语料
REPORT_FILE = ROOT / "data" / "knowledge" / "_archive" / "validate_report.txt"  # 校验报告（UTF-8，避免控制台乱码）

# ── 扩写铁律（与用户确认过的四条规定，逐字对应）────────────────────────────────

SYSTEM_PROMPT = """你是医学知识库撰写助手，为「多跳检索评测」撰写草药/保健品知识文档。输出严格 JSONL（每行一个 JSON 对象，不输出任何 Markdown 代码块或解释文字）。

铁律：
1. 来源纪律：输入中的规则库条目（已通过第三方审核）其 mechanism/advice/severity/evidence_level 必须原样转写，不得改写方向或升级/降级证据。规则库之外的相互作用组合：只写你有把握在权威来源（药品说明书/DailyMed/EMA HMPC/PubMed 综述）中找到支持的；找不到就写「现有证据有限」放进成分篇，不写相互作用篇。禁止编造 PMID 或其他文献编号。
2. 证据诚实：case_report/mixed 条目必须保留「个案报告」「证据不一致」等不确定性表述；theoretical 必须写「理论上/体外研究提示」。条件性风险不得写成无条件风险。
3. 拆分纪律（多跳评测的核心）：草药篇只介绍草药与成分清单，不展开西药相互作用；成分篇只讲药理作用与证据强度；相互作用篇才写具体药物组合。文末可用「见《XX》篇」线索句链接，线索句指向的标题必须真实存在于本次输出或范文语料中。
4. 每篇 300-600 字中文，客观平实，不用营销语言。

输出格式（每行一个对象）：
{"doc_id": "...", "doc_type": "herb|component|interaction", "title": "...", "entities": [词1, 词2], "source": "...", "content": "..."}
interaction 篇额外字段：rule_id（来自规则库的条目必须填 DDI 编号）、direction、severity、evidence_level。"""


def load_jsonl(path: Path):
    """读 JSONL，每行一个 JSON 对象。"""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_herbs():
    return json.loads(HERB_LIST.read_text(encoding="utf-8"))


def load_rules():
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def load_seed_text() -> str:
    """9 篇范文原文（JSONL 文本），作为 few-shot 贴进每次请求。"""
    return SEED_FILE.read_text(encoding="utf-8").strip()


def _name_matches(drug: dict, query: str) -> bool:
    """查询名是否命中规则库 drug 对象（正名或别名，归一化后比较）。

    与 core.drug_rules._drug_matches 同源，但多一个括号回退：
    清单里的「人参（亚洲种）」要能命中规则库里 name 为「人参」的条目
    （DDI-083/084）——归一化去掉括号字符后「人参亚洲种」≠「人参」，
    所以再比较一次去掉括号内容后的名字。
    """
    q = _normalize(query)
    q_alt = _normalize(re.sub(r"[（(][^）)]*[）)]", "", query))
    candidates = [_normalize(drug["name"])] + [_normalize(a) for a in drug["aliases"]]
    return q in candidates or q_alt in candidates


def rules_for_herb(name: str, rules: list) -> list:
    """找出规则库中涉及该草药的全部条目（A 类转写来源）。

    双向检查：草药可能在 drug_a 也可能在 drug_b（本库草×西条目草药都在 a 侧，
    但双向匹配是廉价保险，逻辑与 find_interactions 一致）。
    """
    return [r for r in rules
            if _name_matches(r["drug_a"], name) or _name_matches(r["drug_b"], name)]


def build_task(herb: dict, matched_rules: list) -> str:
    """按 category 生成这一味草药的扩写任务文本。"""
    name = herb["name"]
    comps = "、".join(herb["core_components"])
    cat = herb["category"]
    note = herb.get("note", "")

    if cat == "A":
        # 规则条目全文贴入：模型只需「转写」不许「改写」，方向与等级必须逐字一致
        rules_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in matched_rules)
        n = len(matched_rules)
        return f"""【任务】为「{name}」撰写多跳语料文档。输出严格 JSONL。

草药信息：category=A，核心成分：{comps}
规则库条目（已过审核，共 {n} 条，方向/严重度/证据等级必须原样转写）：
{rules_text}

要求：
1. 写 1 篇 doc_type="herb" 的草药概述（介绍来源/制剂/成分清单，不展开西药相互作用）；
2. 写 1 篇 doc_type="component" 的成分药理（写核心成分中影响药物相互作用的主要成分）；
3. 每条规则库条目转写 1 篇 doc_type="interaction"，共 {n} 篇：rule_id 填对应 DDI 编号，severity/evidence_level 与规则库逐字一致，direction 用一句话概括该条目的风险方向；
4. 文末线索句用《标题》格式，指向的标题必须真实存在（本次输出或范文）。"""
    elif cat == "B":
        return f"""【任务】为「{name}」撰写多跳语料文档。输出严格 JSONL。

草药信息：category=B，核心成分：{comps}
调研线索：{note}（只给方向，结论需有把握才写）

要求：
1. 写 1 篇 doc_type="herb" 的草药概述 + 1 篇 doc_type="component" 的成分药理；
2. doc_type="interaction" 篇：只有你有把握找到权威来源支持的组合才写（0~2 篇），写出的每条必须带 source 与 evidence_level，禁止编造文献编号；
3. 证据不足的组合不写相互作用篇，可在成分篇里以「现有证据有限」如实说明。"""
    else:  # C 类：纯背景，禁止相互作用篇
        return f"""【任务】为「{name}」撰写草药文档。输出严格 JSONL。

草药信息：category=C，核心成分：{comps}

注意：本味药药物相互作用证据少，只写 1 篇 doc_type="herb" 的草药概述（成分信息并入正文介绍，不单独写成分篇），禁止写 doc_type="interaction" 文档。"""


def parse_entries(text: str):
    """把 LLM 输出解析成文档列表。

    返回 (entries, error)：解析失败时 entries 为空、error 有信息，
    调用方拿 error 重试（把错误反馈给模型让它修正）。
    """
    # 剥掉可能出现的 ```json / ``` 围栏（prompt 已禁止，但容错）
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            return [], f"第 {len(entries)+1} 行 JSON 解析失败: {e}（该行: {line[:60]}...）"
    if not entries:
        return [], "输出为空"
    return entries, None


async def generate_one(client, model: str, herb: dict, matched_rules: list, seed_text: str, max_retries: int = 2):
    """单味草药的扩写：请求 → 解析 → 失败带错误重试。"""
    task = build_task(herb, matched_rules)
    messages = [
        {"role": "user", "content": f"以下是三种文档类型的范文（JSONL，格式与调性照此）：\n{seed_text}\n\n{task}"},
    ]
    for attempt in range(max_retries + 1):
        resp = await client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=0.5,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = extract_text_content(resp.content)
        entries, err = parse_entries(text)
        if err is None:
            return entries
        # 重试：把错误信息作为新消息追加，让模型看到并修正
        messages.append({"role": "assistant", "content": text[:2000]})
        messages.append({"role": "user", "content": f"解析失败：{err}。请重新只输出 JSONL，每行一个 JSON 对象，不要任何其他文字。"})
        print(f"  retry {attempt+1}/{max_retries} for {herb['name']}: {err}")
    raise RuntimeError(f"{herb['name']} 扩写失败（重试 {max_retries} 次后仍无法解析）")


# ── 全量校验（合并后跑；报告写 UTF-8 文件）────────────────────────────────────

REQUIRED_FIELDS = ("doc_id", "doc_type", "title", "entities", "source", "content")
INTERACTION_FIELDS = ("direction", "severity", "evidence_level")


def validate_corpus(entries: list, herbs: list, rules: list) -> list:
    """返回问题列表；空列表 = 全部通过。"""
    problems = []
    seen_ids = set()

    for i, e in enumerate(entries):
        where = f"[第{i+1}篇 {e.get('doc_id', '?')}]"
        for f in REQUIRED_FIELDS:
            if f not in e:
                problems.append(f"{where} 缺字段 {f}")
        if e.get("doc_type") not in ("herb", "component", "interaction"):
            problems.append(f"{where} doc_type 非法: {e.get('doc_type')!r}")
        if e.get("doc_id") in seen_ids:
            problems.append(f"{where} doc_id 重复: {e['doc_id']}")
        seen_ids.add(e.get("doc_id"))
        if not e.get("entities") or not e.get("content"):
            problems.append(f"{where} entities/content 不能为空")
        if e.get("doc_type") == "interaction":
            for f in INTERACTION_FIELDS:
                if f not in e:
                    problems.append(f"{where} interaction 缺字段 {f}")

    # 线索句引用校验：只查「见《XX》篇」格式的语料内引用（带「篇」字结尾），
    # 《中国药典》这类真实书籍名不是语料标题，不查
    titles = {e["title"] for e in entries}
    for e in entries:
        for ref in re.findall(r"《([^》]+篇)》", e["content"]):
            if ref not in titles:
                problems.append(f"[{e.get('doc_id')}] 线索句引用了不存在的标题: 《{ref}》")

    # A 类防漏：只检查语料中已生成的草药（herb 篇标题以清单名开头），
    # 还没扩写的草药跳过（试跑阶段必然不齐，不是错误）
    herb_titles = [e["title"] for e in entries if e.get("doc_type") == "herb"]
    for herb in herbs:
        if herb["category"] != "A":
            continue
        # 括号回退：「人参（亚洲种）」的语料标题可能写作「人参概述」
        base, alt = herb["name"], re.sub(r"[（(][^）)]*[）)]", "", herb["name"])
        if not any(t.startswith(base) or t.startswith(alt) for t in herb_titles):
            continue
        matched = rules_for_herb(herb["name"], rules)
        if not matched:
            problems.append(f"[清单A类] {herb['name']} 在规则库中未匹配到任何条目（清单或规则库别名有问题）")
            continue
        for rule in matched:
            ints = [e for e in entries
                    if e.get("doc_type") == "interaction" and e.get("rule_id") == rule["id"]]
            if not ints:
                problems.append(f"[清单A类] {herb['name']} 缺少 rule_id={rule['id']} 的相互作用篇")
                continue
            for e in ints:
                if e.get("severity") != rule["severity"] or e.get("evidence_level") != rule["evidence_level"]:
                    problems.append(
                        f"[{e.get('doc_id')}] 与规则库 {rule['id']} 等级不一致: "
                        f"语料 {e.get('severity')}/{e.get('evidence_level')} vs "
                        f"规则库 {rule['severity']}/{rule['evidence_level']}")
                # 铁律 1 要求 advice 原样转写 → 机器可校验：正文必须包含规则库 advice 全文
                if rule["advice"] not in e.get("content", ""):
                    problems.append(
                        f"[{e.get('doc_id')}] 处置建议与规则库 {rule['id']} 不一致: "
                        f"正文未包含规则库 advice 原文「{rule['advice']}」")
    return problems


def write_report(entries: list, herbs: list, rules: list, problems: list) -> None:
    """校验报告：分布统计 + 问题清单，写 UTF-8 文件（Windows 控制台 GBK 会乱码）。"""
    lines = [f"全量校验报告（{len(entries)} 篇文档）", "=" * 50, ""]
    types = {}
    for e in entries:
        types[e.get("doc_type", "?")] = types.get(e.get("doc_type", "?"), 0) + 1
    lines.append("文档类型分布: " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    a_ints = [e for e in entries if e.get("doc_type") == "interaction" and e.get("rule_id")]
    lines.append(f"带 rule_id 的相互作用篇（A 类）: {len(a_ints)}")
    lines.append("")
    if problems:
        lines.append(f"发现问题 {len(problems)} 个:")
        lines += [f"  - {p}" for p in problems]
    else:
        lines.append("全部通过 ✓")
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

async def run(args, client, model):
    herbs = load_herbs()
    if args.category:
        herbs = [h for h in herbs if h["category"] == args.category]
    if args.limit:
        herbs = herbs[: args.limit]

    rules = load_rules()
    seed_text = load_seed_text()
    CORPUS_DIR.mkdir(exist_ok=True)

    # 并发 4 味：串行 97 味要 1 小时+，4 并发约 15-20 分钟（DeepSeek 单 key 限速内）
    sem = asyncio.Semaphore(4)

    async def _one(herb):
        out_file = CORPUS_DIR / f"{herb['name']}.jsonl"
        if out_file.exists() and not args.force:
            print(f"skip {herb['name']} (output exists)", flush=True)
            return
        matched = rules_for_herb(herb["name"], rules)
        if herb["category"] == "A" and not matched:
            print(f"WARN {herb['name']}: category=A but no rules matched — skipping", flush=True)
            return
        async with sem:
            print(f"generating {herb['name']} ({herb['category']}, {len(matched)} rules)...", flush=True)
            entries = await generate_one(client, model, herb, matched, seed_text)
            out_file.write_text(
                "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
                encoding="utf-8",
            )
            print(f"  -> {herb['name']}: {len(entries)} entries", flush=True)

    await asyncio.gather(*[_one(h) for h in herbs])


def merge_and_validate():
    """合并 corpus/ 下所有草药文件为全量语料，并跑全量校验。"""
    all_entries = []
    for f in sorted(CORPUS_DIR.glob("*.jsonl")):
        all_entries += load_jsonl(f)

    # doc_id 重编号：模型每次调用都参照范文从 004 开始续编，跨文件必然撞号。
    # 合并时按 doc_type 全局重排（HERB-001…/COMP-001…/INT-001…），
    # 内容标识靠 title，doc_id 只要求全局唯一。
    counters: dict = {}
    for e in all_entries:
        t = e.get("doc_type", "?")
        counters[t] = counters.get(t, 0) + 1
        e["doc_id"] = f"{t.upper()}-{counters[t]:03d}"
    FINAL_OUT.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in all_entries) + "\n",
        encoding="utf-8",
    )
    problems = validate_corpus(all_entries, load_herbs(), load_rules())
    write_report(all_entries, load_herbs(), load_rules(), problems)
    print(f"merged {len(all_entries)} entries -> {FINAL_OUT.name}")
    print(f"validation: {len(problems)} problems (report -> {REPORT_FILE.name})")


def main():
    load_dotenv()  # 读项目根目录 .env（ANTHROPIC_API_KEY/BASE_URL/MODEL）
    parser = argparse.ArgumentParser(description="v2 阶段 0.3 多跳语料扩写")
    parser.add_argument("--category", choices=["A", "B", "C"], help="只处理某一类")
    parser.add_argument("--limit", type=int, help="只处理前 N 味（试跑用）")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")
    parser.add_argument("--merge", action="store_true", help="只做合并+校验（不调 API）")
    args = parser.parse_args()

    if args.merge:
        merge_and_validate()
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env")
        sys.exit(1)
    # 0.5 的双协议工厂：DeepSeek 走 anthropic 协议零变化；vLLM 时切 OPENMED_LLM_PROTOCOL=openai
    client = create_llm_client(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
    )
    model = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")
    asyncio.run(run(args, client, model))


if __name__ == "__main__":
    main()
