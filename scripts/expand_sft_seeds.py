"""黄金种子扩充 500 条（阶段 2.1 第 4 步）

从 50 条黄金种子扩充到 ~500 条 SFT 数据：对每篇可用 interaction 语料，
用 DeepSeek 生成多个不同题面变体（不同人群/场景/问法），few-shot 用
种子里的同篇条目，答案由 LLM 基于语料原文撰写。

关键约束（阶段 1 评测卫生的延续）：
  - 扩充只使用考卷 held_out 未涉及的 interaction（从 exams/*.json 动态
    计算，考卷变了排除集自动跟着变），保证 held_out 考试题的泛化度量
    不被训练数据污染；
  - entity_chain 由脚本按素材篇固定填充，LLM 只写 question/查询词/答案，
    避免模型写错 doc_id；
  - 生成的答案不喂回模型（few-shot 答案只作格式示范），并要求重新组织
    语言，防止训练数据出现文本级复制。

输出：data/sft/multihop/expanded/expanded_raw.json（回填前）。
回填（tool_results + observation）复用 backfill_seed_tool_results.py
的 --input 参数（见该脚本）。

用法：
  .venv/Scripts/python.exe scripts/expand_sft_seeds.py --target 500 --per-call 5
  .venv/Scripts/python.exe scripts/expand_sft_seeds.py --resume      # 断点续跑
  .venv/Scripts/python.exe scripts/expand_sft_seeds.py --judge-sample 50  # 答案方向抽检
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "sft" / "multihop" / "seeds" / "seeds_draft.json"
CORPUS = ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"
EXAM_DIR = ROOT / "data" / "eval" / "exams"
OUT_DIR = ROOT / "data" / "sft" / "multihop" / "expanded"
OUT_FILE = OUT_DIR / "expanded_raw.json"

# 一篇素材每次调用生成的变体数（越大越省钱但单次输出越长、越容易截断）
PER_CALL_DEFAULT = 5


def load_corpus() -> dict[str, dict]:
    """doc_id -> 语料行（JSONL 每行一篇）。"""
    out: dict[str, dict] = {}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["doc_id"]] = row
    return out


def held_out_interactions() -> set[str]:
    """从考卷动态计算 held_out 涉及的 interaction 集合（扩充时全部避开）。"""
    held: set[str] = set()
    for p in EXAM_DIR.glob("*.json"):
        exam = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(exam, list):
            continue
        for q in exam:
            if q.get("split") != "held_out":
                continue
            for node in q.get("entity_chain", []):
                if isinstance(node, dict) and node.get("doc_id", "").startswith("INTERACTION-"):
                    held.add(node["doc_id"])
    return held


def exam_questions_by_interaction() -> dict[str, list[str]]:
    """考卷全部题的题面索引：interaction -> 该 interaction 的题面列表。

    用途：写进生成 prompt 当「不得相似」清单。同一篇 interaction 的常见
    问法有限，LLM 自由发挥容易和考卷 train 题面撞车；题面级复制会让
    训练数据直接泄漏考试内容（held_out 的篇已在素材池排除，这里防的是
    train 题面撞车）。
    """
    index: dict[str, list[str]] = {}
    for p in EXAM_DIR.glob("*.json"):
        exam = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(exam, list):
            continue
        for q in exam:
            question = q.get("question", "")
            if not isinstance(question, str) or not question:
                continue
            for node in q.get("entity_chain", []):
                if isinstance(node, dict) and node.get("doc_id", "").startswith("INTERACTION-"):
                    index.setdefault(node["doc_id"], []).append(question)
    return index


def seed_examples(seeds: list[dict], interaction: str, corpus: dict) -> list[dict]:
    """种子里用同一篇 interaction 的条目，作为该篇的 few-shot 示例。"""
    return [
        s for s in seeds
        if interaction in s["entity_chain"]
    ]


def format_example(seed: dict) -> str:
    """把一条种子渲染成 LLM 可读的文本块。"""
    lines = [
        f"题面：{seed['question']}",
        "两步查询：",
    ]
    for plan in seed["search_plan"]:
        lines.append(f"- 第 {plan['step']} 步：{plan['query']}")
    lines.append("最终答案：")
    lines.append(seed["final_answer"])
    return "\n".join(lines)


def build_prompt(interaction_id: str, corpus: dict, examples: list[dict], count: int,
                 exam_questions: dict[str, list[str]]) -> str:
    """一篇素材的生成 prompt：语料原文 + 同篇种子示例 + 考卷题面避让清单 + 变体要求。"""
    it = corpus[interaction_id]
    # 实体链 = herb/component 篇从语料里按实体关联找？不猜——用示例的实体链。
    # 同篇种子的 entity_chain 一定一致（同一 interaction 的配套篇固定），取第一条的。
    chain = " → ".join(examples[0]["entity_chain"]) if examples else interaction_id

    parts = [f"你在为中文用药咨询系统构造训练数据。目标语料（知识库文档）如下：\n"]

    # 语料正文：interaction 篇 + 实体链上的其它篇
    parts.append(f"【语料：{it['title']}】")
    parts.append(it["content"])
    for doc_id in (examples[0]["entity_chain"] if examples else []):
        if doc_id == interaction_id or doc_id not in corpus:
            continue
        row = corpus[doc_id]
        parts.append(f"【语料：{row['title']}】")
        parts.append(row["content"])

    if examples:
        parts.append("\n【同类示例（格式示范，共 %d 条）】" % len(examples))
        for i, ex in enumerate(examples, 1):
            parts.append(f"--- 示例 {i} ---")
            parts.append(format_example(ex))

    # 考卷里同篇 interaction 的题面：生成的题面不得与它们相似（防训练数据
    # 与考试题撞车；held_out 的篇已在素材池排除，这里通常是 train 题面）
    exam_qs = exam_questions.get(interaction_id, [])
    if exam_qs:
        parts.append("\n【考试题面（生成的题面不得与以下任何一条相似）】")
        for i, q in enumerate(exam_qs, 1):
            parts.append(f"{i}. {q}")

    parts.append(f"""
【任务】为上面这篇相互作用语料写 {count} 个全新的用户提问变体，每个变体配两步搜索查询词和一份答案。

硬性要求：
1. 题面必须是口语化的中文用户提问，第一人称，像真实患者会问的话；
   每个变体的触发场景要明显不同（比如：不同基础病、不同年龄人群、不同
   保健品剂型（胶囊/茶/粉/炖汤）、不同问法（"能一起吃吗"/"要紧吗"/
   "听说…可以吗"）），不得是示例题面或考试题面的同义改写；
2. 两步查询：第 1 步是药材/成分的裸名称（如"三七"），第 2 步是
   "药材 西药"或"药材 药物类别"的组合查询；查询词必须是真实用户
   会输入的简短词，不带标点和语气词；
3. 最终答案的结构：先"## 现象复述"（一句话复述用户情况），再
   "## 可核验的用药信息"（作用方向、机制、严重程度、证据等级，
   全部依据上面给的语料，不得编造语料里没有的研究结论或数字），
   最后"## 建议"（处置建议与语料一致）；
4. 答案语言必须重新组织，不得整句复制示例答案或语料原文；
5. 不要出现任何文档编号、规则编号、内部代号。

只输出 JSON 数组，每个元素形如：
{{"question": "...", "search_plan": [{{"step": 1, "query": "..."}}, {{"step": 2, "query": "..."}}], "final_answer": "..."}}
""")
    return "\n".join(parts)


def parse_llm_json(text: str) -> list[dict] | None:
    """从 LLM 输出里提取 JSON 数组（剥掉可能的 markdown 围栏）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        text = text.strip()
        if text.startswith("json"):
            text = text[3:].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def salvage_parse(text: str) -> tuple[list[dict], bool]:
    """抢救解析被截断的 JSON 数组，返回 (完整对象列表, 是否截断)。

    DeepSeek 的 thinking 块会吃掉 max_tokens，text 块常被拦腰截断：
    数组里前几个对象已写完（合法 JSON），最后一条写到一半。逐对象
    raw_decode 能救回所有完整对象，只剩残缺的半条丢弃——比整批计废
    少浪费一次调用（被截断的输出也是花钱生成的）。
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        text = text.strip()
        if text.startswith("json"):
            text = text[3:].strip()
    start = text.find("[")
    if start == -1:
        return [], True
    text = text[start:]
    dec = json.JSONDecoder()
    items: list[dict] = []
    idx = start = 1  # 跳过 "["
    while True:
        # 跳过对象之间的空白和逗号
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text):
            return items, True  # 结尾没有 "]"：数组未闭合，截断
        if text[idx] == "]":
            return items, False  # 完整闭合
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            return items, True  # 下一个对象残了：截断点
        items.append(obj)
        idx = end


def validate_item(item: dict, interaction_id: str, corpus: dict, examples: list[dict], used_questions: set[str]) -> list[str]:
    """单条生成结果的确定性校验，返回问题列表（空 = 通过）。"""
    problems: list[str] = []
    if not isinstance(item.get("question"), str) or len(item["question"]) < 10:
        problems.append("题面缺失或过短")
    if item.get("question") in used_questions:
        problems.append("题面与已有条目重复")
    plan = item.get("search_plan")
    if not isinstance(plan, list) or len(plan) != 2:
        problems.append("search_plan 不是 2 步")
        return problems
    for p in plan:
        q = p.get("query")
        if not isinstance(q, str) or not q.strip() or len(q) > 30:
            problems.append(f"查询词不合法: {q!r}")
    ans = item.get("final_answer", "")
    if "## 现象复述" not in ans or "## 建议" not in ans:
        problems.append("答案缺标准结构（## 现象复述 / ## 建议）")
    # 答案里不允许出现内部编号
    for bad in ("INTERACTION-", "HERB-", "COMPONENT-", "doc_id", "rule_id"):
        if bad in ans or bad in item.get("question", ""):
            problems.append(f"出现内部编号 {bad}")
            break
    return problems


async def generate_batch(client, model: str, interaction_id: str, corpus: dict,
                         examples: list[dict], count: int, used_questions: set[str],
                         exam_questions: dict[str, list[str]]) -> tuple[list[dict], int]:
    """一次 LLM 调用生成一篇素材的 count 个变体；返回 (合格条目, 不合格数)。"""
    prompt = build_prompt(interaction_id, corpus, examples, count, exam_questions)
    ok_items: list[dict] = []
    bad = 0
    for attempt in range(3):
        try:
            resp = await client.messages.create(
                model=model, max_tokens=8192, temperature=0.8,
                # 关 thinking（2026-10-14 实测）：扩充是照语料改写的模式化任务，
                # 思考链不提供新信息。关掉后 5 条完整输出仅 ~2.2k token（开思考
                # ~8.2k 且被截断），成本省 3.7 倍、质量不降反升、不再截断。
                # SDK 0.40 不收 thinking 参数，走 extra_body 透传。
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            from core.llm_utils import extract_text_content
            text = extract_text_content(resp.content)
            data = parse_llm_json(text)
            if data is None:
                # 整段解析失败（大概率被截断）：抢救已写完的完整对象
                data, truncated = salvage_parse(text)
                if truncated:
                    bad += max(0, count - len(data))  # 残缺的半条计入废数
            if not data:
                bad += count
                continue
            for item in data:
                problems = validate_item(item, interaction_id, corpus, examples, used_questions)
                if problems:
                    bad += 1
                    continue
                # 实体链由脚本填充（与同篇种子一致），LLM 不写
                item["entity_chain"] = list(examples[0]["entity_chain"]) if examples else [interaction_id]
                ok_items.append(item)
                used_questions.add(item["question"])
            if ok_items:
                break
        except Exception as ex:
            print(f"    第 {attempt + 1} 次调用失败: {ex}", flush=True)
    return ok_items, bad


async def run_generation(args) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not found in .env")
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        timeout=180.0,  # 生成答案比观察句长，超时放宽到 180s
    )
    model = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")

    corpus = load_corpus()
    held = held_out_interactions()
    exam_qs = exam_questions_by_interaction()
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))

    # 素材池：全部 interaction - held_out 涉及 - 种子没示例可用的（种子里
    # 有同篇示例才给 LLM 做 few-shot；没有示例的篇用不了这个模式）
    interaction_ids = sorted(
        i for i, r in corpus.items()
        if r["doc_type"] == "interaction" and i not in held
    )
    usable = [i for i in interaction_ids if seed_examples(seeds, i, corpus)]
    skipped = [i for i in interaction_ids if i not in usable]
    print(f"interaction 总数 {len(interaction_ids)}，有种子示例的 {len(usable)} 篇，"
          f"无示例跳过 {len(skipped)} 篇: {skipped}", flush=True)

    # 断点续跑：读已有输出，统计每篇 interaction 已生成的条数
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_items: list[dict] = []
    per_inter_counts: dict[str, int] = {}
    if args.resume and OUT_FILE.exists():
        done_items = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        for it in done_items:
            for d in it["entity_chain"]:
                if d.startswith("INTERACTION-"):
                    per_inter_counts[d] = per_inter_counts.get(d, 0) + 1
        print(f"断点续跑：已有 {len(done_items)} 条，分布在 {len(per_inter_counts)} 篇", flush=True)

    used_questions = {it["question"] for it in done_items}

    # 每篇配额向上取整：46 篇 × 11 = 506 ≥ 500。resume 按配额补足缺口，
    # 不是「有 1 条就整篇跳过」——余额耗尽中断时多数篇是 10/11 条，
    # 整篇跳过会一条都补不上
    import math
    per_interaction = max(1, math.ceil(args.target / max(len(usable), 1)))
    pending = [i for i in usable if per_inter_counts.get(i, 0) < per_interaction]
    if args.limit_interactions > 0:
        pending = pending[:args.limit_interactions]
    print(f"每篇配额 {per_interaction} 条，未达标 {len(pending)} 篇待补，每批 {args.per_call} 条，"
          f"并发 {args.concurrency} 路", flush=True)

    # 并发生成：asyncio 信号量限流，used_questions/done_items 用锁保护。
    # DeepSeek 单次调用 30-90s，串行 138 次要 2-3 小时；4 路并发压到 ~40 分钟。
    import asyncio as aio
    sem = aio.Semaphore(args.concurrency)
    lock = aio.Lock()
    generated = 0
    bad_total = 0

    async def generate_one(inter: str) -> tuple[int, int]:
        nonlocal generated, bad_total
        examples = seed_examples(seeds, inter, corpus)
        got = 0
        bad = 0
        remaining = per_interaction - per_inter_counts.get(inter, 0)
        while remaining > 0:
            async with sem:
                count = min(args.per_call, remaining)
                items, b = await generate_batch(client, model, inter, corpus, examples, count, used_questions, exam_qs)
            async with lock:
                for item in items:
                    # ID 只用 len(done_items) 一个基准：resume 时已有条目也会
                    # 计数在内，新条目 ID 从已有条数 +1 连续编号
                    item["seed_id"] = f"EXP-{len(done_items) + 1:04d}"
                    item["source"] = "expanded"
                    done_items.append(item)
                    generated += 1
                bad_total += b
                # 每篇每批后落盘一次：断点续跑粒度 = 批
                OUT_FILE.write_text(json.dumps(done_items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            got += len(items)
            bad += b
            remaining -= count
            print(f"  {inter} 本批 +{len(items)}（废 {b}），累计 {generated}", flush=True)
        return got, bad

    # 46 篇并发跑（信号量限流），生成量可能略超目标（每篇配额向上取整），
    # 多出几条无害：训练集划分时按目标数截取
    await aio.gather(*(generate_one(inter) for inter in pending))

    print(f"完成：新增 {generated} 条，废弃 {bad_total} 条，落盘 {OUT_FILE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="扩充目标条数")
    ap.add_argument("--per-call", type=int, default=PER_CALL_DEFAULT, help="一次 LLM 调用生成的变体数")
    ap.add_argument("--concurrency", type=int, default=4, help="并发生成路数")
    ap.add_argument("--limit-interactions", type=int, default=0, help="只处理前 N 篇素材（0=全部，试跑用）")
    ap.add_argument("--resume", action="store_true", help="断点续跑（跳过已生成的篇）")
    args = ap.parse_args()
    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
