"""扩充数据答案方向 judge 抽检（阶段 2.1 第 5 步）

抽样 50 条扩充数据，LLM 当判官：答案的「作用方向 / 严重程度 / 证据等级」
三字段是否与语料 interaction 篇的权威字段一致（这三项是相互作用的核心事实，
方向错了就是医疗误导；语言质量、建议详略不在判分范围）。

为什么用 judge 而不是纯确定性校验：答案由 LLM 改写生成，方向正确与否
需要语义比对，确定性手段够不着。judge 保留 thinking（需要仔细比对语料）。

抽样：46 篇 interaction 每篇 1 条（覆盖全部素材）+ 4 条随机补足 50。
固定 random seed 保证可复现。

用法：
  .venv/Scripts/python.exe scripts/judge_expanded_answers.py
  .venv/Scripts/python.exe scripts/judge_expanded_answers.py --input <文件> --sample 50
"""
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "sft" / "multihop" / "expanded" / "expanded_raw.json"
CORPUS = ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"


def load_corpus() -> dict[str, dict]:
    """doc_id -> 语料行。"""
    out: dict[str, dict] = {}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["doc_id"]] = row
    return out


def sample_items(data: list[dict], n: int) -> list[dict]:
    """每篇 interaction 1 条 + 随机补足，固定 seed 可复现。"""
    rng = random.Random(20261014)
    by_inter: dict[str, list[dict]] = {}
    for item in data:
        inter = next((d for d in item["entity_chain"] if d.startswith("INTERACTION-")), "?")
        by_inter.setdefault(inter, []).append(item)
    picked: list[dict] = []
    for inter in sorted(by_inter):
        picked.append(rng.choice(by_inter[inter]))
    rest = [it for it in data if it not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, n - len(picked))])
    return picked


def build_judge_prompt(item: dict, corpus: dict) -> str:
    """判官 prompt：语料权威字段 + 答案，只判三字段一致性。"""
    inter_id = next(d for d in item["entity_chain"] if d.startswith("INTERACTION-"))
    inter = corpus[inter_id]
    auth = (
        f"- 作用方向（direction）：{inter.get('direction', '?')}\n"
        f"- 严重程度（severity）：{inter.get('severity', '?')}\n"
        f"- 证据等级（evidence_level）：{inter.get('evidence_level', '?')}"
    )
    return (
        "你是用药咨询答案的审核员。下面有一条知识库语料的权威字段，和一条"
        "基于该语料改写的助手答案。请核对答案里「可核验的用药信息」部分陈述的"
        "三个核心事实是否与权威字段一致：\n\n"
        "【语料权威字段】\n"
        f"{auth}\n\n"
        "【助手答案】\n"
        f"{item['final_answer']}\n\n"
        "判分标准：\n"
        "1. 方向：答案说「A 与 B 合用风险/效果如何变化」必须与 direction 一致；\n"
        "2. 严重程度：答案对风险大小的定性（如「可能增加出血风险」）不得与\n"
        "   severity 矛盾（minor 说成致命、major 说成无害都是错）；\n"
        "3. 证据等级：答案引用的证据类型不得明显夸大于 evidence_level；\n"
        "4. 语言详略、建议条数、标点格式不扣分。\n\n"
        "只输出 JSON：{\"verdict\": \"pass\" 或 \"fail\", \"reason\": \"一句话说明，"
        "fail 时指出哪个字段不一致\"}"
    )


async def judge_one(client, model: str, item: dict, corpus: dict, sem: asyncio.Semaphore) -> dict:
    """单条判分，返回 {seed_id, verdict, reason}（调用失败记 fail）。"""
    prompt = build_judge_prompt(item, corpus)
    for attempt in range(3):
        try:
            async with sem:
                resp = await client.messages.create(
                    # judge 保留 thinking：比对语料需要推理；量小（50 条）成本可接受
                    model=model, max_tokens=4096, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            from core.llm_utils import extract_text_content
            text = extract_text_content(resp.content).strip()
            # 剥掉可能的 markdown 围栏后解析 JSON
            if text.startswith("```"):
                text = text.split("```", 2)[1] if "```" in text[3:] else text
                text = text.strip()
                if text.startswith("json"):
                    text = text[3:].strip()
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                verdict = json.loads(text[start:end + 1])
                if verdict.get("verdict") in ("pass", "fail"):
                    return {"seed_id": item["seed_id"], "question": item["question"],
                            "verdict": verdict["verdict"], "reason": verdict.get("reason", "")}
        except Exception as ex:
            print(f"  {item['seed_id']} 第 {attempt + 1} 次判分调用失败: {ex}", flush=True)
    return {"seed_id": item["seed_id"], "question": item["question"],
            "verdict": "fail", "reason": "judge 调用失败"}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="扩充数据文件")
    ap.add_argument("--sample", type=int, default=50, help="抽检条数")
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    corpus = load_corpus()
    sample = sample_items(data, args.sample)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not found in .env")
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        timeout=180.0,
    )
    model = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")

    print(f"抽检 {len(sample)} 条（4 路并发 judge）", flush=True)
    sem = asyncio.Semaphore(4)
    results = await asyncio.gather(*(judge_one(client, model, it, corpus, sem) for it in sample))

    fails = [r for r in results if r["verdict"] == "fail"]
    lines = [f"抽检总数: {len(results)}", f"通过: {len(results) - len(fails)}",
             f"不通过: {len(fails)}", ""]
    if fails:
        lines.append("不通过清单:")
        for r in fails:
            lines.append(f"  {r['seed_id']} {r['question'][:40]}")
            lines.append(f"    理由: {r['reason'][:200]}")
    out = ROOT / "data" / "sft" / "multihop" / "expanded" / "judge_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out}")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    asyncio.run(main())
