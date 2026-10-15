"""黄金种子真实检索回填（阶段 2.1 第 3 步）

把 seeds_draft.json 里每条 search_plan 的查询词打真实检索链路，剥掉运维键
后把 top5 结果写回每步的 tool_results 字段；第二步查询的 observation
（两步之间的自然衔接句）用 DeepSeek 基于第一步检索结果生成。

为什么要做这一步：训练数据里的轨迹必须含真实工具结果，模型才能学到
「查询 → 看到结果 → 决定下一步」的行为模式。检索结果直接取生产链路的
真实输出（同一库、同一精排模型），保证训练态与生产态一致。

剥键规则（v2-plan 0.5 遗留 TODO）：工具结果里 cached/reranked/
rerank_degraded/latency_ms 等运维字段、score/rrf_score/rerank_score 等
打分字段一律不落盘，只留 id/title/content——训练数据只教「读文档」，
不教「看分数」。

性能：宿主机直连（容器 /search 每次 ~25s 太慢）+ 查询词去重缓存 +
批量精排（所有 (query, content) 打分对打包成 batch 一次过模型，比逐查询
调用快数倍）。50 条 77 个唯一查询，预计 10-20 分钟。

用法：
  .venv/Scripts/python.exe scripts/backfill_seed_tool_results.py                 # 检索回填 + 观察句
  .venv/Scripts/python.exe scripts/backfill_seed_tool_results.py --no-observations    # 只回填检索
  .venv/Scripts/python.exe scripts/backfill_seed_tool_results.py --observations-only  # 只补观察句
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

# 宿主机从 scripts/ 目录跑时 sys.path 不含项目根；项目模块（mcp/core）需要它。
# 容器内（docker compose exec）工作目录就是项目根，此行是幂等空操作。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 宿主机环境坑（必须 import 前设置）：加载本地模型时会先打被墙的
# huggingface.co 检查，不设离线会无限挂起（见 verify_seed_queries.py 同款注释）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "sft" / "multihop" / "seeds" / "seeds_draft.json"

# 检索结果条目只保留这三个键（白名单比黑名单稳：未来加新运维键也不会漏剥）
_KEEP_KEYS = ("id", "title", "content")


def clean_entry(entry: dict) -> dict:
    """剥掉打分键和运维键，只留 id/title/content。"""
    return {k: entry[k] for k in _KEEP_KEYS if k in entry}


def collect_unique_queries(seeds: list) -> list[str]:
    """按首次出现顺序收集全部唯一查询词（去重后重复检索只跑一次）。"""
    seen: set[str] = set()
    queries: list[str] = []
    for seed in seeds:
        for plan in seed["search_plan"]:
            q = plan["query"]
            if q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def build_observations_prompt(seed: dict, plan1: dict | None, plan2: dict) -> str:
    """观察句生成的 prompt：plan1=None 是第一步（无检索结果可依据，只声明行动）。"""
    if plan1 is None:
        # 第一步观察句：还没检索，只能依据用户问题和查询词写「先查什么」
        return (
            "你在写一段中文对话中的衔接句。场景：用药咨询助手的多步检索轨迹。\n\n"
            f"用户问题：{seed['question']}\n"
            f"第一步检索的查询词：{plan2['query']}\n\n"
            "请写一句话（15-40 字），作为助手开始第一步检索之前的行动声明：\n"
            "1. 用自然的中文口语说「先查什么」；\n"
            "2. 只能依据用户问题和查询词，不得引用任何检索结果（这一步还没有检索）；\n"
            "3. 不要出现文档编号、规则编号；不要用 markdown；只输出这一句话本身。"
        )
    titles = "、".join(r["title"] for r in (plan1.get("tool_results") or [])[:5])
    return (
        "你在写一段中文对话中的衔接句。场景：用药咨询助手的多步检索轨迹。\n\n"
        f"用户问题：{seed['question']}\n"
        f"第一步检索的查询词：{plan1['query']}\n"
        f"第一步检索返回的文档标题：{titles}\n"
        f"第二步检索的查询词：{plan2['query']}\n\n"
        "请写一句话（30-60 字），作为助手在第一步检索之后的观察陈述：\n"
        "1. 用自然的中文口语复述第一步查到了什么（只能依据上面给的文档标题，不得编造内容）；\n"
        "2. 顺带引出为什么要做第二步查询，自然过渡；\n"
        "3. 不要出现文档编号、规则编号；不要用 markdown；只输出这一句话本身。"
    )


async def fill_observations(seeds: list, client, model: str) -> int:
    """给 observation 缺失的步生成衔接句（4 路并发）。返回失败条数。

    500 条 × 2 步 = 1000 个观察句，串行要 ~1 小时；并发压到 ~20 分钟。
    每个观察句独立（只依赖自己那步的检索标题），互不冲突，锁只保护
    计数器与进度打印。
    """
    import asyncio as aio
    sem = aio.Semaphore(4)
    lock = aio.Lock()
    failed = 0
    done = 0
    total = sum(
        1 for seed in seeds for plan in seed["search_plan"] if not plan.get("observation")
    )

    async def fill_one(seed: dict, plan: dict, plan1: dict | None) -> None:
        nonlocal failed, done
        prompt = build_observations_prompt(seed, plan1, plan)
        for attempt in range(3):
            try:
                async with sem:
                    resp = await client.messages.create(
                        # 关 thinking（2026-10-14 实测）：观察句是「复述检索结果
                        # 引出的下一步」的短模式化任务，思考链不提供新信息，
                        # 只白吃 token（老坑：thinking 吃光 max_tokens 致 text 残片）。
                        # 关掉后单条 output 从 ~2k token 降到 ~500，成本省 3-4 倍。
                        # SDK 0.40 不收 thinking 参数，走 extra_body 透传。
                        model=model, max_tokens=4096, temperature=0.7,
                        extra_body={"thinking": {"type": "disabled"}},
                        messages=[{"role": "user", "content": prompt}],
                    )
                from core.llm_utils import extract_text_content
                text = extract_text_content(resp.content).strip()
                if text:
                    async with lock:
                        plan["observation"] = text
                    return
            except Exception as ex:
                print(f"  {seed['seed_id']} 第 {plan['step']} 步观察句第 {attempt + 1} 次尝试失败: {ex}",
                      flush=True)
        async with lock:
            failed += 1
        print(f"  [警告] {seed['seed_id']} 第 {plan['step']} 步观察句 3 次重试全部失败，留空",
              flush=True)

    async def tracked(seed: dict, plan: dict, plan1: dict | None) -> None:
        nonlocal done
        await fill_one(seed, plan, plan1)
        async with lock:
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  观察句进度 {done}/{total}", flush=True)

    tasks = []
    for seed in seeds:
        plans = seed["search_plan"]
        for i, plan in enumerate(plans):
            if plan.get("observation"):
                continue  # 已有观察句的步跳过（幂等补缺）
            plan1 = plans[i - 1] if i > 0 else None  # 第一步无前步（不取 plans[-1]）
            tasks.append(tracked(seed, plan, plan1))
    await aio.gather(*tasks)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-observations", action="store_true", help="只回填检索结果，不生成观察句")
    ap.add_argument("--observations-only", action="store_true", help="只补观察句（需已回填 tool_results）")
    ap.add_argument("--input", type=Path, default=SEEDS, help="输入文件（默认黄金种子，扩充数据也用它回填）")
    args = ap.parse_args()

    in_file: Path = args.input
    backup = in_file.with_name(in_file.stem + "_prefill_backup.json")
    seeds = json.loads(in_file.read_text(encoding="utf-8"))

    if not args.observations_only:
        # ── 阶段 1：检索回填（纯确定性，零 LLM 成本）──
        queries = collect_unique_queries(seeds)
        print(f"共 {len(seeds)} 条种子、{sum(len(s['search_plan']) for s in seeds)} 步查询、"
              f"{len(queries)} 个唯一查询词", flush=True)

        import time
        t0 = time.time()
        from mcp.knowledge_base import KnowledgeBase
        from core.reranker import CrossEncoderReranker
        kb = KnowledgeBase(chroma_host="localhost", chroma_port=8001)
        reranker = CrossEncoderReranker()
        # 回填必须用真精排结果，不允许静默降级为词重叠打分（那会污染训练数据）
        if not reranker._ensure_model():
            raise SystemExit("ERROR: reranker 模型加载失败，回填终止")
        model = reranker._model
        print(f"链路初始化 {time.time()-t0:.0f}s\n", flush=True)

        # 召回缓存：每个唯一查询词打一次 hybrid 召回 top20
        print("阶段 1/2：hybrid 召回（服务端 embedding）...", flush=True)
        recall: dict[str, list[dict]] = {}
        for q in queries:
            recall[q] = kb.hybrid_search(q, top_k=20)
        print(f"  召回完成：{len(queries)} 个查询 × 20 候选\n", flush=True)

        # 批量精排：把所有 (query, content) 打分对打包成 batch 一次过模型。
        # 与生产 rerank_with_status 同一模型、同一打分函数、同一排序逻辑，
        # 只是把多次 forward 合并成 batch forward，每对分数完全一致。
        print("阶段 2/2：批量精排（Cross-Encoder，CPU 推理较慢）...", flush=True)
        pairs: list[tuple[str, str]] = []
        backref: list[tuple[int, int]] = []
        for qi, q in enumerate(queries):
            for ci, cand in enumerate(recall[q]):
                pairs.append((q, cand.get("content", "")))
                backref.append((qi, ci))
        t0 = time.time()
        scores = model.predict(pairs, batch_size=64, show_progress_bar=True)
        scores = [float(s) for s in scores]
        print(f"  精排完成：{len(pairs)} 对，用时 {time.time()-t0:.0f}s\n", flush=True)

        # 分数归位到各查询，排序取 top5，剥键
        ranked: dict[str, list[tuple[float, dict]]] = {}
        for (qi, ci), s in zip(backref, scores):
            ranked.setdefault(queries[qi], []).append((s, recall[queries[qi]][ci]))
        top5 = {
            q: [clean_entry(c) for _, c in sorted(v, key=lambda p: p[0], reverse=True)[:5]]
            for q, v in ranked.items()
        }

        for seed in seeds:
            for plan in seed["search_plan"]:
                plan["tool_results"] = top5[plan["query"]]
        print("检索结果已写回内存（tool_results 字段）\n", flush=True)

    if not args.no_observations:
        # ── 阶段 2：观察句生成（DeepSeek，~50 次小调用，成本 <¥0.5）──
        from dotenv import load_dotenv
        load_dotenv()  # 读项目根 .env（ANTHROPIC_API_KEY/BASE_URL/MODEL）
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ERROR: ANTHROPIC_API_KEY not found in .env")
        # 不走 core.llm_utils.create_llm_client 工厂：它不透传 timeout 参数，
        # 而 anthropic SDK 默认超时 10 分钟——串行生成 45 条时一条请求挂起会卡死
        # 整个循环（实测卡在 MH-029 十分钟）。数据脚本里直接构造并显式给 60s。
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(
            api_key=api_key,
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            timeout=60.0,
        )
        model_name = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")
        print("阶段 3/3：DeepSeek 生成观察句...", flush=True)
        failed = asyncio.run(fill_observations(seeds, client, model_name))
        if failed:
            print(f"[警告] {failed} 个观察句生成失败留空，可稍后用 --observations-only 重跑", flush=True)

    # 落盘：先备份原文件（回填只加字段不改已有内容，但留一份原始草稿更稳）
    if not backup.exists():
        shutil.copy2(in_file, backup)
    in_file.write_text(json.dumps(seeds, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已落盘：{in_file}（原始草稿备份在 {backup.name}）", flush=True)


if __name__ == "__main__":
    main()
