"""黄金种子查询词机器先验（阶段 2.1 第 2 步的确定性检查）

种子草稿里每条 search_plan 的查询词跑真实检索链路，验证实体链的全部节点
chunk 是否出现在各步检索结果的并集里。命中 = 该 doc_id 的任一 chunk id
进入结果列表（与判分器口径一致：实体链命中看 chunk id，标题兜底）。

全命中 → 查询词设计合格，人眼只看医疗结论；不命中 → 打印哪一步漏了哪个
节点，回 seeds_draft.json 改查询词后重跑。纯确定性检查，零 LLM 成本。

检索链路：宿主机直连容器 ChromaDB（localhost:8001）+ 本地 reranker，
与生产 search_with_rerank 同款逻辑（hybrid 召回 top20 → RRF → 精排 top5）。
为什么不打容器 /search 端点：容器内每次请求 ~25s（ChromaDB 服务端在
Windows Docker 上 CPU 推理慢），38 次要 16 分钟；宿主机直连一次启动
~90s 后每次检索瞬时。两者查询词排序结果一致（同一库、同一模型）。

宿主机环境坑（必须 import 前设置）：
  - HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1：宿主机加载本地模型时会先打
    被墙的 huggingface.co 检查，不设离线会无限挂起
  - 不要设 HF_HOME=./data/huggingface：那是容器写的符号链接，Windows 读不了
"""

import json
import os
import sys
from pathlib import Path

# 宿主机从 scripts/ 目录跑时 sys.path 不含项目根；项目模块（mcp/core）需要它。
# 容器内（docker compose exec）工作目录就是项目根，此行是幂等空操作。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "sft" / "multihop" / "seeds" / "seeds_draft.json"
CHUNK_INDEX = ROOT / "data" / "eval" / "chunk_index.json"


def doc_to_chunk_ids(chunk_index: dict) -> dict[str, set[str]]:
    """doc_id → 它的全部 chunk id 集合（一个 doc 可能切多个 chunk）。"""
    mapping: dict[str, set[str]] = {}
    for doc_id, doc in chunk_index["docs"].items():
        mapping[doc_id] = {c["id"] for c in doc.get("chunks", [])}
    return mapping


def search_with_rerank(kb, reranker, query: str, top_k: int = 5) -> list[dict]:
    """复刻生产 search_with_rerank：hybrid 召回 → 精排 → top_k。

    与 mcp/tool_manager.py 的逻辑一致：recall_k = max(top_k*4, 20)，
    精排失败自动降级词重叠打分（reranker 内部处理），结果按最终排序返回。
    """
    recall_k = max(top_k * 4, 20)
    candidates = kb.hybrid_search(query, top_k=recall_k)
    if len(candidates) <= top_k:
        return candidates
    ranked, _status = reranker.rerank_with_status(query, candidates, top_k)
    return ranked


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=SEEDS, help="输入文件（默认黄金种子，扩充数据也用同一先验）")
    args = ap.parse_args()
    seeds = json.loads(args.input.read_text(encoding="utf-8"))
    chunk_index = json.loads(CHUNK_INDEX.read_text(encoding="utf-8"))
    doc_chunks = doc_to_chunk_ids(chunk_index)

    # 懒 import 放在离线环境变量之后；模型只加载一次，之后每次检索瞬时
    import time
    t0 = time.time()
    from mcp.knowledge_base import KnowledgeBase
    from core.reranker import CrossEncoderReranker
    kb = KnowledgeBase(chroma_host="localhost", chroma_port=8001)
    reranker = CrossEncoderReranker()
    print(f"链路初始化 {time.time()-t0:.0f}s（含模型加载，一次性开销）\n")

    all_pass = True
    for seed in seeds:
        # 每条 seed 期望覆盖的 chunk id 全集（实体链所有节点）
        required: set[str] = set()
        for doc_id in seed["entity_chain"]:
            required |= doc_chunks.get(doc_id, set())

        # 逐步打检索，记录每步命中了哪些 chunk
        per_step: list[tuple[str, list[str], set[str]]] = []
        for plan in seed["search_plan"]:
            hits = search_with_rerank(kb, reranker, plan["query"])
            hit_set = set(h["id"] for h in hits)
            per_step.append((plan["query"], [h["id"] for h in hits], hit_set))

        # 各步并集 vs 期望覆盖：逐节点报告在哪一步命中
        union = set().union(*(s for _, _, s in per_step))
        missing = required - union
        status = "OK" if not missing else "FAIL"

        print(f"[{status}] {seed['seed_id']}")
        if missing:
            all_pass = False
            for doc_id in seed["entity_chain"]:
                if doc_chunks.get(doc_id, set()) & missing:
                    title = chunk_index["docs"].get(doc_id, {}).get("title", doc_id)
                    print(f"   未命中: {doc_id} {title}")
            for query, hits, _ in per_step:
                print(f"   查询「{query}」top5: {[h[:8] for h in hits]}")
        else:
            # 通过也打印每步命中，方便人工复核查询词是否自然
            for query, _, hit_set in per_step:
                print(f"   查询「{query}」→ {len(hit_set & required)} 个目标节点")

    print()
    print("全部通过" if all_pass else "有未命中，需要改查询词")


if __name__ == "__main__":
    main()
