"""检索调参网格搜索（P2，2026-10-15）：qa single train 上离线搜生产检索链路最优参数。

生产检索链路（mcp/knowledge_base.py + core/reranker.py + mcp/tool_manager.py）：
  查询 → 别名展开 → 向量路召回 recall_pool + BM25 路召回 recall_pool
       → RRF(k, 等权) 融合 → 截断 recall_k 候选 → CrossEncoder 精排 → top5

调的 5 个旋钮（括号里是生产默认值）：
  recall_pool  每路召回的候选池大小     默认 80   [40, 80, 160, 240]
  rrf_k        RRF 平滑常数            默认 60   [10, 60, 100]
  vector_w     向量路权重（加权 RRF）   默认 1.0  [0.6, 1.0, 1.4]（BM25 路 = 2-w；1.0=等权）
  recall_k     精排前候选数            默认 20   [10, 20]
  rerank       精排开关                默认 on   [on, off]

主指标 MRR（与基线同口径：检索 top5 的 chunk id vs gold_chunks ∪ alt_gold_chunks，
统计函数直接复用 baseline_runner.qa_single_stats）。2026-10-15 rev3 由 F1@5 改为
MRR：每题只有 1 个出题依据片段、检索固定返回 5 条，F1@5 = hit@5 ÷ 3，上限锁死
在 0.3333——P2 第一轮网格「最优 0.3333」其实就是到顶，108 组里 40 组并列，
排序好坏完全看不出来。
held_out 题不参与搜索（防调参过拟合），只把最优组放上去看数字。

零 LLM 成本，纯本机：ChromaDB（容器 8001 端口）+ 宿主机 reranker 模型。

缓存策略（关键优化，为什么等价）：
  1. 向量/BM25 两路只按最大池 240 各召回一次，recall_pool 变体从缓存截断——
     召回结果只依赖 query 与 top_n，先取 240 再截前 N 与直接取前 N 完全一致；
  2. 精排只对「36 种 RRF 变体前 20 的并集」算一次分，小 recall_k 从分里截断——
     CrossEncoder 对每个 (query, chunk) 独立打分，候选集大小不影响单对分数；
  3. BM25 索引全库只建一次（懒构建缓存）。
  108 种参数组合因此退化成矩阵截断 + MRR 统计，十几分钟跑完。

注意：精排用「原始查询」（与生产一致——别名展开发生在 handler 内部，
tool_manager 拿到的仍是原查询），召回用「展开后查询」。

用法：
  .venv/Scripts/python.exe scripts/tune_retrieval.py
"""
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.reranker import CrossEncoderReranker, RERANK_CROSS_ENCODER
from evaluation.baseline_runner import qa_single_stats
from mcp.knowledge_base import KnowledgeBase, _expand_aliases

ROOT = Path(__file__).resolve().parents[1]
QA_EXAM = ROOT / "data" / "eval" / "exams" / "qa_v1.json"
OUT_DIR = ROOT / "data" / "eval" / "results" / f"tune_retrieval_{time.strftime('%Y%m%d')}"

MAX_RECALL_POOL = 240   # 召回缓存的最大池（向量/BM25 各召回这么多条再截断）
MAX_RECALL_K = 20       # 精排缓存的最大候选数（RRF 前 20 的并集算一次分）

# 网格：与生产默认值对比的参数空间
# vector_w 是向量路权重，BM25 路 = 2 - w。为什么是 2 - w 而不是 1 - w：
# 生产的等权 RRF 是「两路各贡献 1/(k+rank)」（权重和 = 2），
# 所以等权对应的 w = 1.0；w > 1 偏向量路，w < 1 偏 BM25。
# recall_k 只探到 20：生产线上精排 30 对延迟约 9s 不会采用，30 只是边界点，
# 20 vs 30 的趋势从 10/20 档看得出，砍掉省精排量。
GRID = {
    "recall_pool": [40, 80, 160, 240],
    "rrf_k": [10, 60, 100],
    "vector_w": [0.6, 1.0, 1.4],
    "recall_k": [10, 20],
    "rerank": ["on", "off"],
}
PRODUCTION_DEFAULTS = {"recall_pool": 80, "rrf_k": 60, "vector_w": 1.0, "recall_k": 20, "rerank": "on"}


def weighted_rrf(vector_ranked: list, bm25_ranked: list, k: int = 60, w: float = 1.0) -> list:
    """加权 RRF 融合：score(d) = w/(k+rank_v) + (2-w)/(k+rank_b)，按分数降序返回。

    w=1.0 时与生产 _reciprocal_rank_fusion 的等权版完全等价（两条路各贡献
    1/(k+rank)，权重和为 2）——这是「调参结论能直接套回生产」的前提。
    只依赖排名不依赖分数量纲，所以 BM25 分数与向量相似度可以安全融合。
    """
    scores: dict = {}
    lookup: dict = {}
    for rank, item in enumerate(vector_ranked):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
        lookup[doc_id] = item
    for rank, item in enumerate(bm25_ranked):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + (2.0 - w) / (k + rank + 1)
        lookup[doc_id] = item
    fused = []
    for doc_id in sorted(scores, key=scores.get, reverse=True):
        entry = dict(lookup[doc_id])
        entry["rrf_score"] = round(scores[doc_id], 5)
        fused.append(entry)
    return fused


def retrieve_top5(vector: list, bm25: list, cfg: dict, rerank_scores: dict) -> list:
    """按一组参数从每题缓存算检索 top5 的 chunk id 列表（纯函数，可单测）。"""
    v = vector[:cfg["recall_pool"]]
    b = bm25[:cfg["recall_pool"]]
    fused = weighted_rrf(v, b, k=cfg["rrf_k"], w=cfg["vector_w"])
    if cfg["rerank"] == "off":
        return [item["id"] for item in fused[:5]]
    top_rr = fused[:cfg["recall_k"]]
    ranked = sorted(top_rr, key=lambda item: rerank_scores.get(item["id"], -1e9), reverse=True)
    return [item["id"] for item in ranked[:5]]


def gen_configs() -> list:
    """生成全部参数组合。rerank=off 时 recall_k 不参与链路，该维度合并成一组。"""
    configs = []
    for pool, k, w, rerank in itertools.product(
        GRID["recall_pool"], GRID["rrf_k"], GRID["vector_w"], GRID["rerank"],
    ):
        if rerank == "on":
            for recall_k in GRID["recall_k"]:
                configs.append({"recall_pool": pool, "rrf_k": k, "vector_w": w,
                                "recall_k": recall_k, "rerank": rerank})
        else:
            configs.append({"recall_pool": pool, "rrf_k": k, "vector_w": w,
                            "recall_k": None, "rerank": rerank})
    return configs


def score_case(retrieved: list, gold: list) -> dict:
    """一题的检索统计（直接复用 baseline_runner.qa_single_stats，口径不会分叉）。"""
    return qa_single_stats(retrieved, gold)


def aggregate(rows: list) -> dict:
    """多题统计求均值（空输入返回 0，防除零）。"""
    keys = ("mrr", "hit_at_1", "hit_at_5")
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in keys}


def _rerank_scores(reranker: CrossEncoderReranker, question: str, items: list) -> tuple:
    """对候选并集精排打分，返回 ({chunk_id: 分数}, 状态)。

    模型可用时用大 batch 直算：CrossEncoder 对每个 (query, chunk) 独立打分，
    batch 大小只影响速度不影响分数，与生产默认 batch 的结果逐位一致。
    模型不可用时走 rerank_with_status 的词重叠降级（与生产同路径）。
    """
    if reranker._ensure_model():
        try:
            pairs = [(question, item.get("content", "")) for item in items]
            scores = reranker._model.predict(pairs, batch_size=128)
            return ({item["id"]: float(s) for item, s in zip(items, scores)}, RERANK_CROSS_ENCODER)
        except Exception:
            pass
    ranked, status = reranker.rerank_with_status(question, items, len(items))
    return ({item["id"]: float(item.get("rerank_score", 0.0)) for item in ranked}, status)


def build_case_cache(kb: KnowledgeBase, reranker: CrossEncoderReranker,
                     question: str, gold: list, alias: bool, do_rerank: bool = True) -> dict:
    """一题的召回/精排缓存。alias=True 时召回走别名展开后的查询（精排仍用原查询）。

    do_rerank=False 时只召回不精排（held_out 用：等网格搜出最优组再补算，
    避免对用不上的配置浪费精排量）。"""
    q = _expand_aliases(question) if alias else question
    vector = kb._vector_search_raw(q, MAX_RECALL_POOL)
    bm25 = kb._bm25_search_raw(q, MAX_RECALL_POOL)

    rerank_scores, status = {}, "not_needed"
    if do_rerank:
        # 需要精排分的候选并集：
        # - alias on 档：36 种 RRF 变体（pool×k×w）前 MAX_RECALL_K 的并集，每种组合的
        #   精排输入都是它的子集，从同一份分里截断等价于独立精排（精排分只依赖
        #   query-chunk 对，与候选集无关）
        # - alias off 档：只服务「生产默认参数下别名收益」的单因子对比，
        #   只需生产默认变体前 recall_k=20 的候选，精排量省一大半
        if alias:
            union_ids = set()
            for pool, k, w in itertools.product(GRID["recall_pool"], GRID["rrf_k"], GRID["vector_w"]):
                fused = weighted_rrf(vector, bm25, k=k, w=w)
                union_ids.update(item["id"] for item in fused[:MAX_RECALL_K])
        else:
            fused = weighted_rrf(vector, bm25,
                                 k=PRODUCTION_DEFAULTS["rrf_k"], w=PRODUCTION_DEFAULTS["vector_w"])
            union_ids = {item["id"] for item in fused[:PRODUCTION_DEFAULTS["recall_k"]]}

        union_items = []
        by_id = {item["id"]: item for item in vector + bm25}
        for doc_id in union_ids:
            if doc_id in by_id:
                union_items.append(by_id[doc_id])

        rerank_scores, status = _rerank_scores(reranker, question, union_items)

    return {"question": question, "gold": gold, "alias": alias,
            "vector": vector, "bm25": bm25,
            "rerank_scores": rerank_scores, "rerank_status": status}


def fill_rerank_for_cfg(cache: dict, reranker: CrossEncoderReranker, cfg: dict) -> None:
    """最优组确定后，给没精排的缓存补算该配置所需候选的精排分（原地更新）。

    只精排「该配置的 RRF 前 recall_k」，不用为用不上的配置付精排成本。
    """
    if cfg["rerank"] == "off":
        return
    fused = weighted_rrf(cache["vector"], cache["bm25"], k=cfg["rrf_k"], w=cfg["vector_w"])
    ids = {item["id"] for item in fused[:cfg["recall_k"]]}
    by_id = {item["id"]: item for item in cache["vector"] + cache["bm25"]}
    items = [by_id[i] for i in ids if i in by_id]
    cache["rerank_scores"], cache["rerank_status"] = _rerank_scores(reranker, cache["question"], items)


def load_single_cases(split: str) -> list:
    """读 qa_v1.json，取 kind=single 且指定 split 的题（题面 + 正确片段）。

    正确片段 = gold_chunks ∪ alt_gold_chunks（rev3 补标，与基线判分同口径）。
    """
    data = json.loads(QA_EXAM.read_text(encoding="utf-8"))
    cases = data if isinstance(data, list) else data.get("cases", data)
    return [
        {"question": c["question"],
         "gold": [str(g) for g in (c.get("gold_chunks") or []) + (c.get("alt_gold_chunks") or [])]}
        for c in cases if c.get("kind") == "single" and c.get("split") == split
    ]


def main() -> None:
    train_cases = load_single_cases("train")
    held_cases = load_single_cases("held_out")
    print(f"qa single: train {len(train_cases)} 题（搜索用）, held_out {len(held_cases)} 题（只看数字）")

    kb = KnowledgeBase(chroma_host="localhost", chroma_port=8001)
    kb._ensure_bm25_index()  # 预热全库 BM25 索引（懒构建，只建一次）
    reranker = CrossEncoderReranker()

    # 每题两档缓存（别名 on/off）。held_out 只召回不精排（do_rerank=False），
    # 等网格搜出最优组再补算那一组的精排分——不参与搜索的题不付全量精排成本
    caches = {"train": [], "held": []}
    for group, cases in (("train", train_cases), ("held", held_cases)):
        for i, case in enumerate(cases, 1):
            t0 = time.perf_counter()
            for alias in (False, True):
                caches[group].append(build_case_cache(kb, reranker, case["question"], case["gold"],
                                                      alias, do_rerank=(group == "train")))
            print(f"  cache {group} {i}/{len(cases)} ({time.perf_counter() - t0:.1f}s)")

    # 精排降级检查：降级时精排分是词重叠兜底，数字与生产（真精排）不可比
    # （只看 train：held 此时还没精排，状态是 not_needed）
    degraded = [c for c in caches["train"] if c["rerank_status"] != RERANK_CROSS_ENCODER]
    if degraded:
        print(f"[警告] {len(degraded)} 份缓存精排非 CrossEncoder（状态异常），结果可能失真")

    # ── 单因子：别名收益（保持生产默认参数不变，只开关别名） ──
    def eval_alias(alias: bool) -> dict:
        rows = [score_case(retrieve_top5(c["vector"], c["bm25"], PRODUCTION_DEFAULTS, c["rerank_scores"]),
                           c["gold"]) for c in caches["train"] if c["alias"] == alias]
        return aggregate(rows)

    alias_off_default = eval_alias(False)
    alias_on_default = eval_alias(True)

    # ── 网格搜索（别名 on，train 上选最优） ──
    configs = gen_configs()
    print(f"网格组合数: {len(configs)}")
    results = []
    alias_on_caches = [c for c in caches["train"] if c["alias"]]
    for cfg in configs:
        rows = [score_case(retrieve_top5(c["vector"], c["bm25"], cfg, c["rerank_scores"]), c["gold"])
                for c in alias_on_caches]
        agg = aggregate(rows)
        results.append({"cfg": cfg, **agg})

    results.sort(key=lambda r: r["mrr"], reverse=True)
    best = results[0]

    # 生产默认值在排序中的位置（别名 on 档）
    def cfg_key(cfg: dict) -> tuple:
        return (cfg["recall_pool"], cfg["rrf_k"], cfg["vector_w"], cfg["recall_k"], cfg["rerank"])

    default_rank = next(i for i, r in enumerate(results) if r["cfg"] == PRODUCTION_DEFAULTS)
    best_off = next(r for r in results if r["cfg"]["rerank"] == "off")  # 已按 MRR 降序，第一个 off 即最好

    # 最优组放到 held_out 上看数字（不参与搜索，只看泛化；此处才补精排分）
    held_alias_on = [c for c in caches["held"] if c["alias"]]
    for c in held_alias_on:
        fill_rerank_for_cfg(c, reranker, best["cfg"])
    held_rows = [score_case(retrieve_top5(c["vector"], c["bm25"], best["cfg"], c["rerank_scores"]), c["gold"])
                 for c in held_alias_on]
    held_agg = aggregate(held_rows)

    # 每个参数的边际分布（其余参数取最优时不好算，直接给单因子均值，粗看敏感度）
    def param_sensitivity(key: str) -> dict:
        groups = defaultdict(list)
        for r in results:
            groups[r["cfg"][key]].append(r["mrr"])
        return {str(v): round(sum(fs) / len(fs), 4) for v, fs in sorted(groups.items(), key=lambda kv: str(kv[0]))}

    # ── 报告落盘（UTF-8 文件，控制台 GBK 不乱码） ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "检索调参网格搜索报告（P2）",
        f"日期: {time.strftime('%Y-%m-%d %H:%M')}",
        f"搜索集: qa single train {len(train_cases)} 题（held_out {len(held_cases)} 题不参与搜索）",
        "",
        "── 单因子：别名展开收益（生产默认参数 recall_pool=80/rrf_k=60/vector_w=1.0/recall_k=20/rerank=on）──",
        f"别名 off: MRR = {alias_off_default['mrr']}  (hit@1 {alias_off_default['hit_at_1']}, hit@5 {alias_off_default['hit_at_5']})",
        f"别名 on : MRR = {alias_on_default['mrr']}  (hit@1 {alias_on_default['hit_at_1']}, hit@5 {alias_on_default['hit_at_5']})",
        "",
        f"── 网格搜索 top 10（共 {len(configs)} 组合，按 MRR 降序）──",
    ]
    for rank, r in enumerate(results[:10], 1):
        cfg = r["cfg"]
        tag = "  ← 生产默认" if cfg == PRODUCTION_DEFAULTS else ("  ← 最优" if rank == 1 else "")
        lines.append(
            f"{rank:>2}. MRR={r['mrr']}  hit@1={r['hit_at_1']}  hit@5={r['hit_at_5']}  "
            f"pool={cfg['recall_pool']} rrf_k={cfg['rrf_k']} w={cfg['vector_w']} "
            f"recall_k={cfg['recall_k']} rerank={cfg['rerank']}{tag}"
        )
    lines += [
        "",
        f"生产默认参数（别名 on）排名: 第 {default_rank + 1} 名 / {len(results)}，MRR = {results[default_rank]['mrr']}",
        f"总体最优: MRR = {best['mrr']}（{best['cfg']}）",
        f"精排 off 档最优: MRR = {best_off['mrr']}（{best_off['cfg']}）",
        "",
        "── 单因子敏感度（各参数取值下所有组合的 MRR 均值，粗看哪个旋钮最敏感）──",
    ]
    for key in ("recall_pool", "rrf_k", "vector_w", "recall_k", "rerank"):
        sens = param_sensitivity(key)
        lines.append(f"  {key}: " + "  ".join(f"{v}={m}" for v, m in sens.items()))
    lines += [
        "",
        "── 最优组泛化（held_out 不参与搜索，只看数字）──",
        f"最优组 {best['cfg']} 在 held_out: MRR = {held_agg['mrr']}  (hit@1 {held_agg['hit_at_1']}, hit@5 {held_agg['hit_at_5']})",
    ]

    report_path = OUT_DIR / "tune_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    configs_path = OUT_DIR / "tune_configs.json"
    configs_path.write_text(json.dumps({"alias_off_default": alias_off_default,
                                        "alias_on_default": alias_on_default,
                                        "results": results,
                                        "best": best,
                                        "held_out_best": held_agg},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {report_path}")
    print(f"全部配置: {configs_path}")


if __name__ == "__main__":
    main()
