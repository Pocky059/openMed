"""扩充数据实体链命中率统计（机器先验的报告层）

回填完成后跑：从每条已回填的 tool_results 里统计实体链 chunk 覆盖情况，
不打检索（检索结果已经在数据文件里，先验与回填合并成一遍）。

口径与判分器一致：实体链 doc_id 的 chunk id 集合 ⊆ 各步检索结果并集
才算全覆盖；至少一个节点命中算部分命中；一个不中算零命中。

输出：
  1. 总体三个数（全覆盖 / 部分命中 / 零命中）
  2. 步级命中率（第 1 步裸实体词 vs 第 2 步组合查询）——验证两步模式
  3. 逐篇 interaction 命中率
  4. 未全覆盖样本清单（seed_id + 查询词 + 缺哪些节点），供丢弃/修词决策

用法：
  .venv/Scripts/python.exe scripts/stats_expanded_hits.py
  .venv/Scripts/python.exe scripts/stats_expanded_hits.py --input <回填后的文件>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "sft" / "multihop" / "expanded" / "expanded_raw.json"
CHUNK_INDEX = ROOT / "data" / "eval" / "chunk_index.json"


def doc_to_chunk_ids(chunk_index: dict) -> dict[str, set[str]]:
    """doc_id → 它的全部 chunk id 集合（与 verify_seed_queries.py 同款）。"""
    mapping: dict[str, set[str]] = {}
    for doc_id, doc in chunk_index["docs"].items():
        mapping[doc_id] = {c["id"] for c in doc.get("chunks", [])}
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="回填后的扩充数据文件")
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    chunk_index = json.loads(CHUNK_INDEX.read_text(encoding="utf-8"))
    doc_chunks = doc_to_chunk_ids(chunk_index)

    rows: list[dict] = []
    for item in data:
        # 该条期望覆盖的 chunk id 全集（实体链所有节点）
        required: set[str] = set()
        for doc_id in item["entity_chain"]:
            required |= doc_chunks.get(doc_id, set())

        # 各步检索结果的 chunk id 并集
        union: set[str] = set()
        step_hits: list[tuple[str, set[str]]] = []
        for plan in item["search_plan"]:
            ids = {r["id"] for r in (plan.get("tool_results") or [])}
            step_hits.append((plan["query"], ids & required))
            union |= ids

        missing = required - union
        # 缺的节点里哪个 doc 的（用于报告）
        missing_docs: set[str] = set()
        for doc_id in item["entity_chain"]:
            if doc_chunks.get(doc_id, set()) & missing:
                missing_docs.add(doc_id)
        rows.append({
            "seed_id": item["seed_id"],
            "question": item["question"],
            "chain": item["entity_chain"],
            "missing": missing,
            "missing_docs": missing_docs,
            "step_hits": step_hits,
        })

    # ── 1. 总体三数 ──
    full = sum(1 for r in rows if not r["missing"])
    partial = sum(1 for r in rows if r["missing"] and len(r["missing_docs"]) < len(r["chain"]))
    zero = sum(1 for r in rows if r["missing_docs"] and len(r["missing_docs"]) == len(r["chain"]))
    # 注意：实体链为空的条目不参与分母（扩充数据 entity_chain 恒非空）
    total = len(rows)
    lines = [f"总样本: {total}"]
    lines.append(f"全覆盖（实体链全部节点命中）: {full} ({full/total:.1%})")
    lines.append(f"部分命中（至少一个节点命中）: {partial} ({partial/total:.1%})")
    lines.append(f"零命中（一个节点都没中）: {zero} ({zero/total:.1%})")
    lines.append("")

    # ── 2. 步级命中率 ──
    step1_hit = sum(1 for r in rows if r["step_hits"] and r["step_hits"][0][1])
    step2_hit = sum(1 for r in rows if len(r["step_hits"]) > 1 and r["step_hits"][1][1])
    lines.append(f"第 1 步（裸实体词）至少命中一个节点: {step1_hit}/{total} ({step1_hit/total:.1%})")
    lines.append(f"第 2 步（组合查询）至少命中一个节点: {step2_hit}/{total} ({step2_hit/total:.1%})")
    lines.append("")

    # ── 3. 逐篇 interaction 命中率 ──
    per_inter: dict[str, list[dict]] = {}
    for r in rows:
        inter = next((d for d in r["chain"] if d.startswith("INTERACTION-")), "?")
        per_inter.setdefault(inter, []).append(r)
    lines.append("逐篇命中率（全覆盖/总数）:")
    for inter in sorted(per_inter):
        rs = per_inter[inter]
        f = sum(1 for r in rs if not r["missing"])
        lines.append(f"  {inter}: {f}/{len(rs)}")
    lines.append("")

    # ── 4. 未全覆盖清单 ──
    bad = [r for r in rows if r["missing"]]
    lines.append(f"未全覆盖样本清单（{len(bad)} 条）:")
    for r in sorted(bad, key=lambda x: (len(x["missing"]), x["seed_id"])):
        queries = " / ".join(q for q, _ in r["step_hits"])
        hit_per_step = " ; ".join(
            f"步{qi+1}「{q}」命中{len(h)}个" for qi, (q, h) in enumerate(r["step_hits"])
        )
        lines.append(f"  {r['seed_id']} 缺 {r['missing_docs']}")
        lines.append(f"    Q: {r['question'][:60]}")
        lines.append(f"    {hit_per_step}")
    lines.append("")

    out = ROOT / "data" / "sft" / "multihop" / "expanded" / "hit_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out}")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
