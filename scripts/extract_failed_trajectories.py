"""失败多跳轨迹提取器（阶段 2.1 第 1 步）

把基线 run 里「实体链未命中」的多跳题轨迹从 JSONL 里挖出来存档，
并解析出每题实际用了哪些查询词——这些是黄金种子的改写素材，
也是阶段 4「训练前后轨迹对比」的原始证据（轨迹文件按天轮转，现在不存会丢）。

输入：
  - data/eval/results/baseline-20261013-rev2/cases.jsonl   （失败清单：failure_reason 开头「实体链未命中」）
  - data/eval/exams/multihop_v1.json                       （题面 + 期望实体链）
  - data/trajectories/*.jsonl                              （按 request_id 找轨迹）

输出：
  - data/sft/multihop/failed_trajectories/{case_id}.json   （原始轨迹存档，一字不改）
  - data/sft/multihop/failed_trajectories/_manifest.json   （汇总：题面/查询词/未命中节点/实体链）

注意：轨迹落盘时 SDK 对象块被 default=str 兜底成了 repr 字符串，
所以 assistant 消息里的 tool_use 是 "ToolUseBlock(id=..., input={...}, name=...)" 文本，
解析查询词要从 repr 里提取 input 字典，不能按结构化块读。
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "eval" / "results" / "baseline-20261013-rev2" / "cases.jsonl"
EXAM = ROOT / "data" / "eval" / "exams" / "multihop_v1.json"
TRAJ_DIR = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "sft" / "multihop" / "failed_trajectories"

# 块 repr 例：ToolUseBlock(id='call_00_xxx', input={'query': '...', 'top_k': 5}, name='search_knowledge_base')
_REPR_INPUT = re.compile(r"input=(\{.*?\}), name=")


def parse_tool_use_input(block_text: str) -> dict | None:
    """从 tool_use 块的 repr 字符串里提取 input 字典。

    做法：正则定位 input={...} 段（input 后面一定是 name= 收尾，惰性匹配够用），
    再用 ast.literal_eval 解析成 dict——literal_eval 只认字面量，不会执行任意代码。
    非 tool_use 块（thinking/text）或格式对不上时返回 None。
    """
    if "ToolUseBlock" not in block_text:
        return None
    m = _REPR_INPUT.search(block_text)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None


def extract_queries(messages: list) -> list[str]:
    """按消息顺序抽出全部搜索查询词（只认 search_knowledge_base 工具）。

    tool_use 块在轨迹里是 repr 字符串（见模块 docstring），遍历 assistant 消息的
    content 列表，逐个块解析。一次消息里可能并行调多个搜索（模型一发多条），
    顺序即 repr 里的顺序。
    """
    queries: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, str):
                continue
            inp = parse_tool_use_input(block)
            if inp is not None and inp.get("query"):
                queries.append(inp["query"])
    return queries


def load_exam_map() -> dict[str, dict]:
    """考卷题面 + 期望实体链，按题号索引。"""
    exam = json.loads((EXAM).read_text(encoding="utf-8"))
    return {c["id"]: c for c in exam}


def main() -> None:
    exam_map = load_exam_map()
    # 筛失败题：exam=multihop 且失败原因以「实体链未命中」开头
    failed_rows = []
    with CASES.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("exam") == "multihop" and str(row.get("failure_reason", "")).startswith("实体链未命中"):
                failed_rows.append(row)
    failed_rows.sort(key=lambda r: r["case_id"])
    print(f"失败题 {len(failed_rows)} 条")

    # 扫全部日期文件收集轨迹。cases.jsonl 里的 request_id 是 8 位短码，
    # 轨迹文件里的 request_id 是长格式（短码是它的前缀），所以按前缀匹配。
    all_trajs: list[dict] = []
    for path in sorted(TRAJ_DIR.glob("trajectories-*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                all_trajs.append(json.loads(line))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for row in failed_rows:
        case_id = row["case_id"]
        exam = exam_map[case_id]
        rid = row["request_id"]
        # 前缀匹配：轨迹的 request_id 以 cases 里的 8 位短码开头
        traj = next(
            (t for t in all_trajs
             if str(t.get("request_id") or "").startswith(rid)),
            None,
        )
        if traj is None:
            print(f"  [缺轨迹] {case_id} request_id={rid}")
            manifest.append({"case_id": case_id, "missing_trajectory": True})
            continue

        # 原始轨迹存档（一字不改，阶段 4 对比用）
        (OUT_DIR / f"{case_id}.json").write_text(
            json.dumps(traj, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        queries = extract_queries(traj.get("messages", []))
        manifest.append({
            "case_id": case_id,
            "split": row["split"],
            "question": exam["question"],
            "entity_chain": [
                {"node": n["node"], "doc_id": n["doc_id"], "title": n["title"]}
                for n in exam["entity_chain"]
            ],
            "failure_reason": row["failure_reason"],
            "search_queries": queries,
            "request_id": rid,
        })
        print(f"  [OK] {case_id} 查询 {len(queries)} 条")

    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"存档与汇总已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
