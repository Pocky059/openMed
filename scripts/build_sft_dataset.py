"""SFT 数据集转换（阶段 2.1 第 7 步）：黄金种子 + 扩充数据 → 训练消息序列

每条数据组装成与生产轨迹语义同构的 messages 序列（Anthropic 协议）：

  user:      题面
  assistant: [text 观察句, tool_use search_knowledge_base {query, top_k}]
  user:      tool_result（content 为 JSON 字符串，与轨迹落盘形态一致）
  assistant: [text 观察句, tool_use]
  user:      tool_result
  assistant: final_answer

对齐点（与生产一致，缺一不可）：
  - 工具名 search_knowledge_base（agents/tools.py 注册、轨迹/判分器同款）
  - tool_result 只留 success/query/top_k/results{id,title,content}（剥键规则）
  - tool_use 的 id 确定性命名（toolu_1/toolu_2），top_k=5（回填即 top5）

三集划分：题面 SHA256 前 8 位 hex % 5 == 0 → val（~20%），其余 train。
与考卷 assign_split 同理念（哈希分区确定性、对生成顺序无关），比例不同。
held_out 沿用考卷既有分区，SFT 数据不设 held_out。

排除项：MH-020/MH-022（与考卷 held_out 撞车，用户拍板留档案不进训练集）。

用法：
  .venv/Scripts/python.exe scripts/build_sft_dataset.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "sft" / "multihop" / "seeds" / "seeds_draft.json"
EXPANDED = ROOT / "data" / "sft" / "multihop" / "expanded" / "expanded_raw.json"
OUT_FILE = ROOT / "data" / "sft" / "multihop" / "sft_dataset.jsonl"

# 与考卷 held_out 撞车的种子（2026-10-13 用户拍板：留档案、不进训练集）
EXCLUDED_IDS = {"MH-020", "MH-022"}

# 工具定义：描述与参数 schema 原文取自 agents/tools.py make_tool 注册处
TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "检索医疗知识库并返回最相关的文档片段；可用于症状分诊、用药咨询、预约挂号和急症升级场景。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            "required": ["query"],
        },
    },
}


def assign_split(question: str) -> str:
    """题面哈希划分：~20% 落 val，其余 train（同考卷 assign_split 理念）。"""
    digest = hashlib.sha256(question.encode()).hexdigest()
    return "val" if int(digest[:8], 16) % 5 == 0 else "train"


def make_tool_result(plan: dict) -> str:
    """一步检索结果的 tool_result content（JSON 字符串，剥键后只留必要字段）。"""
    payload = {
        "success": True,
        "query": plan["query"],
        "top_k": 5,
        "results": plan["tool_results"],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_messages(item: dict) -> list[dict]:
    """一条数据 → 消息序列。失败（缺观察句/缺检索结果）抛 ValueError。"""
    plans = item["search_plan"]
    if len(plans) != 2:
        raise ValueError(f"{item['seed_id']} 步数不是 2: {len(plans)}")
    for i, plan in enumerate(plans):
        if not plan.get("observation"):
            raise ValueError(f"{item['seed_id']} 第 {plan['step']} 步缺观察句")
        if not plan.get("tool_results"):
            raise ValueError(f"{item['seed_id']} 第 {plan['step']} 步缺检索结果")

    messages: list[dict] = [
        {"role": "user", "content": item["question"]},
    ]
    for i, plan in enumerate(plans):
        tool_id = f"toolu_{i + 1}"
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": plan["observation"]},
                {"type": "tool_use", "id": tool_id, "name": "search_knowledge_base",
                 "input": {"query": plan["query"], "top_k": 5}},
            ],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id,
                 "content": make_tool_result(plan)},
            ],
        })
    messages.append({"role": "assistant", "content": item["final_answer"]})
    return messages


def main() -> None:
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
    expanded = json.loads(EXPANDED.read_text(encoding="utf-8"))

    # 合并：种子 + 扩充；排除撞车种子
    items = [s for s in seeds if s["seed_id"] not in EXCLUDED_IDS] + expanded
    print(f"输入：种子 {len(seeds)} 条（排除 {sorted(EXCLUDED_IDS)} 后 {len(seeds) - len(EXCLUDED_IDS)} 条）"
          f" + 扩充 {len(expanded)} 条 = {len(items)} 条")

    rows: list[dict] = []
    errors: list[str] = []
    for item in items:
        try:
            messages = build_messages(item)
        except ValueError as ex:
            errors.append(str(ex))
            continue
        rows.append({
            "id": item["seed_id"],
            "split": assign_split(item["question"]),
            "messages": messages,
            "tools": [TOOL_DEF],
        })

    if errors:
        print(f"[警告] {len(errors)} 条组装失败：")
        for e in errors[:10]:
            print(f"  {e}")

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"输出：{len(rows)} 条（train {n_train} / val {n_val}）")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已落盘：{OUT_FILE}")


if __name__ == "__main__":
    main()
