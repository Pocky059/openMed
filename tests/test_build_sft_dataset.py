"""
SFT 数据集转换脚本的单元测试。

覆盖三块：
1. assign_split 的确定性哈希划分（同 id 恒同侧、比例 ~20%）；
2. build_messages 的正常组装（6 条消息序列、工具名、剥键字段、确定性 tool id）；
3. 坏数据拦截（缺观察句/缺检索结果/步数不是 2 都抛 ValueError，防坏样本混进训练集）。

全部离线，不联网、不依赖模型文件。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sft_dataset import (
    EXCLUDED_IDS,
    TOOL_DEF,
    assign_split,
    build_messages,
    make_tool_result,
)


def _plan(query: str, observation: str = "先查一下", results: list | None = None) -> dict:
    """构造一步 search_plan（检索结果默认 2 条假数据，剥键后只留 id/title/content）。"""
    if results is None:
        results = [
            {"id": "chunk-a", "title": "篇A", "content": "内容A"},
            {"id": "chunk-b", "title": "篇B", "content": "内容B"},
        ]
    return {"step": 1, "query": query, "observation": observation, "tool_results": results}


def _item(**overrides) -> dict:
    """构造一条完整数据（两步 + 最终答案），overrides 可覆盖任意字段。"""
    item = {
        "seed_id": "TEST-001",
        "question": "我在吃华法林，能和三七一起吃吗？",
        "search_plan": [
            _plan("三七"),
            _plan("三七 华法林", observation="看到结果里有三七与抗凝药的条目，再查合用"),
        ],
        "final_answer": "## 现象复述\n您正在服用华法林。\n## 可核验的用药信息\n不建议同服。\n## 建议\n咨询医生。",
    }
    item.update(overrides)
    return item


# ── assign_split ──────────────────────────────────────────────────────────────

def test_assign_split_same_input_same_side():
    """同一题面永远落同一侧（哈希确定性，重跑不挪动分区）。"""
    q = "我在吃华法林，能和三七一起吃吗？"
    assert assign_split(q) == assign_split(q)
    assert assign_split(q) in ("train", "val")


def test_assign_split_ratio_about_20_percent():
    """大样本比例接近 20% 落 val（哈希均匀性，不是严格 20%）。"""
    sides = [assign_split(f"题面-{i}") for i in range(2000)]
    val = sides.count("val")
    assert 0.15 < val / 2000 < 0.25


# ── build_messages 正常组装 ───────────────────────────────────────────────────

def test_build_messages_full_sequence():
    """6 条消息序列：user 题面 → 两次 [观察句+工具调用 → 工具结果] → 最终答案。"""
    msgs = build_messages(_item())
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    # 第一条是题面
    assert msgs[0]["content"] == "我在吃华法林，能和三七一起吃吗？"
    # 第一条 assistant：text 观察句 + tool_use
    blocks = msgs[1]["content"]
    assert blocks[0] == {"type": "text", "text": "先查一下"}
    tool = blocks[1]
    assert tool["name"] == "search_knowledge_base"
    assert tool["input"] == {"query": "三七", "top_k": 5}
    assert tool["id"] == "toolu_1"
    # 第二条 assistant 的 tool id 递增
    assert msgs[3]["content"][1]["id"] == "toolu_2"
    # 最终答案是纯文本
    assert msgs[5]["content"].startswith("## 现象复述")


def test_tool_result_is_clean_json_string():
    """tool_result 的 content 是可解析 JSON 字符串，只含剥键字段。"""
    msgs = build_messages(_item())
    result_block = msgs[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_1"
    inner = json.loads(result_block["content"])
    assert inner["success"] is True
    assert inner["query"] == "三七"
    assert inner["top_k"] == 5
    assert len(inner["results"]) == 2
    assert set(inner["results"][0].keys()) == {"id", "title", "content"}


def test_tool_def_uses_production_name():
    """工具定义与生产注册名一致（agents/tools.py 的 search_knowledge_base）。"""
    assert TOOL_DEF["function"]["name"] == "search_knowledge_base"
    assert "query" in TOOL_DEF["function"]["parameters"]["properties"]
    assert TOOL_DEF["function"]["parameters"]["required"] == ["query"]


# ── 坏数据拦截 ───────────────────────────────────────────────────────────────

def test_missing_observation_raises():
    """缺观察句的步必须拦下：训练数据里 assistant 不能凭空调工具。"""
    item = _item()
    item["search_plan"][1]["observation"] = ""
    with pytest.raises(ValueError, match="观察句"):
        build_messages(item)


def test_missing_tool_results_raises():
    """缺检索结果的步必须拦下：观察句引用的结果不存在会教坏模型。"""
    item = _item()
    item["search_plan"][1]["tool_results"] = []
    with pytest.raises(ValueError, match="检索结果"):
        build_messages(item)


def test_wrong_step_count_raises():
    """步数不是 2 必须拦下：两步搜索是这批数据的核心模式。"""
    item = _item()
    item["search_plan"] = [_plan("三七")]
    with pytest.raises(ValueError, match="步数"):
        build_messages(item)


def test_excluded_ids():
    """MH-020/MH-022 在排除清单里（与考卷 held_out 撞车，不进训练集）。"""
    assert EXCLUDED_IDS == {"MH-020", "MH-022"}


# ── make_tool_result ─────────────────────────────────────────────────────────

def test_make_tool_result_json_roundtrip():
    """make_tool_result 输出的字符串能解析回原结构（训练框架要读它）。"""
    plan = _plan("三七")
    s = make_tool_result(plan)
    parsed = json.loads(s)
    assert parsed["results"] == plan["tool_results"]
