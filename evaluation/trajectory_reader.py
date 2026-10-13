"""
轨迹判分读取器（v2 阶段 1 第 5 步）。

核心问题：判分器要吃轨迹数据，但轨迹 JSONL 是"原样落盘"格式——
检索结果的 chunk id 藏在 messages 视图的 tool_result 块（JSON 字符串）里，
steps 视图只有运维标志（cached/reranked 等）不含检索内容。判分器各自手写
解析的话，同一段逻辑会复制多份，且实体链硬匹配依赖 chunk id 必须解析正确。

本模块的答案：
  1. 目录扫描建索引 —— trajectories-*.jsonl 逐行读，按 request_id 索引；
     同一 request_id 重跑多次（基线 CLI 重试/续跑）只留 timestamp 最新一条
  2. 结构化解析 —— Trajectory dataclass 固定三视图（meta/messages/steps）；
     坏行告警跳过（轨迹采集是尽力而为旁路，判分不能被一行坏数据中断）
  3. 检索结果提取 —— parse_search_results 把 tool_result 块里的 JSON
     解析成 SearchResultItem{id,title,content,score,chunk}，供实体链
     硬匹配；parse_search_results_by_call 按 tool_use_id 分组，供
     "每次搜索命中没命中"这类按调用粒度的统计
  4. 数据质量断言 —— assert_eval_data_quality 校验评测模式下
     cached/rerank_degraded 必须全 False、检索必须成功，违规抛异常，
     基线 CLI 捕获后中止整个 run（宁可重跑，不产脏基线）
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class SearchResultItem:
    """一次检索返回的一条资料片段（字段与 search_knowledge_base 返回项一致）。

    id = chunk id（md5 公式，与 data/eval/chunk_index.json 同源）——判分器
    拿它和考卷 entity_chain 的 doc_id 做硬匹配。
    """

    id: str
    title: str = ""
    content: str = ""
    score: float = 0.0
    chunk: int = 0


@dataclass
class Trajectory:
    """一条已解析轨迹（三视图，与 trajectory_logger 落盘结构一一对应）。"""

    trajectory_id: str
    request_id: str
    timestamp: str
    meta: Dict[str, Any]
    messages: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """从落盘 JSON 构造；缺必填键抛 ValueError（读取端决定跳过还是中止）。"""
        for key in ("trajectory_id", "request_id", "timestamp", "meta", "messages", "steps"):
            if key not in data:
                raise ValueError(f"轨迹缺必填键: {key}")
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            request_id=str(data["request_id"]),
            timestamp=str(data["timestamp"]),
            meta=dict(data["meta"]),
            messages=list(data["messages"]),
            steps=list(data["steps"]),
        )

    def search_calls(self, tool_name: str = "search_knowledge_base") -> List[Dict[str, Any]]:
        """steps 视图里该工具的调用记录（含 args 里的 query）。

        多跳判分的"检索步数"用这个数：只数 search_knowledge_base——
        其他工具（build_medication_plan 等）不参与检索规划，不算步数。
        """
        calls: List[Dict[str, Any]] = []
        for step in self.steps:
            for tool_call in step.get("tool_calls", []):
                if tool_call.get("tool_name") == tool_name:
                    calls.append(tool_call)
        return calls

    def search_results(self) -> List[SearchResultItem]:
        """本轨迹全部检索结果（跨调用合并、按 id 去重），实体链覆盖判分直接用。"""
        return parse_search_results(self.messages)

    def final_answer(self) -> str:
        """messages 视图里最后一条 kind=final 消息的 assistant 文本（判分用最终回答）。

        两种落盘形态都认（2026-10-13 pilot 实测生产轨迹是前者，只认后者
        会把正常回答判成「无最终回答」）：
          - 纯字符串：classify_messages 对 assistant 纯文本原样保留
          - 内容块列表：多段 text 块按序拼接；SDK 对象块（default=str 垃圾串）跳过
        没有 final 消息时返回空串——判分方据此记 failure_reason，不抛异常。
        """
        parts: List[str] = []
        for msg in self.messages:
            if not isinstance(msg, dict) or msg.get("kind") != "final":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts = [content]  # 覆盖式赋值，与下面一样只保留最后一条 final
            elif isinstance(content, list):
                parts = []  # 只保留最后一条 final（此前收集的作废）
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
        return "".join(parts)


def _parse_tool_result_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """把 tool_result 块的 content 字段解析成工具返回 dict；解析失败返回 None。

    两种形态都认：落盘时是 json.dumps 字符串（agent_orchestrator 构造），
    但判分器也可能吃到已经 json.loads 过的 dict（测试/未来改动），都按 dict 返回。
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("tool_result 块 JSON 解析失败，跳过: %s", raw[:100])
        return None
    return payload if isinstance(payload, dict) else None


def parse_search_results(messages: List[Dict[str, Any]]) -> List[SearchResultItem]:
    """从 messages 视图提取全部检索结果 item（保序、按 id 去重）。

    定位路径：tool_result 消息（role=user、content 是块列表）→ 每个块是
    {"type": "tool_result", "tool_use_id", "content": "<JSON 字符串>"}，
    JSON 是 search_knowledge_base 的返回 dict（agents/tools.py 构造），
    取其中的 results 数组，每项 {id, title, content, score, chunk}。

    容错设计（轨迹是尽力而为旁路落盘的，什么脏数据都可能有）：
      - 非 dict 的块跳过 —— anthropic 协议下 assistant 消息的 content 是
        SDK 对象，落盘时被 default=str 序列化成字符串，解析必须跳过而非报错
      - 块 content 是 dict（已解析）或字符串（JSON）都认
      - JSON 解不开 / 没有 results 键（非检索工具的结果）→ 跳过并告警
    """
    items: List[SearchResultItem] = []
    seen: Set[str] = set()
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            payload = _parse_tool_result_payload(block.get("content"))
            if payload is None:
                continue
            for item in payload.get("results") or []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                chunk_id = str(item["id"])
                if chunk_id in seen:
                    continue  # 多次搜索命中同一片段，覆盖判分只算一次
                seen.add(chunk_id)
                items.append(SearchResultItem(
                    id=chunk_id,
                    title=str(item.get("title") or ""),
                    content=str(item.get("content") or ""),
                    score=float(item.get("score") or 0.0),
                    chunk=int(item.get("chunk") or 0),
                ))
    return items


def parse_search_results_by_call(messages: List[Dict[str, Any]]) -> Dict[str, List[SearchResultItem]]:
    """按 tool_use_id 分组提取检索结果（每次搜索命中什么的粒度）。

    tool_precision（搜了几次、几次命中）这类按调用粒度的统计用它；
    实体链覆盖判分用合并版 parse_search_results 即可。
    """
    by_call: Dict[str, List[SearchResultItem]] = {}
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = str(block.get("tool_use_id") or "")
            if not tool_use_id:
                continue  # 没有 tool_use_id 无法归组（理论上不存在）
            payload = _parse_tool_result_payload(block.get("content"))
            if payload is None:
                continue
            for item in payload.get("results") or []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                by_call.setdefault(tool_use_id, []).append(SearchResultItem(
                    id=str(item["id"]),
                    title=str(item.get("title") or ""),
                    content=str(item.get("content") or ""),
                    score=float(item.get("score") or 0.0),
                    chunk=int(item.get("chunk") or 0),
                ))
    return by_call


class TrajectoryReader:
    """轨迹目录扫描器：把 trajectories-*.jsonl 按 request_id 建索引。

    使用方式：基线 CLI 在跑完一批题后调用 load_index() 一次，再按
    request_id 取轨迹判分。同一 request_id 重跑多次（--resume/重试）时
    只留 timestamp 最新的一条（ISO 时间字符串可直接比较大小）。
    """

    def __init__(self, log_dir: Union[str, Path] = "data/trajectories"):
        self._log_dir = Path(log_dir)

    def load_index(self) -> Dict[str, Trajectory]:
        """扫描目录下全部 trajectories-*.jsonl，按 request_id 建索引。

        坏行（JSON 解不开 / 缺必填键）告警跳过——轨迹采集是尽力而为
        旁路，判分不能因为一行坏数据中断。
        """
        index: Dict[str, Trajectory] = {}
        for path in sorted(self._log_dir.glob("trajectories-*.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    traj = self._parse_line(path.name, lineno, line)
                    if traj is None:
                        continue
                    existing = index.get(traj.request_id)
                    if existing is None or traj.timestamp >= existing.timestamp:
                        index[traj.request_id] = traj
        return index

    @staticmethod
    def _parse_line(filename: str, lineno: int, line: str) -> Optional[Trajectory]:
        """解析一行 JSONL；任何异常都告警返回 None（坏行不中断判分）。"""
        try:
            data = json.loads(line)
        except json.JSONDecodeError as ex:
            logger.warning("轨迹坏行跳过 %s:%s: %s", filename, lineno, ex)
            return None
        if not isinstance(data, dict):
            logger.warning("轨迹行不是 JSON 对象，跳过 %s:%s", filename, lineno)
            return None
        try:
            return Trajectory.from_dict(data)
        except ValueError as ex:
            logger.warning("轨迹缺必填键，跳过 %s:%s: %s", filename, lineno, ex)
            return None

    def trajectory_of(self, request_id: str) -> Optional[Trajectory]:
        """取某 request_id 的最新轨迹；每次调用都重扫目录（简单直白）。

        基线 CLI 的正确用法是一次 load_index() 拿全部，不要在循环里反复调这个。
        """
        return self.load_index().get(request_id)


def assert_eval_data_quality(trajectory: Trajectory) -> None:
    """评测模式数据质量断言：判分可信度的前提，违规抛 AssertionError。

    检索判分（实体链硬匹配）假设轨迹里的每次搜索都是"真实发生且完整执行"的：
      - cached=True          → 结果来自工具缓存回放，轨迹谎报"检索了"
        （0.2 评测模式已禁缓存，此断言是第二道防线）
      - rerank_degraded=True → Cross-Encoder 不可用降级词重叠兜底，召回质量
        与正常链路不同，检索数字会失真
      - success/result_success=False → 这次搜索失败了，自然没有结果可判

    基线 CLI 对每条参与判分的轨迹先跑本断言，违规直接中止整个 run——
    宁可重跑一遍，不产脏基线。warmup 预热轨迹不参与判分，由 CLI 按
    request_id 前缀过滤，不走本断言。
    """
    for step in trajectory.steps:
        for tool_result in step.get("tool_results", []):
            if tool_result.get("tool_name") != "search_knowledge_base":
                continue  # 非检索工具没有 cached/rerank 键，也不影响检索判分
            violations: List[str] = []
            if tool_result.get("cached"):
                violations.append("cached=True（缓存回放）")
            if tool_result.get("rerank_degraded"):
                violations.append("rerank_degraded=True（精排降级兜底）")
            if not tool_result.get("success"):
                violations.append("success=False（工具调用失败）")
            if not tool_result.get("result_success"):
                violations.append("result_success=False（检索无结果）")
            if violations:
                raise AssertionError(
                    f"轨迹 {trajectory.request_id} 数据质量违规"
                    f"（step {step.get('step_id')}）: " + "；".join(violations)
                )
