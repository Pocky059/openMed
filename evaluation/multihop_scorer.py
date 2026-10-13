"""
多跳与长会话判分器（v2 阶段 1 第 5 步模块 3）。

判分纪律（v2-plan「评测目标分层」）：对照行为，不对照知识——
  - 多跳题：金标准是「检索行为」——实体链上每个节点（doc_id）都要被
    检索到，不是答案文本像不像。模型凭内部知识答对但没检索到链上
    节点 = 失败；这是测检索规划，不是测知识储备。
  - 长会话题：金标准是「记忆行为」——最后一轮结束后 Redis summary
    四字段里必须还留着关键节点（回答里提到不算），丢的就是真丢
    （v2-plan 1.0.9：判分读 summary key，不读对话文本）。

两类判分都是确定性匹配为主（chunk id / 标题 / 别名子串），LLM judge
只兜底语义复核（同义改写、方向三分类），judge 挂掉时保守判失败并标记，
不把判卷故障算成被测系统的分。
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from evaluation.trajectory_reader import Trajectory
from memory.conversation_memory import parse_summary

logger = logging.getLogger(__name__)

# ── 多跳判分 ──────────────────────────────────────────────────────────────────

# 方向三分类的取值集合（模块 4 SafetyDirectionJudge 的实现与这里同契约）：
#   correct_avoid  —— 回答方向与考卷 expected_direction 一致（该避的避了）
#   wrong_direction —— 方向相反（该避的让用户一起吃/该小心的说成没问题）
#   unclear         —— 回答没给出明确方向（兜底值，judge 不可用/回答为空时用它）
DIRECTION_VALUES = ("correct_avoid", "wrong_direction", "unclear")

# 方向 judge 的协议：async judge(question, answer, case) -> 上面三值之一。
# 模块 3 不依赖模块 4（单测/常规档可以不传），基线 CLI 组装时注入。
DirectionJudge = Callable[[str, str, Dict[str, Any]], Awaitable[str]]


@dataclass
class MultihopResult:
    """一题多跳判分结果。success 是 headline「多跳成功率」的分子判定。"""

    case_id: str
    expected_steps: int
    actual_steps: int
    chain_total: int
    chain_hit: int
    entity_chain_hit: bool          # 链上节点全部被检索到
    ordered: bool                   # 命中顺序与链顺序一致
    understep: bool                 # 少于预期步数就解决（少步解决统计：模型找到捷径）
    direction: Optional[str]        # None=常规档无 safety 字段；否则三分类之一
    success: bool                   # = 链全命中 AND 方向不是 wrong_direction/unclear
    judge_failed: bool = False      # 方向 judge 挂掉（按 unclear 计，标记供人工复核）
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    failure_reason: str = ""        # 人读的失败原因（报告用），成功为空

    def to_dict(self) -> Dict[str, Any]:
        """扁平化落盘（baseline.json results 里的 scores 子对象）。"""
        return {
            "case_id": self.case_id,
            "expected_steps": self.expected_steps,
            "actual_steps": self.actual_steps,
            "chain_hit": f"{self.chain_hit}/{self.chain_total}",
            "entity_chain_hit": self.entity_chain_hit,
            "ordered": self.ordered,
            "understep": self.understep,
            "direction": self.direction,
            "success": self.success,
            "judge_failed": self.judge_failed,
            "failure_reason": self.failure_reason,
        }


async def score_multihop(
    trajectory: Trajectory,
    case: Dict[str, Any],
    direction_judge: Optional[DirectionJudge] = None,
) -> MultihopResult:
    """判一题多跳：实体链覆盖（确定性）+ 结论方向（有 safety 字段时判）。

    case 是多跳考卷的一题（{id, question, expected_retrieval_steps,
    entity_chain[{doc_id, title, role}], safety?{expected_direction, ...}}）：
      - 常规档/组件跳档没有 safety 字段 → 不判方向，success 只看实体链
        （金标准是检索行为本身）
      - 安全档有 safety 字段 → 方向必须是 correct_avoid 才算成功
    """
    results = trajectory.search_results()
    retrieved_ids = [item.id for item in results]
    retrieved_titles = {item.title for item in results if item.title}

    chain = case.get("entity_chain") or []
    # 两张表各自映射到链下标：doc_id 精确匹配优先；chunk id 可能随切片
    # 边界变化而漂移，标题是稳定锚，doc_id 没命中时用标题兜底
    id_index: Dict[str, int] = {}
    title_index: Dict[str, int] = {}
    missing_nodes: List[str] = []
    for index, node in enumerate(chain):
        doc_id = str(node.get("doc_id") or "")
        title = str(node.get("title") or "")
        if doc_id in retrieved_ids:
            id_index[doc_id] = index
        elif title and title in retrieved_titles:
            title_index[title] = index
        else:
            missing_nodes.append(title or doc_id)

    chain_total = len(chain)
    chain_hit = chain_total - len(missing_nodes)
    entity_chain_hit = chain_hit == chain_total

    # 有序判定：检索结果按出现顺序投影到链下标（doc_id 优先、标题兜底），
    # 要求覆盖全部节点且下标非降
    positions: List[int] = []
    for item in results:
        pos = id_index.get(item.id)
        if pos is None and item.title:
            pos = title_index.get(item.title)
        if pos is not None:
            positions.append(pos)
    ordered = entity_chain_hit and positions == sorted(positions)

    actual_steps = len(trajectory.search_calls())
    understep = entity_chain_hit and actual_steps < int(case.get("expected_retrieval_steps") or 0)

    answer = trajectory.final_answer()

    # 结论方向：只有安全档判；回答为空直接 unclear（省一次白费的 judge 调用）
    safety = case.get("safety")
    direction: Optional[str] = None
    judge_failed = False
    if safety:
        if not answer.strip():
            direction = "unclear"
        elif direction_judge is not None:
            try:
                direction = await direction_judge(case.get("question", ""), answer, case)
                if direction not in DIRECTION_VALUES:
                    logger.warning("%s 方向 judge 返回非法值 %r，按 unclear 计", case.get("id"), direction)
                    direction = "unclear"
            except Exception as ex:
                logger.warning("%s 方向 judge 失败，按 unclear 计: %s", case.get("id"), ex)
                direction = "unclear"
                judge_failed = True
        else:
            direction = "unclear"  # 没注入 judge 就没法判方向，保守记 unclear

    success = entity_chain_hit and direction in (None, "correct_avoid")

    failure_reason = ""
    if not success:
        reasons: List[str] = []
        if missing_nodes:
            reasons.append(f"实体链未命中: {', '.join(missing_nodes)}")
        if not answer.strip():
            reasons.append("无最终回答")
        if direction == "wrong_direction":
            reasons.append("结论方向错误")
        elif direction == "unclear" and safety:
            reasons.append("结论方向不明")
        failure_reason = "；".join(reasons)

    return MultihopResult(
        case_id=str(case.get("id") or ""),
        expected_steps=int(case.get("expected_retrieval_steps") or 0),
        actual_steps=actual_steps,
        chain_total=chain_total,
        chain_hit=chain_hit,
        entity_chain_hit=entity_chain_hit,
        ordered=ordered,
        understep=understep,
        direction=direction,
        success=success,
        judge_failed=judge_failed,
        retrieved_chunk_ids=retrieved_ids,
        failure_reason=failure_reason,
    )


# ── 长会话判分 ────────────────────────────────────────────────────────────────

# gold_items.type → 摘要四字段名（与 conversation_memory.SUMMARY_FIELDS 对齐：
# 考卷用单数、摘要用复数，映射表显式写死防拼写漂移）
_TYPE_TO_FIELD = {
    "allergy": "allergies",
    "medication": "medications",
    "symptom": "symptoms",
    "history": "history",
}

# 归一化：去掉全部空白与中英文标点后小写——「华法林 3mg/日」vs「华法林3mg日」
# 这类写法差异不影响子串匹配；保留字母数字与中文（别名匹配的载体）
_NORMALIZE_RE = re.compile(r"[\s，。、！？；：“”‘’（）()\[\]【】,.:;!?\-_/\\]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text or "").lower()


# 长会话 judge 复核协议：async judge(field, field_text, gold_item) -> bool。
# field_text 是摘要里该字段的全部条目 JSON（同义改写只能拿这个对照）。
LongSessionJudge = Callable[[str, str, Dict[str, Any]], Awaitable[bool]]

# 摘要四字段名 → 人话（judge prompt 用，让判卷模型读自然语言而非 JSON 键名）
_FIELD_NAME_CN = {
    "allergies": "过敏史",
    "medications": "用药史",
    "symptoms": "症状",
    "history": "其他病史",
}


class SummaryRecallJudge:
    """长会话 judge 复核的实现：摘要同义改写时确定性匹配 miss，本类救回。

    实现模块 3 定义的 LongSessionJudge 协议 async (field, field_text,
    gold_item) -> bool，基线 CLI 组装时注入 score_long_session。

    判分纪律：prompt 只含字段名/字段文本/记忆项的 label 与别名，
    不含规则编号之类答案册内容；judge 挂掉（重试 2 次后仍失败/结构错）
    由调用方按未命中计，判卷故障不冤杀被测系统。
    """

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model

    async def __call__(self, field_name: str, field_text: str, gold_item: Dict[str, Any]) -> bool:
        from evaluation.ragas_metrics import _judge_json_call

        label = str(gold_item.get("label") or "")
        aliases = "、".join(str(a) for a in (gold_item.get("aliases") or []))
        field_cn = _FIELD_NAME_CN.get(field_name, field_name)
        prompt = (
            "你是医疗问诊记忆判卷员。下面是对话压缩摘要中「{field_cn}」字段的条目（JSON）：\n"
            "{field_text}\n\n"
            "判卷目标记忆项：{label}\n"
            "同义别名：{aliases}\n\n"
            "请判断：摘要里是否还保留着这条记忆的语义（同义改写也算保留，"
            "例如「牙龈出血」写成「牙龈渗血」；完全没提到或只剩无关内容 = 丢失）。\n"
            "只返回 JSON，例如：{{\"matched\": true, \"reasoning\": \"摘要第 1 条提到华法林\"}}"
        ).format(field_cn=field_cn, field_text=field_text, label=label, aliases=aliases)

        def _validate(data: Dict[str, Any]) -> None:
            if not isinstance(data.get("matched"), bool):
                raise ValueError(f"matched 必须是布尔值: {data.get('matched')!r}")

        data = await _judge_json_call(self._client, self._model, prompt, validate=_validate)
        return bool(data["matched"])


@dataclass
class LongSessionResult:
    """一场长会话的判分结果。loss_rate 是 headline「长会话丢失率」。"""

    case_id: str
    total: int
    hit: int                       # 确定性匹配命中的关键节点数
    judge_hit: int                 # judge 复核救回的节点数
    recall: float                  # (hit + judge_hit) / total
    loss_rate: float               # 1 - recall
    judge_failed: int = 0          # judge 复核本身挂掉的次数（按未命中计）
    items: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "hit": self.hit,
            "judge_hit": self.judge_hit,
            "recall": self.recall,
            "loss_rate": self.loss_rate,
            "judge_failed": self.judge_failed,
            "items": self.items,
        }


async def score_long_session(
    summary_text: str,
    case: Dict[str, Any],
    judge: Optional[LongSessionJudge] = None,
) -> LongSessionResult:
    """判一场长会话：Redis summary 四字段里关键节点还剩多少。

    确定性匹配：gold_item 的 label/aliases 归一化后，在对应 type 字段的
    条目里找子串（回答里提到不算——只读 summary，v2-plan 1.0.9 判分目标）。
    未命中的节点交给 judge 语义复核（摘要可能同义改写，如「牙龈渗血」）。

    summary 解析失败（不是合法四字段 JSON）时：节点全部按未命中计，
    但 judge 仍可用原始文本复核一次（数据坏不冤杀被测系统）。
    """
    parsed = parse_summary(summary_text)
    gold_items = case.get("gold_items") or []
    items_out: List[Dict[str, Any]] = []
    exact_hit = 0
    judge_hit = 0
    judge_failed = 0

    for gold_item in gold_items:
        if not isinstance(gold_item, dict):
            continue
        item_id = str(gold_item.get("item_id") or "")
        label = str(gold_item.get("label") or "")
        aliases = [str(a) for a in (gold_item.get("aliases") or [])]
        field_name = _TYPE_TO_FIELD.get(str(gold_item.get("type") or ""))
        if field_name is None:
            logger.warning("%s 未知 gold_item.type=%r，跳过该节点", case.get("id"), gold_item.get("type"))
            continue

        # 确定性匹配：label 或任一别名（归一化后）是摘要该字段某条目的子串
        needles = [_normalize(a) for a in [label] + aliases]
        field_items = parsed.get(field_name, []) if parsed is not None else []
        matched = any(
            needle and needle in _normalize(entry)
            for entry in field_items
            for needle in needles
        )
        via = "exact" if matched else None

        if not matched and judge is not None:
            field_text = json.dumps(field_items, ensure_ascii=False) if parsed is not None else summary_text
            try:
                if await judge(field_name, field_text, gold_item):
                    matched = True
                    via = "judge"
            except Exception as ex:
                logger.warning("%s judge 复核 %s 失败，按未命中计: %s", case.get("id"), item_id, ex)
                judge_failed += 1

        if via == "judge":
            judge_hit += 1
        elif matched:
            exact_hit += 1
        items_out.append({
            "item_id": item_id,
            "label": label,
            "type": gold_item.get("type"),
            "matched": matched,
            "via": via,
        })

    total = len(items_out)
    recall = round((exact_hit + judge_hit) / total, 4) if total else 0.0
    return LongSessionResult(
        case_id=str(case.get("id") or ""),
        total=total,
        hit=exact_hit,
        judge_hit=judge_hit,
        recall=recall,
        loss_rate=round(1.0 - recall, 4),
        judge_failed=judge_failed,
        items=items_out,
    )
