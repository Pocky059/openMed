"""
生成质量与检索质量指标（v2 阶段 1 第 5 步，参考 ragas 思想手写）。

为什么要手写而不是装 ragas 库（CLAUDE.md 决策）：
  - 零新依赖：ragas 会拉一整套 LLM 评测框架，本项目评测只需要三个指标
  - 吃轨迹数据：faithfulness 的"检索上下文"直接来自本系统轨迹里的
    检索结果，装库反而要写适配层
  - 公式可复算可审计：指标怎么算出来的写在代码里，面试可以逐行讲

指标三分工（对齐 ragas 的三个核心指标）：
  1. Faithfulness（忠实度）—— 回答里的每句话能不能从本次检索到的资料
     片段推断出来。防的是"模型编造资料里没有的医疗结论"，医疗场景
     阈值定 ≥0.90（见 v2-plan 1.2）。LLM 判句，一次调用判全部句子。
  2. Answer Relevancy（回答相关性）—— 回答是否在回答用户的问题。
     做法：让 LLM 从回答反推 3 个"用户可能提的问题"，再用本地中文
     embedding（bge-small-zh-v1.5）算原始问题与反推问题的余弦均值
     ——向量计算零 API 成本，只有反推问题花一次 LLM 调用。
  3. Context Precision/Recall —— ragas 里这两个是 LLM 判"检索片段
     有没有用"，主观性强；本项目退化为确定性统计（MRR / hit@k），
     检索质量按排名硬算，不用 judge（判分纪律：确定性统计为主，LLM
     判卷只兜底语义复核）。P@k/R@k/F1@k 已于 2026-10-15 rev3 移除：
     单跳题只有 1 个出题依据片段、检索固定返回 5 条，F1@5 命中也只有
     2×0.2×1/1.2 = 1/3，上限被口径锁死，看不出排序好坏。
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

# 判句切分符：中文句号/感叹号/问号/分号 + 换行；英文标点兼容
_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")

# 输入尺寸上限：防止超长回答/超多检索片段把 prompt 撑爆（费用与延迟）
_MAX_ANSWER_CHARS = 4000
_MAX_CONTEXTS = 8
_MAX_CONTEXT_CHARS = 500

# LLM 判卷重试次数（失败兜底约定见各函数 docstring）
_JUDGE_RETRIES = 2


def split_sentences(text: str) -> List[str]:
    """把回答文本按句切分（中文标点为主）。空文本返回空列表。"""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """两个向量的余弦相似度；任一为零向量（或长度不一致）返回 0.0。"""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_json_dict(raw: str) -> Dict[str, Any]:
    """从 LLM 输出里剥掉 ```json 围栏，取第一个 JSON 对象；失败抛 ValueError。

    与 evaluation/evaluator.py 的 LLMJudge 同一套路（不互相 import：
    evaluator 拉 anthropic/intent_recognizer 依赖，判分模块保持轻量）。
    """
    text = raw.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"LLM 未返回 JSON: {raw[:300]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"LLM 返回的不是 JSON 对象: {raw[:300]!r}")
    return data


async def _judge_json_call(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = _JUDGE_RETRIES,
    validate: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """一次 LLM 判卷调用 + JSON 解析 + 重试；全部失败抛最后一次异常。

    客户端接口与 evaluator 一致：client.messages.create(model=..., max_tokens=...,
    temperature=0.0, messages=[{"role": "user", "content": prompt}])。
    temperature=0 是判卷固定约定（同一输入要同一答案，指标可复现）。

    validate：可选的结构校验函数（如 supported 数组长度必须等于句数）——
    "返回结构不对"和"调用失败/JSON 解不开"是同一类坏响应，都该走重试，
    而不是被外层判 0 分。校验抛异常同样进入重试循环。
    """
    last_error: Optional[Exception] = None
    current_prompt = prompt  # 重试时可能附加错误反馈，见下
    for attempt in range(max_retries + 1):  # 首次 + max_retries 次重试
        try:
            resp = await client.messages.create(
                # thinking 关掉（extra_body 透传，SDK 直接传该参数会报 unexpected
                # keyword）：2026-10-13 诊断实测 DeepSeek 的思考块无上限膨胀——
                # 长输入判卷时 4096 个输出 token 全被思考占满、text 块一个都不剩
                # （stop_reason=max_tokens），调大额度治不了本；关掉后输出只花
                # 60 token 且 JSON 完整。判卷是结构化输出任务，不需要思考。
                model=model, max_tokens=4096, temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": current_prompt}],
            )
            data = _extract_json_dict(extract_text_content(resp.content))
            if validate is not None:
                validate(data)
            return data
        except Exception as ex:
            last_error = ex
            logger.warning("判卷调用失败（第 %s/%s 次）: %s", attempt + 1, max_retries + 1, ex)
            # 2026-10-13 全量基线实测教训：原样重发对「结构校验失败」完全无效——
            # temperature=0 下同一输入必然产出同样错的输出（三场 faithfulness
            # 三次重试全部数错句数）。校验/解析类错误把错误信息反馈给 judge，
            # 让它对着错误自我纠正；网络类异常重发原 prompt 即可（错误信息
            # 对网络问题没有纠正价值）。
            if isinstance(ex, ValueError):
                current_prompt = (
                    f"{prompt}\n\n【注意：你上一次的输出没有通过校验，错误是："
                    f"{ex}】\n请修正后只输出 JSON，不要再输出任何其他内容。"
                )
    raise last_error  # type: ignore[misc]


# ── Faithfulness ──────────────────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    """忠实度结果：支持句 / 总句。judge 全部失败时兜底 0.5 并标 judge_failed。"""

    score: float
    supported: int
    total: int
    judge_failed: bool = False


async def judge_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    client: Any,
    model: str,
) -> FaithfulnessResult:
    """逐句判"回答里的每句话能否从本次检索片段推断出来"。

    做法：一次 LLM 调用判全部句子（返回与句数等长的布尔数组），不逐句调——
    判卷成本与题目数成正比，不与句子数成正比。

    失败兜底：重试 2 次后仍失败 → score 0.5、judge_failed=True（判分纪律：
    判卷失败不判 0 分冤枉被测系统，标记出来人工复核）。
    """
    sentences = split_sentences(answer)
    if not sentences:
        return FaithfulnessResult(score=0.0, supported=0, total=0)  # 空回答无可判，0 分
    context_text = "\n---\n".join(
        (c or "")[:_MAX_CONTEXT_CHARS] for c in contexts[:_MAX_CONTEXTS]
    ) or "（无检索片段）"
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    prompt = f"""你是一个医疗问答事实核查员。下面给出用户问题、助手回答（已按句编号）和本次回答引用的知识库检索片段。

任务：逐句判断助手的每句话是否能从检索片段中推断出来（忠实性检查）。
- 能推断：句子内容在检索片段里有依据（直接陈述，或从片段合理推断）
- 不能推断：句子内容在检索片段里完全没有依据（可能是模型编造的）

只返回 JSON，键为 supported，值为与句子编号一一对应的布尔数组（长度必须等于句子总数）：
例如 {{"supported": [true, false, true]}}

用户问题: {question}

检索片段:
{context_text}

助手回答（按句编号）:
{numbered}"""
    def _validate(data: Dict[str, Any]) -> None:
        """结构校验：supported 必须是长度等于句数的布尔数组；错了走重试。"""
        verdicts = data.get("supported")
        if not isinstance(verdicts, list) or len(verdicts) != len(sentences):
            raise ValueError(
                f"supported 数组长度 {len(verdicts) if isinstance(verdicts, list) else '非法'} != 句子数 {len(sentences)}"
            )

    try:
        data = await _judge_json_call(client, model, prompt, validate=_validate)
        supported = sum(1 for v in data["supported"] if v is True)
    except Exception as ex:
        logger.warning("faithfulness 判卷失败，兜底 0.5: %s", ex)
        return FaithfulnessResult(score=0.5, supported=0, total=len(sentences), judge_failed=True)
    return FaithfulnessResult(
        score=round(supported / len(sentences), 4),
        supported=supported,
        total=len(sentences),
    )


# ── Answer Relevancy ──────────────────────────────────────────────────────────

@dataclass
class AnswerRelevancyResult:
    """回答相关性结果。LLM 反推问题失败 → score 0.0 + judge_failed；
    本地 embedding 模型不可用 → 整个指标不可算，调用方拿 None。"""

    score: float
    judge_failed: bool = False


def _default_embedder(text: str) -> Optional[List[float]]:
    """默认向量化：本地 bge-small-zh-v1.5（惰性加载）；模型不可用返回 None。

    延迟 import（core/embedding 拉 chromadb 依赖），与仓库 build_* 延迟加载惯例一致。
    """
    from core.embedding import get_local_embedder
    model = get_local_embedder()
    if model is None:
        return None
    vector = model.encode(text, normalize_embeddings=True)
    return [float(x) for x in vector]


async def judge_answer_relevancy(
    question: str,
    answer: str,
    client: Any,
    model: str,
    embedder: Optional[Callable[[str], Optional[List[float]]]] = None,
    num_questions: int = 3,
) -> Optional[AnswerRelevancyResult]:
    """回答相关性：LLM 从回答反推 N 个用户问题，与原始问题的向量余弦取均值。

    为什么用反推问题而不是直接比"问题 vs 回答"：问题和回答是不同文本
    （提问 vs 陈述），直接比向量偏题；反推出的"回答可能是在回答哪些问题"
    与原始问题同形态，余弦相似度才可比。这就是 ragas Answer Relevancy 的思路。

    嵌入用本地模型（零 API 成本）；embedder 参数留给单测注入假向量器。
    """
    embed = embedder or _default_embedder
    answer_text = (answer or "")[:_MAX_ANSWER_CHARS]
    prompt = f"""你是一个对话分析助手。下面给出一个用户问题和一个助手回答。

任务：根据助手回答的内容，生成 {num_questions} 个"用户可能提出"的原始问题——
即这个回答可能是在回答哪些问题。每个问题一句话。

只返回 JSON，键为 questions，值为字符串数组（长度必须等于 {num_questions}）：
例如 {{"questions": ["问题一", "问题二", "问题三"]}}

用户问题: {question}
助手回答: {answer_text}"""
    def _validate(data: Dict[str, Any]) -> None:
        """结构校验：questions 必须是长度等于 num_questions 的数组；错了走重试。"""
        questions = data.get("questions")
        if not isinstance(questions, list) or len(questions) != num_questions:
            raise ValueError(
                f"questions 数组长度 {len(questions) if isinstance(questions, list) else '非法'} != {num_questions}"
            )

    try:
        data = await _judge_json_call(client, model, prompt, validate=_validate)
        questions = [str(q) for q in data["questions"]]
    except Exception as ex:
        logger.warning("answer relevancy 反推问题失败，判 0 分: %s", ex)
        return AnswerRelevancyResult(score=0.0, judge_failed=True)

    base = embed(question)
    if base is None:
        logger.warning("本地 embedding 模型不可用，answer relevancy 不可算")
        return None
    similarities: List[float] = []
    for q in questions:
        vector = embed(q)
        if vector is None:
            logger.warning("反推问题向量化失败，跳过该问题")
            continue
        similarities.append(cosine_similarity(base, vector))
    if not similarities:
        return None
    return AnswerRelevancyResult(score=round(sum(similarities) / len(similarities), 4))


# ── 检索质量（确定性统计，不用 judge）────────────────────────────────────────

def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """平均倒数排名（简化版：单个查询）：第一个金标准资料排名的倒数。

    金文档排第 1 → 1.0，排第 2 → 0.5，排第 10 → 0.1，没捞到 → 0.0。
    """
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return round(1.0 / rank, 4)
    return 0.0


def hit_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> bool:
    """前 k 个结果里是否有任一金标准资料（布尔命中）。"""
    return bool(set(retrieved_ids[:k]) & set(gold_ids))
