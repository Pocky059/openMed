"""
亮点：医疗意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 本地中文模型 bge-small-zh-v1.5
     （core/embedding.py 惰性单例，与 RAG/记忆库同款，零 API 成本）匹配
     常见表达；模型不可用时退字符 n-gram 哈希兜底
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底，覆盖症状/用药/预约/急症关键词

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.embedding import get_local_embedder
from core.llm_utils import create_llm_client, extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 通用医疗咨询
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    EMERGENCY  = "emergency"   # 急症/紧急情况
    MEDICATION = "medication"  # 用药咨询
    APPOINTMENT = "appointment"  # 预约挂号
    FEEDBACK   = "feedback"    # 正面反馈
    SYMPTOM_CHECK = "symptom_check"            # 症状自查/描述症状
    DEPARTMENT_GUIDE = "department_guide"      # 科室指引
    APPOINTMENT_MANAGE = "appointment_manage"  # 查询/修改预约
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"  # 预约改期
    APPOINTMENT_CANCEL = "appointment_cancel"  # 取消/退号
    MEDICAL_RECEIPT = "medical_receipt"        # 就诊证明/医疗票据
    APPOINTMENT_PAYMENT_ISSUE = "appointment_payment_issue"  # 挂号费支付异常
    MEDICATION_DOSAGE = "medication_dosage"    # 用法用量咨询
    MEDICATION_INTERACTION = "medication_interaction"  # 药物相互作用/不良反应
    HUMAN_HANDOFF = "human_handoff"            # 转人工/转诊
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我最近老是头晕，是怎么回事？", "感冒了要注意什么？", "体检报告要怎么看？"],
    IntentCategory.COMPLAINT:  ["等了三个小时也没人接诊！", "医生态度太差了！", "一直没人处理我的问题！"],
    IntentCategory.REQUEST:    ["帮我取消这次预约", "我需要换个医生", "请帮我安排复诊"],
    IntentCategory.GREETING:   ["你好", "在吗", "早上好"],
    IntentCategory.EMERGENCY:  ["胸口突然剧烈疼痛，喘不上气", "孩子高烧40度还抽搐了", "摔倒后一直流血止不住"],
    IntentCategory.MEDICATION: ["这个药要吃多久？", "感冒药可以和降压药一起吃吗？", "这个药有什么副作用？"],
    IntentCategory.APPOINTMENT: ["我要挂号看内科", "怎么预约专家号？", "预约流程是什么？"],
    IntentCategory.FEEDBACK:   ["医生态度很好！", "看病很顺利，谢谢", "这次体验很满意"],
    IntentCategory.SYMPTOM_CHECK: ["我这几天一直咳嗽还有点低烧，是什么问题？", "肚子疼是不是肠胃炎？", "皮肤起了红疹要不要紧？"],
    IntentCategory.DEPARTMENT_GUIDE: ["头痛应该挂哪个科？", "皮肤过敏看什么科室？", "小孩发烧挂儿科还是急诊？"],
    IntentCategory.APPOINTMENT_MANAGE: ["帮我查一下我的预约记录", "我的挂号信息在哪里看？", "能不能改一下预约的医生？"],
    IntentCategory.APPOINTMENT_RESCHEDULE: ["我要把预约时间改到下周", "能不能把挂号时间提前？", "预约的时间冲突了，想改期"],
    IntentCategory.APPOINTMENT_CANCEL: ["我要取消这次挂号", "退号怎么操作？", "不想去了，帮我取消预约"],
    IntentCategory.MEDICAL_RECEIPT: ["帮我开一张就诊证明", "费用清单在哪里查？", "医疗发票怎么开？"],
    IntentCategory.APPOINTMENT_PAYMENT_ISSUE: ["挂号费扣了两次", "支付失败但显示扣款了", "这个月挂号费好像多扣了"],
    IntentCategory.MEDICATION_DOSAGE: ["这个药一天吃几次？", "吃多了会有什么后果？", "空腹能不能吃这个药？"],
    IntentCategory.MEDICATION_INTERACTION: ["这两种药一起吃会有问题吗？", "吃完药后起了皮疹", "感冒药和抗生素能同时吃吗？"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请帮我转接医生"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.SYMPTOM_CHECK,
    IntentCategory.DEPARTMENT_GUIDE,
    IntentCategory.APPOINTMENT_MANAGE,
    IntentCategory.APPOINTMENT_RESCHEDULE,
    IntentCategory.APPOINTMENT_CANCEL,
    IntentCategory.MEDICAL_RECEIPT,
    IntentCategory.APPOINTMENT_PAYMENT_ISSUE,
    IntentCategory.MEDICATION_DOSAGE,
    IntentCategory.MEDICATION_INTERACTION,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.MEDICATION,
    IntentCategory.APPOINTMENT,
    IntentCategory.EMERGENCY,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.SYMPTOM_CHECK: IntentCategory.QUERY,
    IntentCategory.DEPARTMENT_GUIDE: IntentCategory.QUERY,
    IntentCategory.APPOINTMENT_MANAGE: IntentCategory.APPOINTMENT,
    IntentCategory.APPOINTMENT_RESCHEDULE: IntentCategory.APPOINTMENT,
    IntentCategory.APPOINTMENT_CANCEL: IntentCategory.APPOINTMENT,
    IntentCategory.MEDICAL_RECEIPT: IntentCategory.APPOINTMENT,
    IntentCategory.APPOINTMENT_PAYMENT_ISSUE: IntentCategory.APPOINTMENT,
    IntentCategory.MEDICATION_DOSAGE: IntentCategory.MEDICATION,
    IntentCategory.MEDICATION_INTERACTION: IntentCategory.MEDICATION,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.EMERGENCY,
}

# 紧急关键词：CRITICAL 覆盖典型急症红旗症状，触发急症前置安全门控
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: [
        "紧急", "emergency", "urgent", "asap", "立刻",
        "胸痛", "胸口疼", "呼吸困难", "喘不上气", "休克", "大出血",
        "抽搐", "昏迷", "意识不清", "中毒", "自杀", "自残",
    ],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now", "持续高烧", "剧烈疼痛"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端医疗意图识别器。

    初始化时不加载模型；语义路的本地中文 embedding 模型（bge-small-zh-v1.5，
    core/embedding.py 的进程级惰性单例）在首次请求时加载，模板向量算一次后
    缓存复用。模型不可用时自动退字符 n-gram 哈希，三路融合不中断。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        #构造客户端（v2 阶段 0.5 双协议）：按 OPENMED_LLM_PROTOCOL 决定 Anthropic
        #协议还是 OpenAI 协议，两种客户端对外接口一致（messages.create）
        self.client = create_llm_client(api_key=api_key, base_url=base_url)

        #保存变量
        self.model     = model
        self.threshold = confidence_threshold

        # 语义路固定启用（权重 70/20/10，见 _vote）：优先本地中文模型
        # （get_local_embedder 惰性加载），模型不可用（如离线测试环境）时
        # _embed_text 自动退字符 n-gram 哈希——降级发生在语义路内部，融合权重不变。
        # （曾有一个 _embedding_enabled 开关和 85/15 双路兜底分支，但从未被置 False，
        # 是死代码，已删除——语义路不可用时靠降级而非关路。）

        #准备模板向量化存储dict，先初始化为空，后面填充
        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户医疗意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # 三路意图识别
        # LLM 和 Embedding 并行（语义路不可用时在内部降级），pat 单独运行
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message))
        pat      = self._pattern_recognize(message)

        llm, emb = await asyncio.gather(llm_task, emb_task)

        #vote融合三路
        intent, confidence, source_scores = self._vote(llm, emb, pat)
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        #组装结果
        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

        # LRU 缓存，实际上是FIFO，超过一千条缓存时删除前五百条
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是医疗问诊意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如取消预约优先返回 appointment_cancel，剂量咨询优先返回 medication_dosage，科室指引优先返回 department_guide。
出现胸痛、呼吸困难、大出血、抽搐、昏迷等急症红旗症状时，优先返回 emergency。

示例:
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()

        # 先匹配细粒度关键词
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工", "转专家", "转医生"],
            IntentCategory.SYMPTOM_CHECK: ["咳嗽", "发烧", "发热", "头晕", "肚子疼", "腹泻", "皮疹", "呕吐", "乏力"],
            IntentCategory.DEPARTMENT_GUIDE: ["挂哪个科", "看什么科", "挂什么科", "应该挂号", "department"],
            IntentCategory.APPOINTMENT_CANCEL: ["取消预约", "取消挂号", "退号", "cancel appointment"],
            IntentCategory.MEDICAL_RECEIPT: ["就诊证明", "费用清单", "医疗发票", "开票", "收据", "receipt"],
            IntentCategory.APPOINTMENT_PAYMENT_ISSUE: ["挂号费扣了", "重复扣款", "支付失败", "扣费异常", "payment failed"],
            IntentCategory.APPOINTMENT_RESCHEDULE: ["改期", "改预约时间", "换个时间", "reschedule"],
            IntentCategory.MEDICATION_DOSAGE: ["怎么吃", "一天吃几次", "用法用量", "服用方法", "剂量"],
            IntentCategory.MEDICATION_INTERACTION: ["一起吃", "药物相互作用", "副作用", "不良反应", "过敏反应", "interaction"],
        }

        # 没命中再匹配粗粒度
        generic_patterns = {
            IntentCategory.EMERGENCY:  ["胸痛", "呼吸困难", "剧烈疼痛", "大出血", "抽搐", "昏迷", "喘不上气", "休克"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久", "没人管"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "是不是"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.APPOINTMENT: ["挂号", "预约", "门诊", "专家号", "appointment"],
            IntentCategory.MEDICATION: ["吃药", "用药", "服药", "药品", "说明书", "medication"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }

        #如果llm失败，看有没有emb，再没有再看pat
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        # 三路加权投票：LLM 70% / Embedding 20% / 关键词 10%
        weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf #权重*置信度，累加到对应的意图cat上

        best = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best]
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        # 精修：融合结果落在通用意图、而关键词路命中具体意图时，用关键词覆盖。
        # 注意精修结果同样要过置信度门槛（默认阈值 0.5 时恒过；自定义更高阈值时
        # 低于门槛的精修结果仍应降级为 OTHER，与模块 docstring 的承诺一致）
        if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and best_score < 0.8:
            refined_score = max(best_score, pat_conf)
            if refined_score >= self.threshold:
                source_scores["refined_by_pattern"] = pat_conf
                return pat_intent, refined_score, source_scores

        #整体置信度低于阈值，走澄清流程
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值医疗实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        return {
            "appointment_id": self._unique(
                re.findall(r"(?:预约单?号?|挂号单?号?|appointment(?:_id)?)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)
            ),
            "department": self._unique(re.findall(
                r"(内科|外科|儿科|妇科|产科|皮肤科|眼科|耳鼻喉科|口腔科|心内科|心血管内科|神经内科|"
                r"消化内科|呼吸内科|骨科|泌尿外科|急诊科?|发热门诊|中医科|精神科|肿瘤科)",
                message,
            )),
            "date": self._unique(re.findall(r"(今天|明天|后天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny))", message, re.I)),
            "drug_name": self._unique(re.findall(
                r"([一-龥A-Za-z0-9]{2,12}(?:片|胶囊|颗粒|口服液|注射液|软膏|滴剂|糖浆|栓|贴))", message
            )),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        # 一次批量编码全部模板（本地模型批量比逐条快得多），算完缓存，后续复用
        vecs = await self._embed_texts(all_texts)
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """生成单条文本向量（模板加载走批量版，这里只用于用户消息）。"""
        return (await self._embed_texts([text]))[0]

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量。

        优先用本地中文 embedding 模型（bge-small-zh-v1.5，与 RAG/记忆库同款，
        模型缓存在挂载卷 ./data/huggingface，零 API 成本、离线可用）；模型不可用
        时退化为字符 n-gram 哈希向量，保证三路融合不中断。

        已删除远端 Embedding API（voyage）分支：双协议客户端
        （AsyncAnthropic / LLMProtocolAdapter）都不暴露 embeddings 资源，
        该分支在生产从未生效，留着是死代码加花钱隐患。
        """
        model = get_local_embedder()
        if model is not None:
            try:
                # encode 是 CPU 密集同步调用，放线程池避免阻塞事件循环
                vecs = await asyncio.to_thread(
                    model.encode, texts, normalize_embeddings=True
                )
                return [[float(v) for v in vec] for vec in vecs]
            except Exception as ex:
                logger.warning(f"本地 Embedding 失败，使用字符 n-gram 哈希兜底: {ex}")

        return [self._local_embedding(text) for text in texts]

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """字符 n-gram 哈希向量——本地模型不可用时的最后兜底（字面近似，非语义）。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.EMERGENCY, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
