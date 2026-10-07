"""
Cross-Encoder 精排器。

检索优化链路的最后一环：BM25 + 向量混合召回 → RRF 融合 → Top-K 截断 → 这里的 Cross-Encoder 精排。

Bi-Encoder（ChromaDB 内置的向量检索）把 query 和文档分别编码成向量再比余弦相似度，速度快但精度有限；
Cross-Encoder 把 query 和文档拼在一起送入模型联合编码，能建模两者的细粒度交互，精度更高但更慢，
因此只在混合召回已经把候选集缩小到几十条之后使用，兼顾效果和延迟。

模型不可用（未安装 sentence-transformers 或模型下载失败）时，
优雅降级为词重叠打分，保证检索链路不因为精排模块故障而中断。
词重叠打分基于 jieba 分词（_tokenize）；若 jieba 也未安装，进一步退化为逐字符切分——
两级降级都不会让检索链路中断。
"""
import logging
import os
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# huggingface_hub 在 import 时读取该环境变量（值固定到常量），所以尽量早设置：
# 给下载的"死连接"加 30s 读超时。注意这只防连接卡死，不防"慢但活着"的下载，
# 慢下载由 _ensure_model 的墙钟超时兜底。
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

# 首次加载 Cross-Encoder 模型（含从 HuggingFace 下载 ~1.1GB）的墙钟超时（秒）。
# 可用环境变量覆盖。超时后本次请求降级为词重叠精排，下载在后台线程继续。
_LOAD_TIMEOUT_S = float(os.getenv("OPENMED_RERANKER_LOAD_TIMEOUT", "120"))

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

_DEFAULT_MODEL = "BAAI/bge-reranker-base"

# rerank_with_status() 返回的状态常量（供轨迹/评测区分精排真实性与降级）
RERANK_CROSS_ENCODER = "cross_encoder"       # Cross-Encoder 模型真实精排
RERANK_LEXICAL_FALLBACK = "lexical_fallback" # 模型不可用/推理失败，降级为词重叠打分
RERANK_SKIPPED = "skipped"                   # 候选数量不足，未执行精排


class CrossEncoderReranker:
    """
    Cross-Encoder 精排器，懒加载模型，加载或推理失败时自动降级为词重叠打分。
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model: Any = None
        self._load_failed = False
        self._bg_box: List[Any] = []   # 后台加载线程完成后把模型放进这里，等待接管
        self._bg_error: Any = None     # 后台线程快速失败的异常（区分"超时"与"真失败"，日志用）

    def _ensure_model(self) -> bool:
        """
        懒加载 Cross-Encoder 模型；只尝试加载一次，失败后不再重复尝试。

        模型首次使用会从 HuggingFace 下载（~1.1GB），网络慢时可能耗时很久。
        因此加载放在后台线程，主线程最多等 _LOAD_TIMEOUT_S 秒：
          - 超时：本次调用降级为词重叠精排；下载线程继续跑，
            完成后 _bg_box 被填充，后续调用自动接管、恢复正常精排
          - 加载异常：同样降级（词重叠），且不再重试
        """
        if self._model is not None:
            return True
        # 上次超时的后台下载完成了，接管模型并解除降级
        if self._bg_box:
            self._model = self._bg_box[0]
            self._bg_box = []
            self._load_failed = False
            logger.info(f"Cross-Encoder 精排模型已加载（后台下载完成）: {self._model_name}")
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder

            def _load() -> None:
                try:
                    self._bg_box.append(CrossEncoder(self._model_name))
                except Exception as ex:
                    # 记录到 _bg_error：线程快速失败时日志应如实说"加载失败、不会恢复"，
                    # 而不是谎报"超过 120s、后台继续下载"
                    self._bg_error = ex
                    logger.warning(f"Cross-Encoder 后台加载失败，持续使用词重叠精排: {ex}")

            thread = threading.Thread(target=_load, daemon=True, name="reranker-model-loader")
            thread.start()
            thread.join(timeout=_LOAD_TIMEOUT_S)

            if not self._bg_box:
                # 超时或加载失败：本次降级
                self._load_failed = True
                if self._bg_error is not None:
                    logger.warning("Cross-Encoder 模型加载失败（不再重试），降级为词重叠精排")
                else:
                    logger.warning(
                        f"Cross-Encoder 模型加载超过 {_LOAD_TIMEOUT_S:.0f}s，本次降级为词重叠精排；"
                        f"模型将在后台继续下载，完成后自动恢复精排"
                    )
                return False
            self._model = self._bg_box[0]
            self._bg_box = []
            logger.info(f"Cross-Encoder 精排模型已加载: {self._model_name}")
            return True
        except Exception as ex:
            logger.warning(f"Cross-Encoder 模型加载失败，降级为词重叠精排: {ex}")
            self._load_failed = True
            return False

    def rerank(self, query: str, items: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对召回候选集重新打分排序，取 Top-K（兼容旧接口，丢弃状态）。

        items 中每个元素需含 "content" 字段（用于和 query 联合编码）。
        需要区分"真精排 / 降级 / 未精排"时请用 rerank_with_status()。
        """
        return self.rerank_with_status(query, items, top_k)[0]

    def rerank_with_status(
        self, query: str, items: List[Dict[str, Any]], top_k: int = 5
    ) -> tuple:
        """
        精排并返回 (结果列表, 状态)。

        状态（模块级常量）：
          - RERANK_CROSS_ENCODER：Cross-Encoder 模型真实精排
          - RERANK_LEXICAL_FALLBACK：模型不可用或推理失败，降级为词重叠打分
          - RERANK_SKIPPED：候选数量不足，未执行精排

        为什么要把状态暴露出来：轨迹记录（v2 阶段 0.1）需要忠实标注 reranked，
        否则评测会误把词重叠降级当成精排生效，数字失真。
        """
        if not items:
            return [], RERANK_SKIPPED
        if len(items) <= top_k and len(items) <= 1:
            return items[:top_k], RERANK_SKIPPED

        if self._ensure_model():
            try:
                pairs = [(query, item.get("content", "")) for item in items]
                scores = self._model.predict(pairs)
                ranked = sorted(zip(items, scores), key=lambda pair: float(pair[1]), reverse=True)
                result = []
                for item, score in ranked[:top_k]:
                    entry = dict(item)
                    entry["rerank_score"] = round(float(score), 4)
                    result.append(entry)
                return result, RERANK_CROSS_ENCODER
            except Exception as ex:
                logger.warning(f"Cross-Encoder 精排推理失败，降级为词重叠精排: {ex}")

        return self._lexical_rerank(query, items, top_k), RERANK_LEXICAL_FALLBACK

    def _lexical_rerank(self, query: str, items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """降级策略：基于 jieba 分词的词重叠比例打分，不依赖任何外部模型。"""
        query_tokens = set(self._tokenize(query))
        scored = []
        for item in items:
            content_tokens = set(self._tokenize(item.get("content", "")))
            overlap = len(query_tokens & content_tokens)
            denom = len(query_tokens) or 1
            score = overlap / denom
            scored.append((item, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        result = []
        for item, score in scored[:top_k]:
            entry = dict(item)
            entry["rerank_score"] = round(score, 4)
            result.append(entry)
        return result

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        if _JIEBA_AVAILABLE:
            return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
        return list(text)
