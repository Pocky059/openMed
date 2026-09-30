"""
Cross-Encoder 精排器。

检索优化链路的最后一环：BM25 + 向量混合召回 → RRF 融合 → Top-K 截断 → 这里的 Cross-Encoder 精排。

Bi-Encoder（ChromaDB 内置的向量检索）把 query 和文档分别编码成向量再比余弦相似度，速度快但精度有限；
Cross-Encoder 把 query 和文档拼在一起送入模型联合编码，能建模两者的细粒度交互，精度更高但更慢，
因此只在混合召回已经把候选集缩小到几十条之后使用，兼顾效果和延迟。

模型不可用（未安装 sentence-transformers 或模型下载失败）时，
优雅降级为基于 jieba 分词的词重叠打分，保证检索链路不因为精排模块故障而中断。
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """
    Cross-Encoder 精排器，懒加载模型，加载或推理失败时自动降级为词重叠打分。
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model: Any = None
        self._load_failed = False

    def _ensure_model(self) -> bool:
        """懒加载 Cross-Encoder 模型；只尝试加载一次，失败后不再重复尝试。"""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info(f"Cross-Encoder 精排模型已加载: {self._model_name}")
            return True
        except Exception as ex:
            logger.warning(f"Cross-Encoder 模型加载失败，降级为词重叠精排: {ex}")
            self._load_failed = True
            return False

    def rerank(self, query: str, items: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对召回候选集重新打分排序，取 Top-K。

        items 中每个元素需含 "content" 字段（用于和 query 联合编码）。
        """
        if not items:
            return []
        if len(items) <= top_k and len(items) <= 1:
            return items[:top_k]

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
                return result
            except Exception as ex:
                logger.warning(f"Cross-Encoder 精排推理失败，降级为词重叠精排: {ex}")

        return self._lexical_rerank(query, items, top_k)

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
