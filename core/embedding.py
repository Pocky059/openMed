"""共享的中文 embedding 工厂 —— 所有 ChromaDB collection 统一从这里取。

为什么要有这个文件：
  ChromaDB 默认的 all-MiniLM-L6-v2 是英文模型（384 维），对中文文本的向量
  投影近乎随机，等于把向量召回变成纯噪声（RAG 曾因此让金文档掉出候选，
  三级记忆的 episodic/user_profile 也踩过同一个坑）。
  换成 BAAI/bge-small-zh-v1.5（512 维，约 100MB），与精排模型
  bge-reranker-base 同属 BGE 家族。

统一入口的意义：
  collection 的 embedding 配置在创建时固化（维度写死在库里），换模型必须
  删库重建。RAG 知识库（mcp/knowledge_base.py）、三级记忆（memory/
  conversation_memory.py）、以及任何重建 collection 的脚本（scripts/）都必须
  用同一个函数——只要有一处漏传，重建出来的 collection 又会落回英文默认模型。

服务端模式下（docker compose 的 chromadb 服务）模型由服务端加载（HF 缓存
挂载在 ./data/huggingface），客户端零模型开销；本地嵌入式模式才在客户端
惰性加载（首次运行需下载约 100MB 模型）。
"""
import logging
from typing import Any, Optional

from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 本地模型惰性单例：整个进程只加载一次，供需要"直接算向量"的模块复用
# （意图识别语义路），与 ChromaDB collection 的 embedding 函数是两个用途
_local_model: Any = None
_local_model_failed = False


def make_embedding_function():
    """构造中文 embedding 函数。所有 collection 创建/重建时都必须传它。"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_local_embedder() -> Optional[Any]:
    """惰性加载并返回本地中文 embedding 模型（SentenceTransformer）。

    首次调用时从磁盘缓存加载（容器里模型缓存在挂载卷 ./data/huggingface，
    约 100MB）；加载失败返回 None 且不再重试——调用方自行决定降级策略
    （例如意图识别退字符 n-gram 哈希）。

    注意返回的是原始 SentenceTransformer 模型（encode 文本用），不是
    ChromaDB 的 EmbeddingFunction，两者别混用。
    """
    global _local_model, _local_model_failed
    if _local_model is not None:
        return _local_model
    if _local_model_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"本地中文 embedding 模型已加载: {EMBEDDING_MODEL}")
        return _local_model
    except Exception as ex:
        _local_model_failed = True
        logger.warning(f"本地中文 embedding 模型加载失败（后续走调用方兜底）: {ex}")
        return None
