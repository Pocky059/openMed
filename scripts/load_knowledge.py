"""知识库入库脚本：多跳语料 + 医院场景语料 → ChromaDB

用途：把审核通过的 217 篇多跳语料（multihop_corpus.jsonl）和 12 篇医院场景
文档（data/knowledge/hospital/*.json）批量导入 ChromaDB 知识库集合。

设计要点：
  1. 清空重建（幂等）：先删掉 collection 再重建，重复执行结果一致，
     不会像增量导入那样越跑越多
  2. 多跳语料的 entities/doc_type/rule_id 等字段经 add_documents 的白名单
     透传进每个 chunk 的 metadata（评测时验证「搜没搜齐实体」）
  3. 连容器 ChromaDB 服务（localhost:8001）；embedding 统一用中文模型
     bge-small-zh-v1.5：服务端模式下由容器按客户端传来的配置加载模型，
     宿主机无需下载；服务不可用时退化到本地持久化模式（./data/chroma）

注意：本脚本直接操作 ChromaDB，不经过容器内 app。跑完后必须重启 app 容器
（docker compose restart app）——app 进程内的 BM25 词频索引是懒构建缓存的，
不重启会继续使用旧的索引内容，与新库对不上。

用法：python scripts/load_knowledge.py
"""

import json
import sys
from pathlib import Path

# 让脚本能 import 项目根下的 core/mcp 包（standalone 运行时不在默认 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp"))

from core.embedding import make_embedding_function  # noqa: E402
from knowledge_base import (  # noqa: E402
    KnowledgeBase,
    _load_hospital_docs,
)

CORPUS_PATH = PROJECT_ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"


def load_multihop_corpus() -> list:
    """读合并产物：每行一篇文档，字段与语料 JSONL 一致（含 entities 等）。"""
    docs = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


def load_env() -> dict:
    """读 .env 的 CHROMA_HOST/CHROMA_PORT（本地开发约定：宿主机 8001 映射容器 8000）。

    只做最小解析（KEY=VALUE，跳过注释行），不引入 python-dotenv 依赖。
    """
    env = {"CHROMA_HOST": "localhost", "CHROMA_PORT": "8001"}
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip() in env:
                    env[key.strip()] = value.strip()
    return env


def main():
    # 1. 连接（空库时 __init__ 会自动导入 hospital 默认文档，稍后清空重建，无影响）
    env = load_env()
    kb = KnowledgeBase(chroma_host=env["CHROMA_HOST"], chroma_port=int(env["CHROMA_PORT"]))

    # 2. 清空重建 collection（幂等的关键一步）。
    #    必须传中文 embedding 函数：collection 的 embedding 配置创建时固化，
    #    漏传会落回英文默认模型（换 embedding 必须删库重建，这里正好每次都删）
    kb._client.delete_collection(KnowledgeBase.COLLECTION_NAME)
    kb._collection = kb._client.get_or_create_collection(
        name=KnowledgeBase.COLLECTION_NAME,
        metadata={"description": "OpenMed 医疗 RAG 知识库"},
        embedding_function=make_embedding_function(),
    )
    print(f"collection 已清空重建（模式: {'服务端' if kb._use_server else '本地'}）")

    # 3. 分批导入多跳语料（217 篇，带 entities/rule_id 等 metadata）。
    #    分批是为了：① 单次请求太大容易触发 HTTP 超时（embedding 在服务端逐片段算，
    #    CPU 容器较慢）② 每批打进度，出问题知道卡在哪批
    corpus_docs = load_multihop_corpus()
    n_corpus_chunks = 0
    batch_size = 40
    for start in range(0, len(corpus_docs), batch_size):
        batch = corpus_docs[start:start + batch_size]
        n_corpus_chunks += kb.add_documents(batch)
        print(f"  多跳语料 {min(start + batch_size, len(corpus_docs))}/{len(corpus_docs)} 篇已导入")

    # 4. 导入医院场景语料（12 篇，无 entities，纯 title/content）
    hospital_docs = _load_hospital_docs()
    n_hospital_chunks = kb.add_documents(hospital_docs)

    # 5. 验证
    total = kb._collection.count()
    print(f"多跳语料 {len(corpus_docs)} 篇 -> {n_corpus_chunks} 个片段")
    print(f"医院语料 {len(hospital_docs)} 篇 -> {n_hospital_chunks} 个片段")
    print(f"collection 总片段数: {total}")

    # 抽查：确认 entities 确实进了 metadata
    sample = kb._collection.get(limit=10, include=["metadatas"])
    with_entities = [m for m in sample["metadatas"] if m.get("entities")]
    print(f"抽查前 10 个片段，{len(with_entities)} 个带 entities metadata（期望 >0）")

    if total == 0 or not with_entities:
        print("入库校验失败！")
        sys.exit(1)
    print("入库完成。下一步：docker compose restart app，再做 /chat 冒烟验收")


if __name__ == "__main__":
    main()
