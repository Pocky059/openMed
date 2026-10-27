"""
RAG 医疗知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding），同步维护 BM25 词频索引
  2. 混合检索：BM25 关键词召回 + 向量语义召回 → RRF 融合 → Top-K 截断
     解决单路检索的两个典型问题：
       - 纯向量检索对药品名、专有名词等短词的召回不稳定
       - 纯关键词检索无法理解语义相近但字面不同的问法
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler，
     precise 排序交给 core/reranker.py 的 Cross-Encoder 精排完成

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 患者画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# 文件化医院知识库目录（药品手册/科室指南/症状手册/挂号流程/系统说明）。
# 用 __file__ 锚定项目根，Docker（工作目录 /app）与本地运行都能定位到同一路径。
_HOSPITAL_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "hospital"

# 中文 embedding 工厂已抽到 core/embedding.py（RAG 与三级记忆共用）：
# ChromaDB 默认的 all-MiniLM-L6-v2 是英文模型（384 维），对中文近乎随机投影，
# 换用 bge-small-zh-v1.5（512 维）后向量召回才有效；collection 的 embedding
# 配置创建时固化，重建 collection 的脚本也必须传同一个函数。
from core.embedding import make_embedding_function


def _load_hospital_docs() -> List[Dict[str, str]]:
    """加载 data/knowledge/hospital/ 下的文件化医院知识库。

    目录里每个 *.json 文件是一个文档数组，元素需含 title 与 content 两个字段。
    单个文件解析失败只跳过该文件（记 warning），不拖垮整体导入——这保证了
    以后扩充语料时，一个坏文件不会让整个 RAG 默认语料加载失败。
    返回空列表表示目录不存在或没有任何可用文档，由调用方决定是否回退内置文档。
    """
    docs: List[Dict[str, str]] = []
    if not _HOSPITAL_DIR.is_dir():
        return docs
    for path in sorted(_HOSPITAL_DIR.glob("*.json")):
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            for item in items:
                if isinstance(item, dict) and item.get("title") and item.get("content"):
                    docs.append({"title": item["title"], "content": item["content"]})
                else:
                    logger.warning(f"hospital 知识库中缺少 title/content 的条目被跳过: {path.name}")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"hospital 知识库文件解析失败，跳过: {path.name} ({exc})")
    return docs


def _tokenize(text: str) -> List[str]:
    """中文用 jieba 分词，英文/数字用正则兜底；jieba 不可用时退化为字符级切分。"""
    text = (text or "").lower()
    if _JIEBA_AVAILABLE:
        return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    tokens = _TOKEN_RE.findall(text)
    tokens += [ch for ch in text if "一" <= ch <= "鿿"]
    return tokens


# ── 查询别名展开（P2，2026-10-15）──────────────────────────────────────────────
# 用户叫法与语料写法不一致会导致「文档在库里却搜不到」：BM25 是字面匹配，
# 异字别名词分数直接为 0；向量模型对「长得完全不像但同一个东西」的别名
# 也常常拉不近（基线归因 + SFT 扩充先验检查都实测到这类漏检，如贯叶连翘→圣约翰草）。
#
# 解法：查询进检索前，若命中别名词，把语料规范名追加进查询串。保留原词
# （原词对向量路仍有价值），追加词让 BM25 能字面命中规范名所在文档。
# 映射放在 data/knowledge/aliases.json（数据文件，加别名不动代码），
# 每条必须医药上同物、零歧义。查询改写（口语化表达→规范医学术语）不在这里做，
# 那是 Agent 的职责（2.1 的 SFT 数据正在强化它）——这里只兜底封闭别名集合。

_ALIASES_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "aliases.json"
_aliases_cache: Optional[List[Dict[str, List[str]]]] = None


def _load_aliases() -> List[Dict[str, List[str]]]:
    """惰性加载别名映射；文件缺失/损坏时返回空表（检索链路不受影响）。"""
    global _aliases_cache
    if _aliases_cache is not None:
        return _aliases_cache
    try:
        raw = json.loads(_ALIASES_PATH.read_text(encoding="utf-8"))
        _aliases_cache = [
            {"from": entry["from"], "to": entry["to"]}
            for entry in raw["aliases"]
            if entry.get("from") and entry.get("to")
        ]
    except (OSError, json.JSONDecodeError, KeyError):
        logger.warning(f"别名表不可用，检索不做展开: {_ALIASES_PATH}")
        _aliases_cache = []
    return _aliases_cache


def _expand_aliases(query: str) -> str:
    """查询里出现任一 from 词 → 追加该条 to 词（多组命中全部追加，去重）。"""
    added: List[str] = []
    for entry in _load_aliases():
        if any(word in query for word in entry["from"]):
            added.extend(entry["to"])
    seen = set()
    unique = [w for w in added if not (w in seen or seen.add(w))]
    return f"{query} {' '.join(unique)}".strip() if unique else query


class KnowledgeBase:
    """
    基于 ChromaDB + BM25 的医疗 RAG 知识库。

    Embedding 统一走 core/embedding.py 的中文模型 BAAI/bge-small-zh-v1.5
    （collection 创建时必须传 embedding_function，见 __init__；默认英文模型
    all-MiniLM-L6-v2 对中文近乎随机投影，曾导致金文档掉出候选，勿回退）。
    BM25 索引在内存中基于 Chroma 已有文档懒构建，随文档增删失效重建。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            # 注意：0.5.23 的 Settings 没有 HTTP 超时字段（底层 httpx 是 timeout=None 无限等待），
            # 不要往里加 chroma_server_http_timeout_ms——那是 0.4.x 字段，会触发
            # pydantic ValidationError，被本 except 吞掉后静默降级为本地模式（踩过坑）
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 统一传中文 embedding 函数（不再用 ChromaDB 默认的英文模型）：
        # - 服务端模式：EF 只把模型配置序列化发给服务端，由服务端加载 bge-small-zh，
        #   客户端本地不下载、不跑模型（实测 add 无本地开销）
        # - 本地模式：EF 在本地惰性加载模型（首次 add/query 时下载到 HF 缓存目录）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "OpenMed 医疗 RAG 知识库"},
            embedding_function=make_embedding_function(),
        )

        # BM25 索引懒构建缓存
        self._bm25_index: Optional[Any] = None
        self._bm25_corpus_ids: List[str] = []
        self._bm25_dirty = True

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "...", ...}, ...]
        长文档会自动切片（每片 500 字）。支持 md/txt/json 来源在上层预处理成该结构后统一导入。

        除 title/content 外，可选字段（entities/doc_id/doc_type/source/
        severity/evidence_level/direction）会原样写入每个 chunk 的 metadata——
        评测时靠它验证「模型有没有搜齐实体、搜对文档类型」；这些字段只进 metadata
        不进正文，不污染检索内容。可选字段缺省时行为与旧版完全一致（只写
        title/chunk_index/total_chunks），对既有调用方向后兼容。

        注意：rule_id 已从白名单移除（2026-10-10）——语料与规则库的对应关系
        痕迹在数据层已抹除（正文编号 + JSONL 字段都删了），metadata 也不该再
        保留这条显式关联；判分直接对照规则库答案册，不需要语料侧编号。
        """
        ids, docs, metas = [], [], []
        # 允许透传进 metadata 的字段白名单（白名单防止脏字段混入；
        # rule_id 有意缺席——语料与规则库应是「同一上游、独立提炼」，
        # 不保留「本段对应规则库哪一条」的显式痕迹）
        passthrough_fields = (
            "entities", "doc_id", "doc_type", "source",
            "severity", "evidence_level", "direction",
        )

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                meta = {"title": title, "chunk_index": i, "total_chunks": len(chunks)}
                # 可选字段逐个透传；空值（None/空串/空列表）不写入，避免 ChromaDB 报错。
                # 注意：本版 ChromaDB 不允许 list 作为 metadata 值，entities 这类
                # list 字段需序列化为 JSON 字符串存入（评测侧读回时 json.loads 还原）
                for field in passthrough_fields:
                    value = doc.get(field)
                    if value in (None, "", []):
                        continue
                    meta[field] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                metas.append(meta)

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            self._bm25_dirty = True
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, Any]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        纯向量语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        return self._vector_search_raw(query, top_k)

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    def hybrid_search(self, query: str, top_k: int = 5, recall_pool: int = 30) -> List[Dict[str, Any]]:
        """
        混合检索：BM25 词频召回 + 向量语义召回 → RRF 融合 → Top-K 截断。

        recall_pool 控制每一路召回的候选池大小，融合后再截断到 top_k，
        为后续 Cross-Encoder 精排留出足够的候选空间。

        入口先做别名展开（_expand_aliases）：生产 Agent、评测直测、
        SFT 回填等所有走本方法的链路统一吃到，行为确定性可复现。
        """
        query = _expand_aliases(query)
        vector_hits = self._vector_search_raw(query, recall_pool)
        bm25_hits = self._bm25_search_raw(query, recall_pool)
        fused = self._reciprocal_rank_fusion([vector_hits, bm25_hits])
        return fused[:top_k]

    async def hybrid_search_async(self, query: str, top_k: int = 5, recall_pool: int = 30) -> List[Dict[str, Any]]:
        """异步混合检索；底层为同步实现，放入线程池执行。"""
        return await asyncio.to_thread(self.hybrid_search, query, top_k, recall_pool)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    def list_documents(self) -> Dict[str, Any]:
        """按文档标题聚合库内片段，供知识库管理页展示（前端 /knowledge/docs）。

        内置/用户判断：标题命中 hospital 目录或兜底内置文档 => builtin，
        其余是运行时通过 /knowledge/add、/knowledge/upload 导入的 user 文档。
        归类依据是「标题」而非入库入口——历史数据（旧版入库、无 source
        标记）也能正确归类，不用迁移存量 ChromaDB。
        """
        data = self._collection.get(include=["metadatas"])
        titles: Dict[str, int] = {}
        for meta in data.get("metadatas", []):
            title = (meta or {}).get("title", "")
            if title:
                titles[title] = titles.get(title, 0) + 1

        builtin_titles = {doc["title"] for doc in _load_hospital_docs()}
        builtin_titles |= {doc["title"] for doc in _FALLBACK_DOCS}
        docs = [
            {"title": title, "chunks": count, "source": "builtin" if title in builtin_titles else "user"}
            for title, count in sorted(titles.items())
        ]
        builtin_count = sum(1 for d in docs if d["source"] == "builtin")
        return {
            "docs": docs,
            "total_chunks": sum(titles.values()),
            "builtin_count": builtin_count,
            "user_count": len(docs) - builtin_count,
        }

    async def list_documents_async(self) -> Dict[str, Any]:
        """异步获取文档列表（同步 ChromaDB 客户端放入线程池）。"""
        return await asyncio.to_thread(self.list_documents)

    def get_document(self, title: str) -> Dict[str, Any]:
        """按标题取回一篇文档的全部片段，供前端只读阅览（/knowledge/doc/{title}）。

        片段按 chunk_index 升序还原全文顺序；找不到该标题返回空 chunks。
        """
        data = self._collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            if (meta or {}).get("title") != title:
                continue
            chunks.append({
                "index": (meta or {}).get("chunk_index", 0),
                "content": doc,
            })
        chunks.sort(key=lambda c: c["index"])
        return {"title": title, "chunks": chunks}

    async def get_document_async(self, title: str) -> Dict[str, Any]:
        """异步获取单篇文档片段（同步 ChromaDB 客户端放入线程池）。"""
        return await asyncio.to_thread(self.get_document, title)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))

        这里直接返回混合检索（BM25+向量+RRF）的召回池，精排交给
        MCPToolManager.search_with_rerank() 中的 CrossEncoderReranker 完成。
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        recall_pool = max(int(top_k) * 4, 20)
        return await self.hybrid_search_async(query, top_k=top_k, recall_pool=recall_pool)

    # ── 向量召回 ──────────────────────────────────────────────────────────────

    def _vector_search_raw(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items: List[Dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "id":       doc_id,
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    # ── BM25 召回 ─────────────────────────────────────────────────────────────

    def _ensure_bm25_index(self) -> None:
        """懒构建/重建 BM25 索引，基于 Chroma 集合中的全部文档。"""
        if not _BM25_AVAILABLE:
            return
        if not self._bm25_dirty and self._bm25_index is not None:
            return

        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        self._bm25_docs = docs
        self._bm25_metas = metas
        self._bm25_corpus_ids = ids

        if docs:
            tokenized = [_tokenize(doc) for doc in docs]
            self._bm25_index = BM25Okapi(tokenized)
        else:
            self._bm25_index = None
        self._bm25_dirty = False

    def _bm25_search_raw(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not _BM25_AVAILABLE:
            return []
        self._ensure_bm25_index()
        if self._bm25_index is None or not self._bm25_corpus_ids:
            return []

        scores = self._bm25_index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        items: List[Dict[str, Any]] = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metas[idx] or {}
            items.append({
                "id":      self._bm25_corpus_ids[idx],
                "title":   meta.get("title", ""),
                "content": self._bm25_docs[idx],
                "score":   round(float(scores[idx]), 4),
                "chunk":   meta.get("chunk_index", 0),
            })
        return items

    # ── RRF 融合 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion：score(d) = Σ 1 / (k + rank_i(d))

        不依赖各路召回分数的量纲（BM25 分数和向量相似度不可直接比较），
        只用排名信息融合，是多路召回融合的稳健做法。
        """
        fused_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                doc_id = item.get("id") or f"{item.get('title')}_{item.get('chunk')}"
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                if doc_id not in doc_lookup:
                    doc_lookup[doc_id] = item

        ordered_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)
        fused: List[Dict[str, Any]] = []
        for doc_id in ordered_ids:
            entry = dict(doc_lookup[doc_id])
            entry["rrf_score"] = round(fused_scores[doc_id], 5)
            fused.append(entry)
        return fused

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认医疗知识库文档（药品手册、科室指南、症状手册、挂号流程、系统说明）。

        优先从 data/knowledge/hospital/ 目录加载文件化知识库——语料与代码分离，
        后续扩充医院场景语料（如更多药品手册）只需加 JSON 文件，不用改代码。
        目录缺失或文件全部不可用时，回退到下方内置文档，保证最小化部署下
        RAG 仍有兜底语料可用。
        """
        docs = _load_hospital_docs()
        if docs:
            self.add_documents(docs)
            logger.info(f"已导入默认医疗知识库（hospital 目录）: {len(docs)} 篇文档")
            return

        logger.warning("hospital 目录无可用文档，回退到内置默认知识库")
        self.add_documents(_FALLBACK_DOCS)
        logger.info(f"已导入默认医疗知识库（内置兜底）: {len(_FALLBACK_DOCS)} 篇文档")


# hospital 目录不可用时的兜底内置文档（最小化部署保证 RAG 仍有语料）。
# 提成模块级常量：list_documents 判断「内置文档」时也要用这份标题集合。
_FALLBACK_DOCS: List[Dict[str, str]] = [
    {
        "title": "布洛芬缓释胶囊说明书摘要",
        "content": (
            "布洛芬缓释胶囊说明书摘要。"
            "适应症：用于缓解轻至中度疼痛，如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经，也用于普通感冒或流感引起的发热。"
            "用法用量：口服，成人一次 1 粒（0.3g），早晚各一次，或遵医嘱，不超过说明书标注的每日上限。"
            "禁忌：对本品过敏者禁用；活动性消化性溃疡患者禁用；严重肝肾功能不全者禁用；妊娠晚期妇女禁用。"
            "不良反应：偶见恶心、腹痛、腹泻等胃肠道反应，长期使用需注意胃肠道出血风险。"
            "药物相互作用：与其他非甾体抗炎药（如阿司匹林）同服会增加不良反应风险，不建议联用；与抗凝血药同用可能增加出血倾向。"
            "特殊人群：孕妇、哺乳期妇女、儿童用药请咨询医师或药师。"
        ),
    },
    {
        "title": "对乙酰氨基酚片说明书摘要",
        "content": (
            "对乙酰氨基酚片说明书摘要。"
            "适应症：用于普通感冒或流行性感冒引起的发热，也用于轻至中度疼痛如头痛、关节痛、牙痛。"
            "用法用量：口服，成人一次 0.5g，一日 3-4 次，两次用药间隔不少于 4 小时，24 小时内不超过 4 次。"
            "禁忌：对本品过敏者禁用；严重肝肾功能不全者禁用；不能与其他含对乙酰氨基酚成分的复方感冒药同服，避免过量。"
            "不良反应：偶见皮疹、恶心；过量或长期大量使用可能引起肝损害。"
            "药物相互作用：与酒精同服会增加肝脏毒性风险；与华法林等抗凝药同用需谨慎。"
            "特殊人群：儿童用药需按体重换算剂量，请遵医嘱。"
        ),
    },
    {
        "title": "阿莫西林胶囊说明书摘要",
        "content": (
            "阿莫西林胶囊说明书摘要。"
            "适应症：用于敏感菌所致的呼吸道感染、尿路感染、皮肤软组织感染等细菌感染性疾病。"
            "用法用量：口服，成人一次 0.5g，一日 3 次，需遵医嘱按疗程服用，不可自行停药或减量。"
            "禁忌：对青霉素类抗生素过敏者禁用；传染性单核细胞增多症患者不宜使用。"
            "不良反应：可能出现皮疹、恶心、腹泻等；如出现呼吸困难、全身皮疹等过敏反应需立即停药就医。"
            "药物相互作用：与丙磺舒同用会升高血药浓度；服药期间避免饮酒。"
            "特殊人群：孕妇、哺乳期妇女使用前请咨询医生。"
        ),
    },
    {
        "title": "科室介绍：内科与外科",
        "content": (
            "内科主要诊治发热、咳嗽、乏力、消化不良、慢性病随访等不需要手术处理的内部器官疾病，"
            "细分为呼吸内科、消化内科、心血管内科、神经内科、内分泌科等。"
            "外科主要处理需要手术或创伤处理的疾病，如骨折、阑尾炎、肿瘤切除等，"
            "细分为普通外科、骨科、泌尿外科、神经外科等。"
            "如果不确定该挂哪个科，可以先挂内科或全科门诊，由接诊医生判断是否需要转诊至专科。"
        ),
    },
    {
        "title": "科室介绍：儿科、妇科与急诊科",
        "content": (
            "儿科负责 0-14 岁儿童的常见病、多发病诊治，如发热、咳嗽、腹泻、疫苗接种咨询等。"
            "妇科负责女性生殖系统相关疾病诊治，如月经不调、妇科炎症、孕前检查等；孕期产检请挂产科。"
            "急诊科负责处理危及生命或需要紧急处理的情况，如胸痛、呼吸困难、意识不清、大出血、严重外伤等，"
            "急诊科 24 小时开放，出现红旗症状应直接前往急诊科或拨打急救电话，不建议等待普通门诊排队。"
        ),
    },
    {
        "title": "常见病问答：感冒与发热",
        "content": (
            "Q: 感冒发烧要吃抗生素吗？A: 普通感冒多为病毒感染，抗生素对病毒无效，一般以对症退热、多饮水、休息为主，"
            "如果持续高烧超过 3 天或出现脓痰等细菌感染迹象，需就医评估是否需要抗生素。"
            "Q: 体温多少算发热？A: 一般以腋温超过 37.3℃ 视为发热，38.5℃ 以上建议服用退烧药并观察。"
            "Q: 孩子发烧能不能物理降温？A: 可以用温水擦拭四肢辅助降温，避免使用酒精擦浴，若体温持续不降或出现抽搐应立即就医。"
            "Q: 咳嗽超过两周需要注意什么？A: 持续咳嗽超过两周建议就诊排查支气管炎、肺炎或其他慢性呼吸道疾病。"
        ),
    },
    {
        "title": "常见病问答：肠胃不适与皮肤过敏",
        "content": (
            "Q: 肚子疼一定是肠胃炎吗？A: 不一定，腹痛原因很多，包括肠胃炎、阑尾炎、胆囊炎等，"
            "如果伴随发热、持续加重或右下腹剧痛，需及时就医排查急腹症。"
            "Q: 皮肤起红疹要不要紧？A: 多数为过敏或接触性皮炎，可先停用可疑致敏物并观察，"
            "如果红疹快速扩散、伴随呼吸困难或面部肿胀，属于严重过敏反应，需立即就医或拨打急救电话。"
            "Q: 腹泻期间可以正常饮食吗？A: 建议清淡饮食、少食多餐，避免油腻和生冷食物，注意补充水分和电解质。"
        ),
    },
    {
        "title": "预约挂号与就诊流程说明",
        "content": (
            "预约挂号流程说明。"
            "用户可以通过线上小程序或人工窗口预约挂号，选择科室、医生和就诊时间段。"
            "号源分为普通号和专家号，专家号需提前 3-7 天预约，节假日号源较为紧张。"
            "预约成功后会生成预约单号，就诊当日请提前 15-30 分钟到院取号。"
            "如需改期或取消，请在就诊前一天 18:00 前操作，临时取消可能影响信用记录。"
            "挂号费根据科室和医生职称不同而不同，普通号通常在 10-50 元，专家号 50-300 元不等，具体以医院公示为准。"
            "如遇重复扣费或支付成功但未生成预约的情况，请保留支付凭证并联系人工核实，系统通常会在 1-3 个工作日内处理退费。"
        ),
    },
]