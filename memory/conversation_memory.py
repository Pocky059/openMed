"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆按固定顺序拼接（to_prompt_text）：会话摘要 →
    相关历史（ChromaDB 相似度序，取前 3）→ 用户画像 → 最近消息（时间序，取后 8）
  - 工作记忆达到 COMPRESS_AT 条数时自动压缩（LLM 摘要），防止 context 爆炸
  - 情景记忆/用户画像的 Embedding 由 collection 配置的中文模型
    （bge-small-zh-v1.5，core/embedding.py）生成：服务端模式下模型加载在
    ChromaDB 服务侧（HF 缓存挂载卷），客户端零模型开销
"""
import hashlib
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis.asyncio as redis
from core.embedding import make_embedding_function
from core.llm_utils import create_llm_client, extract_text_content

logger = logging.getLogger(__name__)


# ── 四字段结构化摘要（2026-10-12 用户拍板落地）────────────────────────────────
# 压缩摘要从「2-3 句话自由文本」改为固定四字段 JSON：自由文本下 LLM 会自行
# 「精简」，过敏史/用药史等安全关键信息被省略——长会话评测要测的正是病史丢失。
# 字段与长会话考卷 gold_items.type 一一对应（allergy→allergies /
# medication→medications / symptom→symptoms / history→history），判分时
# 直接读 summary key 的 JSON 逐字段查关键节点是否还在。
SUMMARY_FIELDS = ("allergies", "medications", "symptoms", "history")

# 压缩摘要生成 prompt（四字段铁律）
_SUMMARY_PROMPT_TEMPLATE = """把以下对话中已出现的医疗信息逐条提炼成 JSON，字段固定四个：
- allergies: 过敏史（药物/食物过敏及过敏表现）
- medications: 用药史（正在服用的药及剂量；已停用的药注明已停）
- symptoms: 症状（当前症状、持续时间、诱因）
- history: 就诊史（既往疾病、手术、检查结果、家族史）

铁律：
1. 对话中已出现的每一条医疗信息都必须逐条保留——病史是安全关键信息，宁可多列、不可遗漏；
2. 没提到的字段返回空数组；
3. 只输出 JSON，不要任何其他文字。

对话:
{text}"""


def parse_summary(text: str) -> Optional[Dict[str, List[str]]]:
    """把摘要文本解析成四字段 dict；不是合法四字段 JSON 时返回 None。

    返回 None 的两种用途：旧版自由文本摘要的识别（合并时回退 LLM 路径），
    以及合并纯函数的参数校验。字段必须齐全且都是字符串列表，缺字段/类型
    不符都算非法（宁可走回退路径，不吞半个 JSON）。
    """
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    parsed: Dict[str, List[str]] = {}
    for field in SUMMARY_FIELDS:
        value = data.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return None
        parsed[field] = value
    return parsed


def merge_field_summaries(old_text: str, new_text: str) -> Optional[str]:
    """确定性合并两段四字段摘要：按字段拼接去重（保序，旧段在前）。

    任一段不是合法四字段 JSON 时返回 None——调用方回退 LLM 合并（兼容
    旧版自由文本摘要）。用确定性合并而非 LLM 合并的理由：合并是纯拼装
    不是语义加工，LLM 合并会二次「精简」导致病史丢失，且白花一次调用。
    """
    old = parse_summary(old_text)
    new = parse_summary(new_text)
    if old is None or new is None:
        return None
    merged: Dict[str, List[str]] = {}
    for field in SUMMARY_FIELDS:
        items: List[str] = []
        seen = set()
        for item in old[field] + new[field]:
            if item not in seen:
                seen.add(item)
                items.append(item)
        merged[field] = items
    return json.dumps(merged, ensure_ascii=False)


def truncate_summary(text: str, max_chars: int) -> str:
    """按安全优先级截断四字段摘要（超长时丢低优先级字段的最新条目）。

    字段优先级 = SUMMARY_FIELDS 顺序：allergies > medications > symptoms >
    history——过敏史/用药史是安全关键字段，最后牺牲；history 的最新条目最先丢。
    字段内条目旧在前（合并保序），从尾部（最新）开始丢。

    为什么要按优先级而不是硬截：之前的 merged[:max_chars] 从头保留，把
    history 尾部的最新病史连同其他字段尾部一起切掉，且不区分字段——安全
    关键字段可能被 history 挤出。丢的时候也让最低优先级的字段先承担损失。

    非四字段 JSON（旧版自由文本）不处理——调用方走自己的截断逻辑。
    """
    parsed = parse_summary(text)
    if parsed is None:
        return text[:max_chars]

    def dump() -> str:
        return json.dumps(parsed, ensure_ascii=False)

    # 从最低优先级字段开始丢尾部条目，每丢一条重新量长度
    for field in reversed(SUMMARY_FIELDS):
        while len(dump()) > max_chars and parsed[field]:
            parsed[field].pop()
    # 兜底：条目丢光仍超长（理论上单条不可能超长）时硬截，保证 ≤ max_chars
    return dump()[:max_chars]


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            # ensure_ascii=False：中文画像被转义成 \uXXXX 时人和模型都读不动，保留原文
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=False)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages[-8:]:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 工作记忆读取上限（_get_working_memory 取最近 N 条）
    COMPRESS_AT   = 15    # 达到此条数时触发压缩（add_message 检查），保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数
    # 四字段 JSON 摘要的长度上限（2026-10-12 从 800 提到 1600：800 截断 JSON 会破坏
    # 可解析性，病史多的长会话需要更多空间——长会话考卷每场 3-5 项病史实测够用）。
    # 超长截断按安全优先级丢低优先级字段的最新条目（见 truncate_summary，
    # 2026-10-26：原硬截 merged[:MAX] 不分字段，安全关键字段可能被挤掉）
    SUMMARY_MAX_CHARS = 1600
    PROFILE_DOC_PREFIX = "user_profile:"

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
    ):
        #双协议客户端（v2 阶段 0.5）：按 OPENMED_LLM_PROTOCOL 切换 Anthropic/OpenAI 协议
        self._client = create_llm_client(api_key=api_key, base_url=base_url)
        self._model  = model

        self._redis = redis.from_url(redis_url, decode_responses=True)

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        # 必须传中文 embedding 函数（core/embedding.py）：collection 的 embedding
        # 配置在创建时固化，漏传会落回 ChromaDB 英文默认模型——中文查询的向量
        # 召回近乎随机投影，情景记忆静默失效（2026-10-07 修复，与 RAG 同病）。
        self._episodic = chroma.get_or_create_collection(
            "episodic", embedding_function=make_embedding_function()
        )
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection(
            "user_profile", embedding_function=make_embedding_function()
        )

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）+ 压缩阈值检查。
        # Redis 不可用时降级跳过（2026-10-26 补齐）：写不进去就记日志返回，
        # 不阻塞主链路——在线可用性优先于记忆完整性。ChromaDB 两侧早就有
        # try/except，Redis 侧一直裸奔：Redis 挂了 /chat 直接 500。
        try:
            await self._redis.lpush(key, json.dumps({
                "role":      msg.role.value,
                "content":   msg.content,
                "ts":        msg.timestamp.isoformat(),
                "metadata":  msg.metadata,
            }))
            await self._redis.expire(key, 86400)  # 24h TTL

            # 超过压缩阈值时触发压缩
            if await self._redis.llen(key) >= self.COMPRESS_AT:
                await self._compress(user_id, conv_id)
        except Exception as ex:
            logger.warning(f"工作记忆写入失败（Redis 不可用，本轮记忆降级）: {ex}")

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼用户偏好，更新用户画像。
        用 LLM 提炼偏好，然后存入 ChromaDB（ChromaDB 内置 embedding，不依赖外部 API）。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        current_profile = await self._get_profile(user_id)

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        profile_ctx = json.dumps(current_profile, ensure_ascii=False) if current_profile else "{}"
        prompt = f"""从以下对话和已有用户画像中提炼或更新用户偏好和关键实体，返回 JSON。
对话:
{text}

已有画像:
{profile_ctx}

返回格式: {{"preferences": ["..."], "entities": {{"产品": [], "问题类型": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                # thinking 关掉（extra_body 透传）：2026-10-13 pilot 实测长对话下
                # 思考块吃光 4096 输出额度、text 只剩半截 JSON（画像整轮跳过更新）；
                # 提炼是结构化输出任务，关思考后输出完整且只花几十 token
                model=self._model, max_tokens=4096, temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (extract_text_content(resp.content) or "").strip()
            # LLM 可能返回空内容或无花括号的文本（如限流/异常响应），此时跳过本次更新，
            # 而不是让 json.loads("") 抛 "Expecting value" 异常
            s, e = raw.find("{"), raw.rfind("}")
            if not raw or s < 0 or e <= s:
                logger.warning(f"画像提炼返回内容不含 JSON，跳过本次更新: {raw[:80]!r}")
                return
            profile_data = json.loads(raw[s:e + 1])
            if not isinstance(profile_data, dict):
                logger.warning(f"画像提炼返回内容不是 JSON 对象，跳过本次更新: {raw[:80]!r}")
                return

            doc_id = self._profile_doc_id(user_id)
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            try:
                await asyncio.to_thread(self._profile.delete, ids=[doc_id])
            except Exception:
                pass

            # 直接传 documents，让 ChromaDB 内置模型生成 embedding（不依赖 Voyage API）
            await asyncio.to_thread(
                self._profile.add,
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "updated_at": datetime.now().isoformat(),
                }],
            )
            logger.info(f"用户画像已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(
            user_id,
            conv_id,
            query or (recent[-1].content if recent else ""),
        )

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过；Redis 不可用时降级为空——读失败不阻塞）
        try:
            summary = await self._redis.get(self._summary_key(user_id, conv_id)) or ""
        except Exception as ex:
            logger.warning(f"会话摘要读取失败（Redis 不可用，本轮按无摘要处理）: {ex}")
            summary = ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    async def get_summary(self, user_id: str, conv_id: str) -> str:
        """读当前会话的四字段摘要原文（v2 阶段 1 第 7 步：基线 CLI 长会话判分用）。

        判分纪律（v2-plan 1.0.9）：长会话考卷判分只认 Redis summary key 里
        的关键节点（回答里提到不算）——所以 CLI 需要一个公开的读取入口，
        之前只有 get_context 内部顺手读过一次，外部拿不到。
        没压缩过（摘要不存在）返回空串。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        return await self._redis.get(self._summary_key(user_id, conv_id)) or ""

    async def forget_user(self, user_id: str, conv_id: str) -> None:
        """删除一个用户的全部记忆：三级记忆各删各的（2026-10-15 rev3）。

          - 工作记忆 + 会话摘要（Redis）：按 (user_id, conv_id) 删 wm/summary 两个 key
          - 情景记忆 + 用户画像（ChromaDB）：按 metadata 里的 user_id 删，跨会话一并清掉

        用途：基线评测每道走对话的题开跑前调用。评测的 user_id/conv_id 固定是
        eval-<题号>，不清的话上一次运行留下的记忆会注入这一次——2026-10-12 先跑
        pilot 再跑全量，全量跑 LS-001 时第 1 轮上下文里已有 pilot 存下的摘要；
        情景记忆/画像存在 ChromaDB 里不会过期，换模型跑还会读到上一个模型的回答。
        也是「用户要求删除个人数据」的落地入口。

        与其他记忆方法不同，这里删除失败直接抛异常不吞：调用方（评测）需要
        确知记忆已清干净，静默失败等于带着脏记忆继续跑。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        await self._redis.delete(self._wm_key(user_id, conv_id), self._summary_key(user_id, conv_id))
        await asyncio.to_thread(self._episodic.delete, where={"user_id": user_id})
        await asyncio.to_thread(self._profile.delete, where={"user_id": user_id})

    async def reset_user(self, user_id: str) -> Dict[str, int]:
        """清空一个用户的全部记忆（前端「重置记忆」按钮用，2026-10-15 演示）。

        与 forget_user 的区别：forget_user 只删 (user_id, conv_id) 单个会话的
        Redis 窗口；演示用户 web-demo 每刷新页面就开一个新 conv_id，会话数量
        不定，所以这里扫前缀把该用户所有会话的 wm/summary 一并删掉。
        ChromaDB 两侧（情景记忆/用户画像）本来按 user_id 删就跨会话，同 forget_user。

        返回删除计数 {redis_keys, episodic_rows, profile_rows}，前端据此提示。
        """
        user_id = self._safe_text(user_id)
        keys: List[str] = []
        async for key in self._redis.scan_iter(f"wm:{user_id}:*"):
            keys.append(key)
        async for key in self._redis.scan_iter(f"summary:{user_id}:*"):
            keys.append(key)
        redis_keys = await self._redis.delete(*keys) if keys else 0
        # ChromaDB delete 返回删除的 id 列表（版本差异可能为 None），len 兜底
        episodic_ids = await asyncio.to_thread(self._episodic.delete, where={"user_id": user_id})
        profile_ids = await asyncio.to_thread(self._profile.delete, where={"user_id": user_id})
        return {
            "redis_keys": redis_keys,
            "episodic_rows": len(episodic_ids or []),
            "profile_rows": len(profile_ids or []),
        }

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        # LLM 摘要（四字段结构化：见模块顶部 _SUMMARY_PROMPT_TEMPLATE）
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(_SUMMARY_PROMPT_TEMPLATE.format(text=text))
        try:
            resp = await self._client.messages.create(
                # thinking 关掉（extra_body 透传）：2026-10-13 pilot 实测长对话
                # 压缩摘要时思考块吃光 4096、text 块为空 → 「摘要 0 字」、
                # summary key 写入空串（长会话丢失率 1.0 的根因）
                model=self._model, max_tokens=4096, temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        # 读旧摘要（Redis 不可用时按首次压缩处理——读侧降级为空，不阻塞）
        skey = self._summary_key(user_id, conv_id)
        try:
            old_summary = await self._redis.get(skey) or ""
        except Exception as ex:
            logger.warning(f"读取旧摘要失败（Redis 不可用，按首次压缩处理）: {ex}")
            old_summary = ""
        new_summary = await self._merge_summary(old_summary, summary)

        # 旧消息摘要存入情景记忆（ChromaDB，不依赖 Redis）——放在 Redis 写
        # 之前：Redis 挂了历史片段也能落盘，只是工作记忆/摘要窗口丢了
        await self._store_episodic(user_id, conv_id, summary)

        # 摘要 + 重置工作记忆都写 Redis：挂了一个就整体跳过（写失败不阻塞）
        try:
            await self._redis.setex(skey, 86400, new_summary)
            key = self._wm_key(user_id, conv_id)
            await self._redis.delete(key)
            for m in reversed(keep):
                await self._redis.lpush(key, json.dumps({
                    "role": m.role.value, "content": m.content,
                    "ts": m.timestamp.isoformat(), "metadata": m.metadata,
                }))
            await self._redis.expire(key, 86400)
            logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")
        except Exception as ex:
            logger.warning(f"压缩结果写入 Redis 失败（本轮压缩仅存入情景记忆）: {ex}")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        """读工作记忆；Redis 不可用时降级为空列表（读失败不阻塞主链路）。"""
        key  = self._wm_key(user_id, conv_id)
        try:
            raws = await self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        except Exception as ex:
            logger.warning(f"工作记忆读取失败（Redis 不可用，本轮按无记忆处理）: {ex}")
            return []
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, conv_id: str, query: str) -> List[str]:
        """语义检索情景记忆。Embedding 由 collection 配置的中文模型生成，不依赖外部 API。

        两段式设计：先按 conv_id 检索本会话的历史（情景记忆只在压缩时写入，
        新会话下主查询恒为空），条数不足 TOP_K 时再去掉 conv_id 条件做真正的
        跨会话检索兜底——"跨会话"能力由 fallback 达成，主查询优先同会话上下文。
        """
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            results = await self._query_episodic(
                query_text,
                n_results=self.HISTORY_TOP_K,
                # chromadb 0.5.x 起多条件过滤必须用 $and 数组，多键字典会报
                # "Expected where to have exactly one operator"
                where={"$and": [
                    {"user_id": self._safe_text(user_id)},
                    {"conv_id": self._safe_text(conv_id)},
                ]},
            )
            docs = self._extract_docs(results)
            if len(docs) < self.HISTORY_TOP_K:
                fallback = await self._query_episodic(
                    query_text,
                    n_results=self.HISTORY_TOP_K,
                    where={"user_id": self._safe_text(user_id)},
                )
                docs.extend(self._extract_docs(fallback))
            return self._dedupe_texts(docs)[: self.HISTORY_TOP_K]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, summary: str) -> None:
        """将压缩后的对话摘要存入情景记忆。Embedding 由 collection 配置的中文模型生成。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # 直接传 documents，embedding 由 collection 配置的中文模型生成
            await asyncio.to_thread(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat()}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像（取最新一条）。"""
        try:
            doc_id = self._profile_doc_id(user_id)
            direct = await asyncio.to_thread(self._profile.get, ids=[doc_id])
            if direct.get("documents"):
                return json.loads(direct["documents"][0])

            results = await asyncio.to_thread(self._profile.get, where={"user_id": user_id})
            return self._latest_profile_from_results(results)
        except Exception:
            pass
        return {}

    async def close(self) -> None:
        """关闭异步 Redis 连接。连接已断时静默——shutdown 不应因 Redis 故障报错。"""
        try:
            await self._redis.aclose()
        except Exception as ex:
            logger.warning(f"关闭 Redis 连接失败: {ex}")

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @classmethod
    def _profile_doc_id(cls, user_id: str) -> str:
        return f"{cls.PROFILE_DOC_PREFIX}{user_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value

    async def _query_episodic(
        self,
        query_text: str,
        n_results: int,
        where: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._episodic.query,
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    @staticmethod
    def _extract_docs(results: Dict[str, Any]) -> List[str]:
        """提取检索命中的文档文本，并把 metadata 的 ts 拼成「[日期 的对话]」前缀。

        ts 是时间维度，医疗场景不能丢：模型需要区分「上周还在吃的药」和
        「三年前的药」——不带时间戳的历史片段会误导用药判断（2026-10-26 补齐）。
        文档与 metadata 按索引对齐（query 返回双层列表）；任一缺失时降级为
        不带前缀（不抛，历史片段本身仍可用）。
        """
        docs = results.get("documents") or []
        if not docs:
            return []
        first = docs[0] if isinstance(docs[0], list) else docs
        metas = results.get("metadatas") or []
        first_metas = metas[0] if metas and isinstance(metas[0], list) else metas
        out = []
        for i, doc in enumerate(first):
            if not isinstance(doc, str) or not doc.strip():
                continue
            ts = ""
            if i < len(first_metas) and isinstance(first_metas[i], dict):
                raw_ts = str(first_metas[i].get("ts") or "")
                if raw_ts:
                    ts = raw_ts[:10]  # ISO 取日期部分：2026-10-15T14:30:00 → 2026-10-15
            out.append(f"[{ts} 的对话] {doc}" if ts else doc)
        return out

    @staticmethod
    def _dedupe_texts(values: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    async def _merge_summary(self, old_summary: str, new_summary: str) -> str:
        old_summary = self._safe_text(old_summary).strip()
        new_summary = self._safe_text(new_summary).strip()
        if not old_summary:
            return new_summary[: self.SUMMARY_MAX_CHARS]
        if not new_summary:
            return old_summary[: self.SUMMARY_MAX_CHARS]

        # 两段都是四字段 JSON → 确定性字段级合并（保序去重，零 LLM 成本、
        # 零遗漏风险）；任一段是旧版自由文本 → 回退 LLM 合并（兼容旧数据）
        merged = merge_field_summaries(old_summary, new_summary)
        if merged is not None:
            # 超长按安全优先级截断（见 truncate_summary），不做硬截
            return truncate_summary(merged, self.SUMMARY_MAX_CHARS)

        prompt = self._safe_text(
            f"""你是对话摘要器。请把下面两段摘要合并为一段不超过 {self.SUMMARY_MAX_CHARS} 个中文字符的摘要。
保留：用户偏好、关键实体、待办事项、约束条件、未解决问题。
只输出摘要正文，不要编号，不要解释。

旧摘要:
{old_summary}

新增摘要:
{new_summary}
"""
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                # thinking 关掉（extra_body 透传）：同 _compress 的思考块吃光坑
                max_tokens=4096,
                temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            merged = self._safe_text(extract_text_content(resp.content)).strip()
            if merged:
                return merged[: self.SUMMARY_MAX_CHARS]
        except Exception as ex:
            logger.warning(f"合并摘要失败，回退为截断拼接: {ex}")

        merged = self._safe_text(f"{old_summary}\n{new_summary}").strip()
        return merged[-self.SUMMARY_MAX_CHARS :]

    @staticmethod
    def _latest_profile_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        if not documents:
            return {}

        candidates: List[tuple[str, Dict[str, Any], str]] = []
        for idx, doc in enumerate(documents):
            if not doc:
                continue
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            ts = str(metadata.get("updated_at") or metadata.get("ts") or "")
            candidates.append((ts, metadata, doc))

        if not candidates:
            return {}

        candidates.sort(key=lambda item: item[0], reverse=True)
        latest_doc = candidates[0][2]
        try:
            return json.loads(latest_doc)
        except Exception:
            return {}
