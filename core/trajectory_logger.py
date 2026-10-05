"""
亮点：轨迹级数据采集（v2 阶段 0.1）

核心问题：SFT/RL 训练需要"模型决策过程"的完整记录，但现有 /trace 接口只保留
内存态的工具侧信息（重启即失、无 LLM 消息流），无法支撑离线训练数据构建。

本模块的答案：
  1. 完整消息流落盘 —— 每条轨迹包含 LLM 侧消息序列（tool_use / tool_result / 最终回答），
     与阶段 2 的 SFT 数据格式同构（"模具"）
  2. 失败轨迹也落盘 —— 工具失败、超轮次的轨迹同样是阶段 2 的负样本来源
  3. 后台异步写 —— 队列 + 单写线程，落盘不阻塞在线请求
  4. 隐私 —— user_id 只存 SHA256 前 16 位；data/trajectories/ 已进 .gitignore
"""
import hashlib
import json
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 前导消息的固定前缀：编排器拼进对话的"背景占位"，不是用户真实输入，
# 阶段 2 构造训练数据时需要按 kind 过滤掉它们。
_PREAMBLE_PREFIXES = ("[背景信息]", "[结构化实体]", "[角色输入契约]")


def hash_user_id(user_id: str) -> str:
    """user_id 单向哈希：轨迹落盘不保存原始标识（医疗隐私），取 SHA256 前 16 位。"""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def classify_messages(messages: List[Dict[str, Any]], agent_type: str) -> List[Dict[str, Any]]:
    """
    给一条消息流补 kind 标记，用于轨迹落盘（SFT 模具）。

    为什么是纯函数：_call_llm 发给 LLM 的消息不能夹带额外字段（会干扰 API），
    所以只在轨迹副本上做分类，不改动原始消息。

    kind 规则（按消息在序列中的结构和位置确定性判断）：
      - user + "[背景信息]/[结构化实体]/[角色输入契约]" 前缀 → context（编排器注入的背景）
      - assistant + 纯字符串 → ack（对背景的固定确认语）；若是最后一条则视为 final
      - user + 纯字符串（非前导） → query（用户真实问题）
      - assistant + 含 tool_use 块 → tool_call（模型决定调用工具）
      - user + 列表 → tool_result（工具执行结果回填）
      - assistant + 纯文本块 → final（最终回答）
    """
    classified: List[Dict[str, Any]] = []
    for index, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        entry = {"role": role, "agent_type": agent_type, "content": content}
        if role == "assistant":
            if isinstance(content, str):
                entry["kind"] = "final" if index == len(messages) - 1 else "ack"
            elif any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                entry["kind"] = "tool_call"
            else:
                entry["kind"] = "final"
        elif isinstance(content, list):
            entry["kind"] = "tool_result"
        elif isinstance(content, str) and content.startswith(_PREAMBLE_PREFIXES):
            entry["kind"] = "context"
        else:
            entry["kind"] = "query"
        classified.append(entry)
    return classified


class TrajectoryLogger:
    """
    轨迹 JSONL 落盘器：record() 入队即返回（非阻塞），后台单线程按天写文件。

    设计取舍：
      - 尽力而为 —— 队列写满或磁盘异常时丢弃并告警，轨迹采集绝不能反压在线请求
      - 按天切分 —— trajectories-YYYY-MM-DD.jsonl，阶段 2 构建数据时可按日期分批取
      - daemon 线程 —— 进程退出时未落盘的轨迹会丢（服务正常关停会走 close()）
    """

    _QUEUE_MAXSIZE = 10000  # 队列上限：磁盘写慢时兜底，防止内存无限增长
    _SENTINEL = object()    # 哨兵：close() 时通知写线程退出

    def __init__(self, log_dir: str = "data/trajectories", enabled: bool = True):
        self._enabled = enabled
        self._log_dir = Path(log_dir)
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        if enabled:
            self._thread = threading.Thread(target=self._writer_loop, name="trajectory-writer", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, trajectory: Dict[str, Any]) -> None:
        """入队一条轨迹。非阻塞：写盘发生在后台线程，主链路不被拖慢。"""
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(trajectory)
        except queue.Full:
            logger.warning("轨迹队列已满（%s 条），丢弃当前轨迹", self._QUEUE_MAXSIZE)

    def close(self) -> None:
        """优雅停止：投递哨兵并等待后台线程把队列写空。重复调用安全。"""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            logger.warning("轨迹队列已满，close 时丢弃未落盘轨迹")
        self._thread.join(timeout=5.0)
        self._thread = None

    # ── 后台写线程 ────────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        """单写线程主循环：消费队列直到收到哨兵。"""
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            try:
                self._append(item)
            except Exception as ex:  # 磁盘异常只影响轨迹采集，不影响服务
                logger.warning("轨迹写盘失败: %s", ex)

    def _append(self, trajectory: Dict[str, Any]) -> None:
        """追加一行 JSONL；文件名按当天日期切分。"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"trajectories-{date_str}.jsonl"
        # ensure_ascii=False 保留中文可读性；default=str 兜底非 JSON 原生类型
        line = json.dumps(trajectory, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
