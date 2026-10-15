"""
基线评测跑批器（v2 阶段 1 第 7 步）。

核心问题：五套考卷怎么跑、怎么判、怎么汇总成一份 baseline.json——这是
v2 训练闭环「改造前」数字的唯一来源（后续 SFT/GRPO 全部对比它），所以
流程必须与生产链路同构、判分必须可复现、坏数据必须拦住。

本模块的答案（与 docs/v2-plan.md 1.2 第 7 步设计一致）：
  1. 串行跑题 —— 被测链路（ChatService/orchestrator）有实例级状态（Agent
     统计、识别器缓存、轨迹队列），必须串行；judge 判分与跑题分离，可并发
  2. 轨迹关联 —— 不依赖 request_id（编排器内部随机生成），按 meta.conv_id
     关联；CLI 用确定性 conv_id `eval-<case_id>`。同一 conv_id 历次运行的
     轨迹都在盘上，所以跑题前记下 started_at，只取这之后的轨迹（多轮题
     等齐全部轮次再判）；旧 run 的行没有 started_at，从它记录的最后一轮
     轨迹往前取「剧本用户轮数」条（select_run_turns，2026-10-15 rev3）
  3. 数据质量门 —— 参与判分的轨迹先过 assert_eval_data_quality
     （cached/rerank_degraded/检索失败 → 中止整个 run，宁可重跑不产脏基线）
  4. 方向 judge 缓存 —— 多跳 success 与安全违规率都要判方向，同一个
     CachingDirectionJudge 实例缓存 case_id 结果（temperature 0 + 同答案，
     缓存安全），27 道安全档题省一半 judge 调用
  5. 断点续跑 —— cases.jsonl 每行一份，--resume 跳过已完成行；--cases 只
     重跑指定题；--merge 配合 --cases 部分重跑（新行覆盖旧行，其余保留；
     轨迹质量报告与急症收件箱同样按题合并，不会只剩重跑的那几题）
  6. webhook 模拟收件箱 —— 急症红旗题会触发 EmergencyAgent 的告警推送，
     默认接本地 mock server（随机端口）收告警落盘，绝不外发真实通知
  7. 记忆隔离 —— 每道走对话的题开跑前清掉该评测用户的三级记忆
     （2026-10-15 rev3 修复：确定性 id 下 pilot 留下的记忆串进了全量跑，
     见 _reset_eval_memory）
  8. 判卷与被测系统分开配置 —— 判卷读 build_judge_cfg（OPENMED_JUDGE_*），
     换 Qwen3 等自部署模型跑时判卷仍是固定的第三方模型（2026-10-15）

五套考卷的判分分工（「确定性为主、LLM 兜底」纪律）：
  - intent / emergency / qa single：跑题时确定性判完（对比期望/检索统计；
    qa single 是 MRR/hit@k，正确片段 = gold_chunks ∪ alt_gold_chunks）
  - qa multi：LLMJudge 四维打分（context 传 gold_answer 校准，主判分）
  - multihop：score_multihop（实体链确定性 + 方向 judge 只安全档）
  - long_session：score_long_session（只读 Redis summary 四字段）
  - safety：score_safety_case（三道预检 + 方向 judge 复用缓存）
  - 生成质量层（方案 C，2026-10-13 用户拍板）：qa multi / multihop 加
    faithfulness + answer_relevancy，进 report 不进 headline。
    long_session / emergency 不评（2026-10-13 用户拍板）：长会话考的是
    记忆留存（判分依据是 Redis summary，最终回答无检索上下文可对照），
    急症是固定安全模板——faithfulness 对这两集没有区分度。
    2026-10-15 rev3 口径：faithfulness 的检索上下文取整场对话所有轮次；
    qa multi 里走到急症固定回复的不评；整场没检索就作答的不调 judge、
    不进 faithfulness 均值、报告单独计数（「没检索」与「没照着检索说」
    是两种毛病，混在一个均值里分不清）

换被测模型时（--skip-model-independent）：qa single 直测检索、强制 LLM 失败
子集短路 LLM，两者都不经过被测模型，结果与模型无关，跳过不跑。

数据质量纪律：跑题产生的检索必须真实发生——qa single 走
tool_manager.search_with_rerank（与 /search 端点同一条链路，含精排），
评测模式（OPENMED_EVAL_MODE=1）下工具缓存已禁用。
"""
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evaluation.trajectory_reader import Trajectory, TrajectoryReader, assert_eval_data_quality

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────

# Redis/记忆与轨迹的 user_id/conv_id 都用这个前缀（隐私哈希前先隔离评测数据）
EVAL_USER_PREFIX = "eval-"

# 轨迹等待：落盘是后台线程，跑完题立即读会漏；每 2s 轮询，最多 30s
_TRAJECTORY_WAIT_INTERVAL_S = 2.0
_TRAJECTORY_WAIT_TIMEOUT_S = 30.0

# judge 并发上限：DeepSeek 速率友好，4 路并发判分
_JUDGE_CONCURRENCY = 4

# qa single 直测的 top_k（与 /search 端点默认一致）
_QA_SINGLE_TOP_K = 5

# 预热 query：触发 reranker（Cross-Encoder）冷加载，避免第一题被冷启动拖慢
_WARMUP_QUERIES = ("感冒了挂哪个科", "华法林和什么不能一起吃", "高血压饮食注意什么")

# qa multi 剧本轮前缀：剧本里的助手轮是出题时的引导话术，评测时不喂
# （系统自己生成回答）；只喂「用户：」轮
_TURN_USER_PREFIX = "用户："
_TURN_ASSISTANT_PREFIX = "助手："


class EvalAbortError(Exception):
    """评测中止：数据质量违规 / 不安全配置。整个 run 中止，不产 baseline。"""


class _Progress:
    """串行跑题/判分的轻量进度条（零新依赖，第 7 步配套，2026-10-13 用户要的）。

    为什么不用 tqdm：只为一个进度条引一个依赖不值，标准库 print 几行就能写。
    为什么字符条用 # 和 -：█/░ 在 Windows 控制台（GBK）会编码报错，
    ASCII 字符在任何终端都安全（见环境注意事项「Windows 控制台显示乱码」）。

    输出到 stderr（stdout 可能被重定向），\r 动态刷新同一行；
    跑完自动补换行，不覆盖下一段输出。enabled=False 时全部 no-op
    （单测默认关，测试输出干净）。
    """

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._total = 0
        self._done = 0
        self._label = ""

    def note(self, msg: str) -> None:
        """非动态提示行（如「预热中」）：enabled 才打印，带换行。"""
        if self._enabled:
            print(msg, file=sys.stderr, flush=True)

    def start(self, label: str, total: int) -> None:
        """开一段新进度（跑题/判分各一段），total=0 时什么都不打。"""
        if not self._enabled:
            return
        self._label = label
        self._total = total
        self._done = 0
        self._render("")

    def tick(self, note: str = "") -> None:
        """完成一项：计数 +1 并刷新（未 start 或已跑完时 no-op）。"""
        if not self._enabled:
            return
        self._done += 1
        self._render(note)

    def _render(self, note: str) -> None:
        if not self._total:
            return
        pct = self._done * 100 // self._total
        filled = self._done * 20 // self._total  # 20 格字符条
        bar = "#" * filled + "-" * (20 - filled)
        line = f"\r{self._label} [{self._done}/{self._total}] |{bar}| {pct}% {note}"
        print(line, end="", file=sys.stderr, flush=True)
        if self._done >= self._total:
            print(file=sys.stderr)  # 收尾换行


def build_run_id() -> str:
    """默认 run_id：按天一个（同一天重跑覆盖同目录，--run-id 可显式指定）。"""
    return f"baseline-{datetime.now().strftime('%Y%m%d')}"


def case_conv_id(case_id: str) -> str:
    """确定性 conv_id：轨迹关联与记忆隔离都靠它，不能随机。"""
    return f"{EVAL_USER_PREFIX}{case_id}"


def qa_single_stats(retrieved: List[str], relevant: List[str]) -> Dict[str, Any]:
    """single 档一题的检索统计（确定性）：MRR / hit@1 / hit@5。

    relevant = gold_chunks ∪ alt_gold_chunks（出题依据 + rev3 补标的其他含答案片段）。
    跑题与 --rejudge-from 离线重算共用这一个函数，两条路径口径不会分叉。
    """
    from evaluation.ragas_metrics import hit_at_k, mrr
    return {
        "mrr": mrr(retrieved[:_QA_SINGLE_TOP_K], relevant),
        "hit_at_1": int(hit_at_k(retrieved, relevant, 1)),
        "hit_at_5": int(hit_at_k(retrieved, relevant, _QA_SINGLE_TOP_K)),
    }


def select_run_turns(trajectories: List[Trajectory], started_at: str = "",
                     anchor_id: str = "", expected_turns: int = 1) -> List[Trajectory]:
    """从同一 conv_id 的全部轨迹（按时间升序）里挑出「这一次运行」的那几轮。

    为什么要挑：conv_id 是确定性的 eval-<题号>，pilot / 全量 / 换模型重跑都
    写进同一批轨迹文件，不挑就会把上一次运行的轮次混进来（2026-10-15 rev3 查出）。
      - 新跑的行有 started_at（跑题前记的时间）：取这之后的全部轨迹
      - 旧 run 的行没有 started_at：从它记录的最后一轮（anchor_id）往前
        取 expected_turns 条（= 剧本里用户发言的轮数，一轮一条轨迹）
    """
    if started_at:
        return [t for t in trajectories if t.timestamp >= started_at]
    upto = trajectories
    if anchor_id:
        idx = next((i for i, t in enumerate(trajectories) if t.trajectory_id == anchor_id), None)
        if idx is None:
            return []
        upto = trajectories[: idx + 1]
    return upto[-max(1, expected_turns):]


def user_turn_count(case: Optional[Dict[str, Any]]) -> int:
    """一道题会发几轮用户消息（= 产几条轨迹）：qa multi 数剧本里非助手轮，
    长会话数 speaker=user 的轮，其余单轮题 1。"""
    if not case:
        return 1
    turns = case.get("turns") or []
    if not turns:
        return 1
    if isinstance(turns[0], dict):  # 长会话：SessionTurn
        return max(1, sum(1 for t in turns if t.get("speaker") == "user"))
    return max(1, sum(1 for t in turns if not str(t).startswith(_TURN_ASSISTANT_PREFIX)))


# ── 急症告警模拟收件箱 ────────────────────────────────────────────────────────

# 跑完急症集后等待异步推送落网的秒数（EmergencyAgent 用 create_task 发
# webhook 不 await，本地回环毫秒级，2s 绰绰有余）
_WEBHOOK_DRAIN_S = 2.0


class EmergencyWebhookInbox:
    """急症告警的模拟收件箱（v2-plan 1.3 webhook 机制，2026-10-13 用户拍板）。

    为什么不用「设环境变量指向 mock」：EmergencyAgent._webhook_url 是类属性，
    模块 import 时快照 env——CLI 里 argparse 之后再设已太晚，所以这里显式
    把类属性指到 mock 地址（一行注入，生产代码零改动）。

    实现：标准库 http.server 起在随机端口（零新依赖、不抢固定端口），
    POST 收到的告警 JSON 原样存进内存，退出时落盘 jsonl 供留档。
    """

    def __init__(self, out_path: Any):
        self._out_path = Path(out_path)
        self.received: List[Dict[str, Any]] = []
        self.port: int = 0
        self._server: Any = None
        self._thread: Any = None

    def __enter__(self) -> "EmergencyWebhookInbox":
        import http.server
        import threading

        inbox = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # 收告警：读 body → 追加收件箱 → 回 200
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    payload = {"raw": body.decode("utf-8", errors="replace")}
                payload["received_at"] = datetime.now().isoformat()
                inbox.received.append(payload)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, *args):  # 评测收件箱静默，不刷访问日志
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("急症告警模拟收件箱已启动: http://127.0.0.1:%s", self.port)
        return self

    def __exit__(self, *exc: Any) -> None:
        import time

        # 先等已在途的推送落网，再关接收端口并落盘（不留内存态）
        time.sleep(_WEBHOOK_DRAIN_S)
        self._server.shutdown()
        self._thread.join(timeout=5.0)
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("w", encoding="utf-8") as f:
            for payload in self.received:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        logger.info("急症告警收件箱共收到 %s 条，已落盘 %s", len(self.received), self._out_path)


def render_webhook_summary(payload: Dict[str, Any]) -> str:
    """告警的人话摘要（模板 + 结构化实体拼句，内容固定不判分）。"""
    entities = payload.get("entities") or {}
    return (
        f"用户 {payload.get('user_id') or '?'} 说「{payload.get('message') or ''}」，"
        f"触发急症升级（紧急度={payload.get('urgency') or '?'}，"
        f"意图={payload.get('intent') or '?'}，"
        f"结构化实体={json.dumps(entities, ensure_ascii=False) or '无'}），请人工客服跟进"
    )


# ── 方向 judge 缓存 ───────────────────────────────────────────────────────────

class CachingDirectionJudge:
    """方向 judge 的缓存包装：同 case_id 只判一次，结果共享给两个判分器。

    为什么需要：多跳 success（score_multihop）与安全违规率（score_safety_case）
    都要判「回答方向对不对」，同一题判两次既翻倍成本、又可能因为两次调用
    结果不一致让报告自相矛盾。temperature 0 + 同一问题同一答案 → 缓存安全。
    """

    def __init__(self, inner: Callable):
        self._inner = inner
        self._cache: Dict[str, str] = {}
        self.calls = 0  # 实际调用底层 judge 的次数（测试/报告用）

    async def __call__(self, question: str, answer: str, case: Dict[str, Any]) -> str:
        cid = str(case.get("id") or question)
        if cid not in self._cache:
            self.calls += 1
            self._cache[cid] = await self._inner(question, answer, case)
        return self._cache[cid]


# ── 跑批器 ────────────────────────────────────────────────────────────────────

class BaselineRunner:
    """五套考卷的跑题 + 判分 + 聚合 + 落盘。

    ctx 是 core/app_factory.assemble() 的产物（与生产同构）；测试传假组件。
    judge_client 默认按判卷配置（build_judge_cfg，与被测系统分开）构造真客户端；
    测试传 test_ragas_metrics.FakeClient。judge_cfg 不传时现读环境变量。
    """

    def __init__(
        self,
        ctx: Any,
        run_id: Optional[str] = None,
        results_root: Optional[Any] = None,
        trajectory_dir: str = "data/trajectories",
        judge_client: Any = None,
        embedder: Optional[Callable[[str], Optional[List[float]]]] = None,
        allow_webhook: bool = False,
        progress: bool = False,
        judge_cfg: Optional[Dict[str, Any]] = None,
    ):
        self._ctx = ctx
        self.run_id = run_id or build_run_id()
        self._results_root = Path(results_root) if results_root else (
            Path(__file__).resolve().parent.parent / "data" / "eval" / "results" / self.run_id
        )
        self._trajectory_dir = trajectory_dir
        self._allow_webhook = allow_webhook
        self._embedder = embedder
        self._progress = _Progress(progress)
        self._webhook_payloads: List[Dict[str, Any]] = []  # 收件箱收到的告警（报告用）
        self._quality_report: Dict[str, Any] = {"total": 0, "violations": [], "missing": []}
        # 每个 conv_id 本次运行的全部轮次轨迹（_collect_trajectories 填；qa multi
        # faithfulness 用整场对话的检索内容判）
        self._conv_turns: Dict[str, List[Trajectory]] = {}
        # 换被测模型时跳过与模型无关的两部分（qa single / 强制 LLM 失败子集）
        self._skip_model_independent = False
        # 判分阶段取 case 原文（gold 字段）的考卷快照：实例级，不同 runner 互不串
        self._cases_by_id: Dict[str, Dict[str, Any]] = {}

        from api.chat_service import ChatService
        self._chat = ChatService(ctx.orchestrator, ctx.memory)

        # 判卷配置与被测系统分开（2026-10-15）：被测系统换 Qwen3 时判卷不跟着换
        if judge_cfg is None:
            from core.app_factory import build_judge_cfg
            judge_cfg = build_judge_cfg(ctx.cfg)
        if judge_client is None:
            # 被测系统走 OpenAI 协议（vLLM 自部署模型）却没单独配判卷模型：判卷会
            # 沿用 ANTHROPIC_MODEL（此时已改成被测模型名）= 自己给自己打分，直接拦下
            system_protocol = os.getenv("OPENMED_LLM_PROTOCOL", "anthropic").strip().lower()
            if system_protocol == "openai" and not judge_cfg.get("explicit"):
                raise EvalAbortError(
                    "被测系统走 OpenAI 协议（自部署模型）时必须单独配置判卷模型："
                    "设 OPENMED_JUDGE_MODEL（及 OPENMED_JUDGE_API_KEY / OPENMED_JUDGE_BASE_URL），"
                    "否则判卷会变成被测模型自己。")
            from core.llm_utils import create_llm_client
            judge_client = create_llm_client(judge_cfg["api_key"], judge_cfg.get("base_url"),
                                             protocol=judge_cfg.get("protocol", "anthropic"))
        self._judge_client = judge_client
        self._judge_model = judge_cfg.get("model", "")
        self._system_model = ctx.cfg.get("model", "")

        # LLM 路强制失败的识别器（强制 LLM 失败子集专用新实例：缓存隔离）。
        # 它是被测系统的识别器（只是 LLM 路被短路），用被测系统的模型配置
        from core.intent_recognizer import IntentRecognizer
        self._forced_recognizer = IntentRecognizer(
            api_key=ctx.cfg["api_key"], base_url=ctx.cfg.get("base_url"),
            model=self._system_model, force_llm_failure=True,
        )

    # ── 主流程 ────────────────────────────────────────────────────────────────

    async def run_all(
        self,
        exams: Dict[str, List[Dict[str, Any]]],
        manifest: Optional[Dict[str, Any]] = None,
        pilot: bool = False,
        case_ids: Optional[List[str]] = None,
        resume: bool = False,
        merge: bool = False,
        skip_model_independent: bool = False,
    ) -> Dict[str, Any]:
        """完整流程：预热 → 跑题 → 轨迹收集+质量断言 → LLM 判分 → 聚合 → 落盘。

        返回 build_baseline 的完整数据（同时也已写盘 results/<run_id>/ 四件
        + data/eval/baseline.json）。exams 的 case 是纯 dict（loader 的
        dataclass 由调用方 asdict 转好传入）。

        resume（断点续跑）：跳过 cases.jsonl 已有行（跑过即完成，判分已落盘）；
        merge（部分重跑）：指定题强制重跑、新行覆盖旧行，其余行原样保留，
        常配 --cases 用（单独 resume 会把全部题重跑一遍白花钱）。两种模式下
        轨迹质量报告与急症收件箱也按题合并（旧 run 的条目保留、重跑题的换新）。
        skip_model_independent（换被测模型时用）：不跑 qa single 与强制 LLM
        失败子集——两者都不经过被测模型，换模型结果一定不变。
        """
        self._ensure_output_dir()
        self._skip_model_independent = skip_model_independent

        # 0. 断点续跑/部分重跑：读已有 cases.jsonl（merge 时保留未重跑的行）
        #    以及上一次的轨迹质量报告与急症收件箱（落盘时与本次结果合并）
        existing_rows: Dict[str, Dict[str, Any]] = {}
        prev_quality: Optional[Dict[str, Any]] = None
        prev_webhooks: List[Dict[str, Any]] = []
        if resume or merge:
            existing_rows = self._load_existing_rows()
            prev_quality, prev_webhooks = self._load_previous_side_outputs()

        wanted = set(case_ids) if case_ids else None
        rows: List[Dict[str, Any]] = []

        def pick(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            picked = self._pick(cases, pilot, wanted)
            if not existing_rows or merge:
                return picked
            # resume：跳过已有行
            return [c for c in picked if c.get("id") not in existing_rows]

        # 各集选题先算好（pick 是纯函数，进度条总数与实际跑的题一一对应）
        intent_cases = pick(exams.get("intent", []))
        forced_cases = [] if skip_model_independent else pick(
            [c for c in exams.get("intent", []) if c.get("llm_failure")])
        qa_cases = pick([c for c in exams.get("qa", [])
                         if not (skip_model_independent and c.get("kind") == "single")])
        mh_cases = pick(exams.get("multihop", []))
        ls_cases = pick(exams.get("long_session", []))
        em_cases = pick(exams.get("emergency", []))

        # 1. 预热：触发 reranker 冷加载（warmup 不走编排器，不产生轨迹）
        self._progress.note("预热检索中（reranker 冷加载，首次约 20s）…")
        await self._warmup()
        self._progress.start("跑题", len(intent_cases + forced_cases + qa_cases
                                        + mh_cases + ls_cases + em_cases))

        # 2. 串行跑五套（跑题阶段内完成确定性判分）
        rows += await self.run_intent(intent_cases)
        rows += await self.run_forced_failure(forced_cases)
        single_rows, multi_rows = await self.run_qa(qa_cases)
        rows += single_rows + multi_rows
        rows += await self.run_multihop(mh_cases)
        rows += await self.run_long_session(ls_cases)
        rows += await self.run_emergency(em_cases)

        if not rows:
            logger.info("没有需要跑的题（--resume 且全部已完成）")

        # 3. 轨迹收集（多跳判分必需；质量断言范围 = 全部 chat 轨迹）
        traj_index = await self._collect_trajectories(rows)
        self._quality_report = self._assert_quality(rows, traj_index)

        # 4. LLM 判分（qa multi / multihop / long_session / safety）
        await self._judge_rows(rows, traj_index)

        # 5. 合并续跑行 + 落盘 cases.jsonl
        merged = {r["case_id"]: r for r in existing_rows.values()}
        merged.update({r["case_id"]: r for r in rows})
        all_rows = [merged[cid] for cid in sorted(merged)]
        self._strip_generation_quality(all_rows)
        self._save_rows(all_rows)

        # 5b. 续跑/部分重跑：轨迹质量报告与急症收件箱按题合并——不合并的话
        #     报告只剩本次重跑的几题（如轨迹总数从 141 变成 10、告警从 20 变成 3）
        if prev_quality is not None:
            self._quality_report = self._merge_quality_reports(
                prev_quality, self._quality_report, rows, existing_rows)
            rerun_users = {case_conv_id(r["case_id"]) for r in rows}
            self._webhook_payloads = [p for p in prev_webhooks
                                      if p.get("user_id") not in rerun_users] + self._webhook_payloads

        # 6. 聚合 + baseline + report
        baseline = self._build_baseline(all_rows, manifest)
        self._save_outputs(baseline, all_rows)
        return baseline

    async def run_rejudge(
        self,
        prev_run_id: str,
        exams: Dict[str, List[Dict[str, Any]]],
        manifest: Optional[Dict[str, Any]] = None,
        case_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """从旧 run 产物重聚合出新 run（不全量重跑）。

        用途（2026-10-13 用户指令）：判分口径修订后，把旧 run 的 cases.jsonl
        拿过来——重判指定题（--cases），其余行原样保留，按新口径清理后
        聚合出新的 baseline.json + report.md，落盘新 run_id 目录。

        重判只依赖旧行里的 detail 快照（summary / gold_answer 等判分输入
        跑题时就存下来了）与旧轨迹文件（data/trajectories 按天落盘还在），
        所以不需要重跑对话；轨迹质量报告与急症告警直接复制旧 run 的。
        qa single 不需要 --cases 指定：跑题时存了前 5 个检索结果，按当前
        考卷的正确片段全部离线重算（确定性、零成本，rev3 改 MRR 时用）。
        """
        self._ensure_output_dir()

        # 0. 读旧 run 产物
        prev_dir = self._results_root.parent / prev_run_id
        prev_cases = prev_dir / "cases.jsonl"
        if not prev_cases.exists():
            raise FileNotFoundError(
                f"旧 run 产物不存在: {prev_cases}（run_id 对不上？）")
        rows = [json.loads(line) for line in prev_cases.read_text(encoding="utf-8").splitlines()
                if line.strip()]

        # 1. 考卷快照（判分阶段 _find_case 取 gold 字段用）
        self._cases_by_id = {str(c["id"]): dict(c)
                             for cases in exams.values() for c in cases}

        # 1b. qa single 按当前考卷离线重算（不调 LLM、不重新检索）
        self._rescore_qa_single(rows)

        # 2. 复制旧 run 的轨迹质量报告与急症收件箱（轨迹/告警没重跑，结果不变）
        quality_path = prev_dir / "trajectory_quality_report.json"
        if quality_path.exists():
            self._quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
        else:
            self._quality_report = {"total": 0, "violations": [], "missing": []}
        webhooks_path = prev_dir / "emergency_webhooks.jsonl"
        if webhooks_path.exists():
            self._webhook_payloads = [json.loads(line) for line in
                                      webhooks_path.read_text(encoding="utf-8").splitlines()
                                      if line.strip()]

        # 3. 重判指定题（其余行原样保留）；judge 分发与 run_all 共用 _judge_rows
        wanted = set(case_ids) if case_ids else None
        to_rejudge = [r for r in rows if wanted is None or r["case_id"] in wanted]
        if wanted is not None:
            unknown = wanted - {r["case_id"] for r in rows}
            if unknown:
                logger.warning("旧 run 里没有这些题号，跳过: %s", sorted(unknown))
        if to_rejudge:
            traj_index = await self._collect_trajectories(to_rejudge)
            await self._judge_rows(to_rejudge, traj_index)
            rejudged = {r["case_id"]: r for r in to_rejudge}
            rows = [rejudged.get(r["case_id"], r) for r in rows]

        # 4. 口径清理 + 聚合 + 落盘（与 run_all 同路径）
        self._strip_generation_quality(rows)
        self._save_rows(rows)
        baseline = self._build_baseline(rows, manifest)
        self._save_outputs(baseline, rows)
        return baseline

    def _ensure_output_dir(self) -> None:
        self._results_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _strip_generation_quality(rows: List[Dict[str, Any]]) -> None:
        """口径清理：long_session / emergency 不评生成质量层（2026-10-13 用户拍板）。

        聚合前统一删除这两个集行里的 scores.generation_quality——新跑的行
        本来就不会有；--merge 保留的旧行（旧口径跑出来的）可能还带着，
        不清理的话 report.md 会继续列它们的 faithfulness/relevancy。
        """
        for row in rows:
            if row.get("exam") in ("long_session", "emergency"):
                (row.get("scores") or {}).pop("generation_quality", None)

    def _rescore_qa_single(self, rows: List[Dict[str, Any]]) -> None:
        """按当前考卷的正确片段离线重算 qa single 统计（--rejudge-from 用）。

        single 档的前 5 个检索结果跑题时已存进 detail.retrieved_chunks，判分
        口径变了（rev3：补标 alt_gold_chunks、F1@5 改 MRR）不需要重新检索。
        检索失败的行（没有 retrieved_chunks）保持原样。
        """
        for row in rows:
            detail = row.get("detail") or {}
            if row.get("exam") != "qa" or detail.get("gold_chunks") is None:
                continue  # 只处理 single 档（multi 档 detail 没有 gold_chunks）
            case = self._find_case(row["case_id"])
            if case is None or not detail.get("retrieved_chunks"):
                continue
            gold = [str(g) for g in case.get("gold_chunks") or []]
            alt = [str(g) for g in case.get("alt_gold_chunks") or []]
            detail["gold_chunks"], detail["alt_gold_chunks"] = gold, alt
            row["scores"] = qa_single_stats(list(detail["retrieved_chunks"]), gold + alt)

    def _load_previous_side_outputs(self) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """续跑/部分重跑时读本 run 目录里上一次的轨迹质量报告与急症收件箱。

        目录里没有（第一次跑）返回 (None, [])——调用方据此决定要不要合并。
        """
        quality_path = self._results_root / "trajectory_quality_report.json"
        quality = (json.loads(quality_path.read_text(encoding="utf-8"))
                   if quality_path.exists() else None)
        webhooks_path = self._results_root / "emergency_webhooks.jsonl"
        webhooks = ([json.loads(line) for line in
                     webhooks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if webhooks_path.exists() else [])
        return quality, webhooks

    def _merge_quality_reports(self, prev: Dict[str, Any], new: Dict[str, Any],
                               new_rows: List[Dict[str, Any]],
                               existing_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """轨迹质量报告按题合并：旧报告里重跑题的条目换成本次的，其余保留。

        旧报告只存了总数（没存逐题清单），重跑题在旧 run 里是否计入过总数，
        看它旧行有没有 trajectory_id（_assert_quality 只给计数的题写这个字段）。
        """
        rerun = {r["case_id"] for r in new_rows if self._expects_trajectory(r)}
        counted_before = sum(1 for cid in rerun
                             if (existing_rows.get(cid) or {}).get("trajectory_id"))
        return {
            "total": max(0, prev.get("total", 0) - counted_before) + new.get("total", 0),
            "violations": [v for v in prev.get("violations", [])
                           if v.get("case_id") not in rerun] + list(new.get("violations", [])),
            "missing": [m for m in prev.get("missing", []) if m not in rerun]
                       + list(new.get("missing", [])),
        }

    async def _reset_eval_memory(self, uid: str) -> None:
        """跑一道走对话的题之前，清掉这个评测用户的三级记忆（2026-10-15 rev3）。

        为什么必须清：评测 user_id/conv_id 是确定性的 eval-<题号>，以前跑题前
        从不清——2026-10-12 先跑三次 pilot 再跑全量，全量跑 LS-001 时第 1 轮
        上下文里已经有 pilot 存下的摘要（正好是这道题要考的病史），rev2 里
        10 道 pilot 题被污染；情景记忆/画像存在 ChromaDB 不会过期，不清的话
        换 Qwen3 跑会读到 DeepSeek 留下的回答。只清 eval- 前缀（防误删真实用户）。
        """
        if not uid.startswith(EVAL_USER_PREFIX):
            raise EvalAbortError(f"拒绝清理非评测用户的记忆: {uid}")
        await self._ctx.memory.forget_user(uid, uid)

    @staticmethod
    def _mark_run(row: Dict[str, Any], started_at: str, turns_sent: int) -> Dict[str, Any]:
        """给走对话的题记下本次运行的起点与发了几轮（_collect_trajectories 靠它
        只取本次运行的轨迹、并等齐全部轮次）。"""
        row["started_at"] = started_at
        row["turns_sent"] = turns_sent
        return row

    @staticmethod
    def _pick(cases: List[Dict[str, Any]], pilot: bool,
              wanted: Optional[set]) -> List[Dict[str, Any]]:
        """pilot 截断与 --cases 过滤（先过滤后截断，保持文件顺序）。"""
        if wanted is not None:
            cases = [c for c in cases if c.get("id") in wanted]
        return cases[:3] if pilot else cases

    async def _warmup(self) -> None:
        """3 条检索预热：Cross-Encoder 首次加载 ~18s，不预热会拖慢第一题。"""
        for query in _WARMUP_QUERIES:
            try:
                await self._ctx.tool_manager.search_with_rerank("knowledge_search", query, top_k=3)
            except Exception as ex:
                logger.warning("预热检索失败（不中止，跑题时还会暴露）: %s", ex)

    # ── 跑题：意图集 ───────────────────────────────────────────────────────────

    async def run_intent(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """意图集：单句直接识别（与生产 /chat 同识别器），确定性判分。"""
        rows: List[Dict[str, Any]] = []
        for case in cases:
            t0 = time.perf_counter()
            result = await self._ctx.recognizer.recognize(case["message"])
            actual = result.intent.value if result.intent else "other"
            expected = case["expected_intent"]
            rows.append(self._row(case, "intent", passed=actual == expected, scores={
                "confidence": round(result.confidence, 4),
                "source": result.reasoning,
            }, detail={"expected": expected, "actual": actual}))
            self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
        return rows

    # ── 跑题：强制 LLM 失败子集 ──────────────────────────────────────────────

    async def run_forced_failure(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 失败降级子集：forced 识别器跑投票；路由准确率走 _route_decision 纯函数。

        两档度量（v2-plan 1.2 第 6 步）：
          ① 降级投票意图准确率：forced 识别器（emb/关键词降级）predicted vs expected
          ② 降级路由准确率：expected_intent 注入 _route_decision 生成期望路由，
             actual_intent 注入生成实际路由，对比 primary_agent——零额外 LLM 成本
        """
        from agents.agent_orchestrator import Request
        from core.intent_recognizer import IntentCategory

        rows: List[Dict[str, Any]] = []
        for case in cases:
            t0 = time.perf_counter()
            result = await self._forced_recognizer.recognize(case["message"])
            actual = result.intent.value if result.intent else "other"
            expected = case["expected_intent"]
            uid = case_conv_id(case["id"])

            # 期望/实际路由都用同一个确定性函数算（urgency 走关键词门控，
            # 与 LLM 无关，两次注入相同）——差异纯由意图错传导
            try:
                expected_req = Request(message=case["message"], user_id=uid, conv_id=uid,
                                       intent=IntentCategory(expected), urgency=result.urgency)
                actual_req = Request(message=case["message"], user_id=uid, conv_id=uid,
                                     intent=result.intent, urgency=result.urgency)
                expected_primary = self._ctx.orchestrator._route_decision(expected_req).primary_agent.value
                actual_primary = self._ctx.orchestrator._route_decision(actual_req).primary_agent.value
                routing_correct = expected_primary == actual_primary
            except Exception as ex:
                logger.warning("%s 路由计算失败，按路由错误计: %s", case.get("id"), ex)
                expected_primary, actual_primary, routing_correct = "", "", False

            voting_correct = actual == expected
            rows.append(self._row(case, "intent_llm_failure", passed=voting_correct, scores={
                "voting_correct": voting_correct,
                "routing_correct": routing_correct,
            }, detail={
                "expected": expected, "actual": actual,
                "expected_primary": expected_primary, "actual_primary": actual_primary,
            }))
            self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
        return rows

    # ── 跑题：问答集 ───────────────────────────────────────────────────────────

    async def run_qa(self, cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """问答集：single 走 tool_manager.search_with_rerank 直测检索链路；
        multi 走 ChatService 多轮（剧本助手轮不喂）。"""
        single_rows: List[Dict[str, Any]] = []
        multi_rows: List[Dict[str, Any]] = []
        for case in cases:
            t0 = time.perf_counter()
            if case["kind"] == "single":
                single_rows.append(await self._run_qa_single(case))
            else:
                multi_rows.append(await self._run_qa_multi(case))
            self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
        return single_rows, multi_rows

    async def _run_qa_single(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """single 档：检索统计在跑题时确定性算完（judge 阶段不参与）。

        正确片段 = gold_chunks（出题依据）∪ alt_gold_chunks（rev3 补标的其他
        含答案片段）；统计 MRR / hit@1 / hit@5（qa_single_stats）。
        """
        gold = [str(g) for g in (case.get("gold_chunks") or [])]
        alt = [str(g) for g in (case.get("alt_gold_chunks") or [])]
        result = await self._ctx.tool_manager.search_with_rerank(
            "knowledge_search", case["question"], top_k=_QA_SINGLE_TOP_K)
        if not result.success or not isinstance(result.data, list):
            return self._row(case, "qa", passed=False, scores={}, detail={
                "gold_chunks": gold, "alt_gold_chunks": alt, "retrieved_chunks": [],
            }, failure_reason=f"检索失败: {result.error}")

        retrieved = [str(item.get("id") or "") for item in result.data]
        row = self._row(case, "qa", passed=True, scores=qa_single_stats(retrieved, gold + alt),
                        detail={"gold_chunks": gold, "alt_gold_chunks": alt,
                                "retrieved_chunks": retrieved})
        if getattr(result, "rerank_degraded", False):
            row["detail"]["rerank_degraded"] = True
            logger.warning("%s 精排降级（词重叠兜底），检索数字与正常链路不同", case.get("id"))
        return row

    async def _run_qa_multi(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """multi 档：逐轮喂用户消息（剧本助手轮跳过），最后回答交 LLMJudge。"""
        uid = case_conv_id(case["id"])
        await self._reset_eval_memory(uid)
        started_at = datetime.now().isoformat()
        sent = 0
        last_question = ""
        last_response = ""
        for turn in case.get("turns") or []:
            text = str(turn)
            if text.startswith(_TURN_ASSISTANT_PREFIX):
                continue  # 剧本里的助手话术只是引导，系统自己生成真实回答
            if text.startswith(_TURN_USER_PREFIX):
                text = text[len(_TURN_USER_PREFIX):]
            last_question = text
            resp = await self._chat.handle_chat_turn(text, user_id=uid, conv_id=uid)
            last_response = resp.response
            sent += 1
        return self._mark_run(self._row(case, "qa", passed=None, scores={}, detail={
            "question": last_question,
            "gold_answer": case.get("gold_answer") or "",
            "answer": last_response,
        }), started_at, sent)

    # ── 跑题：多跳集 ───────────────────────────────────────────────────────────

    async def run_multihop(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """多跳集：一题一问走完整 agent 链路；实体链/方向判分在 judge 阶段。"""
        rows: List[Dict[str, Any]] = []
        for case in cases:
            t0 = time.perf_counter()
            uid = case_conv_id(case["id"])
            await self._reset_eval_memory(uid)
            started_at = datetime.now().isoformat()
            resp = await self._chat.handle_chat_turn(case["question"], user_id=uid, conv_id=uid)
            rows.append(self._mark_run(self._row(case, "multihop", passed=None, scores={}, detail={
                "answer": resp.response,
            }), started_at, 1))
            self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
        return rows

    # ── 跑题：长会话集 ─────────────────────────────────────────────────────────

    async def run_long_session(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """长会话集：按剧本逐轮喂 user 轮（一场一个 conv_id），跑完读 Redis summary。

        判分纪律（v2-plan 1.0.9）：只读 summary key，回答里提到不算——
        所以这里存的是 summary 原文，judge 阶段拿它判。
        """
        rows: List[Dict[str, Any]] = []
        for case in cases:
            t0 = time.perf_counter()
            uid = case_conv_id(case["id"])
            await self._reset_eval_memory(uid)
            started_at = datetime.now().isoformat()
            sent = 0
            for turn in case.get("turns") or []:
                if turn.get("speaker") != "user":
                    continue  # 剧本 assistant 轮不喂，系统自己生成回答
                await self._chat.handle_chat_turn(turn["text"], user_id=uid, conv_id=uid)
                sent += 1
            summary = await self._ctx.memory.get_summary(uid, uid)
            rows.append(self._mark_run(self._row(case, "long_session", passed=None, scores={}, detail={
                "summary": summary,
            }), started_at, sent))
            self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
        return rows

    # ── 跑题：急症集 ───────────────────────────────────────────────────────────

    async def run_emergency(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """急症集：确定性判分（升级与否读 ChatResponse 字段，不用 judge）。

        webhook 模拟收件箱（v2-plan 1.3，2026-10-13 用户拍板）：
          - 默认：起本地 mock server，把 EmergencyAgent._webhook_url 指过去
            （类属性 import 时快照 env，不能用「临时设环境变量」，必须显式注入），
            红旗题告警全部落网落盘，绝不外发
          - 容器配了真实 URL 且没 --allow-emergency-webhook：中止（防事故）
          - 显式 --allow：跳过收件箱，按容器配置真发
        """
        if not cases:
            return []

        real_url = os.environ.get("OPENMED_EMERGENCY_WEBHOOK_URL", "").strip()
        if real_url and not self._allow_webhook:
            raise EvalAbortError(
                "检测到 OPENMED_EMERGENCY_WEBHOOK_URL 已设置：急症集红旗题会触发"
                "真实 webhook 通知，评测绝不能外发。确认安全后加 --allow-emergency-webhook 重跑。")

        from agents.agent_orchestrator import EmergencyAgent

        if self._allow_webhook:
            # 显式允许真发：类属性已快照真实 URL，跑题即可
            rows = []
            for case in cases:
                t0 = time.perf_counter()
                rows.append(await self._run_emergency_case(case))
                self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
            return rows

        # 默认：模拟收件箱（收告警落盘 results/<run_id>/emergency_webhooks.jsonl）
        inbox_path = self._results_root / "emergency_webhooks.jsonl"
        with EmergencyWebhookInbox(inbox_path) as inbox:
            EmergencyAgent._webhook_url = f"http://127.0.0.1:{inbox.port}/emergency"
            try:
                rows = []
                for case in cases:
                    t0 = time.perf_counter()
                    rows.append(await self._run_emergency_case(case))
                    self._progress.tick(f"{case['id']} {time.perf_counter() - t0:.1f}s")
            finally:
                EmergencyAgent._webhook_url = None  # 跑完恢复，不残留注入
        self._webhook_payloads = list(inbox.received)
        return rows

    async def _run_emergency_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        uid = case_conv_id(case["id"])
        await self._reset_eval_memory(uid)
        started_at = datetime.now().isoformat()
        resp = await self._chat.handle_chat_turn(case["message"], user_id=uid, conv_id=uid)
        expected = bool(case["expected_escalated"])
        actual = bool(resp.escalated)
        return self._mark_run(self._row(case, "emergency", passed=actual == expected, scores={
            "expected_urgency": case["expected_urgency"],
            "actual_urgency": resp.urgency,
            "actual_escalated": actual,
            "expected_escalated": expected,
        }, detail={
            "expected_escalated": expected, "actual_escalated": actual,
            "expected_urgency": case["expected_urgency"], "actual_urgency": resp.urgency,
        }), started_at, 1)

    # ── 轨迹收集与质量断言 ─────────────────────────────────────────────────────

    @staticmethod
    def _expects_trajectory(row: Dict[str, Any]) -> bool:
        """该行是否预期产轨迹：只有走过 Agent 编排（orchestrator.run）的题才落轨迹。

        不产轨迹的三类（2026-10-13 pilot 实测，之前被误报「轨迹缺失」）：
          - intent / intent_llm_failure：直调 recognize_intent，不进编排
          - qa single：直测 /search 检索链路，也不进编排
        qa multi 与 multihop / long_session / emergency 都走 ChatService → 产轨迹。
        """
        if row["exam"] in ("intent", "intent_llm_failure"):
            return False
        if row["exam"] == "qa" and row.get("detail", {}).get("gold_answer") is None:
            return False  # gold_chunks 有值是 single 档（直测检索）
        return True

    async def _collect_trajectories(self, rows: List[Dict[str, Any]]) -> Dict[str, Trajectory]:
        """按 conv_id 收集「本次运行」的轨迹：跑完立即读会漏（后台写线程），轮询等落盘。

        所有预期产轨迹的行都等到（长会话/急症跑在最后，不等它们会误报缺失）；
        其中 multihop 轨迹缺失在 _assert_quality 里中止整个 run（判分依赖它）。

        同一 conv_id 历次运行的轨迹都在盘上，只取本次运行的（select_run_turns）：
        新跑的行等到 started_at 之后的轨迹条数 ≥ turns_sent（多轮题全部轮次落盘，
        之前只等「有一条」，--cases 重跑多轮题时最后一轮还没写完就开判，会拿
        倒数第二轮当最终回答）；旧 run 的行（重判）按记录的最后一轮往前取。
        返回 conv_id → 本次最后一轮轨迹；全部轮次存 self._conv_turns。
        """
        reader = TrajectoryReader(self._trajectory_dir)
        needed = [r for r in rows if self._expects_trajectory(r)]
        deadline = asyncio.get_event_loop().time() + _TRAJECTORY_WAIT_TIMEOUT_S
        index: Dict[str, Trajectory] = {}
        turns_by_conv: Dict[str, List[Trajectory]] = {}
        pending: List[str] = []

        while True:
            by_conv: Dict[str, List[Trajectory]] = {}
            for traj in reader.load_index().values():
                conv_id = str((traj.meta or {}).get("conv_id") or "")
                if conv_id.startswith(EVAL_USER_PREFIX):
                    by_conv.setdefault(conv_id, []).append(traj)
            for trajs in by_conv.values():
                trajs.sort(key=lambda t: t.timestamp)

            pending = []
            for row in needed:
                conv_id = case_conv_id(row["case_id"])
                started_at = row.get("started_at") or ""
                expected = int(row.get("turns_sent") or 0) or user_turn_count(
                    self._find_case(row["case_id"]))
                turns = select_run_turns(by_conv.get(conv_id, []), started_at=started_at,
                                         anchor_id=row.get("trajectory_id") or "",
                                         expected_turns=expected)
                if turns:
                    index[conv_id] = turns[-1]
                    turns_by_conv[conv_id] = turns
                # 新跑的行要等齐全部轮次；旧行（无 started_at）有就算到
                if not turns or (started_at and len(turns) < expected):
                    pending.append(conv_id)

            if not pending or asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(_TRAJECTORY_WAIT_INTERVAL_S)

        self._conv_turns = turns_by_conv
        if pending:
            logger.warning("轨迹等待超时（%ss），%s 条未齐: %s",
                           _TRAJECTORY_WAIT_TIMEOUT_S, len(pending), sorted(pending))
        return index

    def _assert_quality(self, rows: List[Dict[str, Any]],
                        traj_index: Dict[str, Trajectory]) -> Dict[str, Any]:
        """轨迹数据质量断言。multihop 轨迹违规 → 中止（判分依赖它）；
        其余轨迹违规只记报告（不影响对应判分）。
        """
        report: Dict[str, Any] = {"total": 0, "violations": [], "missing": []}
        for row in rows:
            if not self._expects_trajectory(row):
                continue  # intent / qa single 不产轨迹，不参与质量统计
            conv_id = case_conv_id(row["case_id"])
            traj = traj_index.get(conv_id)
            if traj is None:
                if row["exam"] == "multihop":
                    raise EvalAbortError(
                        f"{row['case_id']} 轨迹未落盘（等待 {_TRAJECTORY_WAIT_TIMEOUT_S}s）"
                        f"——判分无法进行，中止 run")
                report["missing"].append(row["case_id"])
                continue
            report["total"] += 1
            try:
                assert_eval_data_quality(traj)
            except AssertionError as ex:
                if row["exam"] == "multihop":
                    raise EvalAbortError(f"{row['case_id']} 数据质量违规: {ex}")
                report["violations"].append({"case_id": row["case_id"], "reason": str(ex)})
            row["trajectory_id"] = traj.trajectory_id
            row["request_id"] = traj.request_id
        return report

    # ── LLM 判分 ───────────────────────────────────────────────────────────────

    async def _judge_rows(self, rows: List[Dict[str, Any]],
                          traj_index: Dict[str, Trajectory]) -> None:
        """judge 阶段：qa multi / multihop / long_session 并发判分。

        方向 judge 用同一个缓存实例（多跳与安全共享）；
        faithfulness / answer_relevancy 只给 qa multi / multihop（方案 C，
        long_session / emergency 不评——见模块 docstring 判分分工）。
        emergency 升级与否跑题时已确定性判完，judge 阶段无任务。
        """
        from evaluation.evaluator import LLMJudge
        from evaluation.multihop_scorer import SummaryRecallJudge
        from evaluation.safety_scorer import SafetyDirectionJudge

        direction_judge = CachingDirectionJudge(
            SafetyDirectionJudge(self._judge_client, self._judge_model))
        summary_judge = SummaryRecallJudge(self._judge_client, self._judge_model)
        qa_judge = LLMJudge(self._judge_client, self._judge_model)
        semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)

        async def _limited(coro, label):
            async with semaphore:
                result = await coro
            self._progress.tick(label)
            return result

        tasks = []
        for row in rows:
            # 重判路径（run_rejudge）的旧行可能带着上一次的 judge_failed=True，
            # 判分函数只「设 True」不「重置 False」——进判分前先归零，结果
            # 完全由本次判分决定（2026-10-13 重聚合实测：三场 LS 旧兜底标记残留）。
            row["judge_failed"] = False
            if row["exam"] == "qa" and row["detail"].get("gold_answer") is not None:
                tasks.append(_limited(self._judge_qa_multi(row, qa_judge, traj_index), row["case_id"]))
            elif row["exam"] == "multihop":
                tasks.append(_limited(self._judge_multihop(row, traj_index, direction_judge), row["case_id"]))
            elif row["exam"] == "long_session":
                tasks.append(_limited(self._judge_long_session(row, summary_judge), row["case_id"]))
            # safety 判分复用 multihop 行（安全档），放在 multihop 判分里一起做
        if tasks:
            self._progress.start("判分", len(tasks))
            await asyncio.gather(*tasks)

    async def _judge_qa_multi(self, row: Dict[str, Any], qa_judge: Any,
                              traj_index: Dict[str, Trajectory]) -> None:
        """qa multi：LLMJudge 四维打分（主判分）+ 生成质量层（方案 C）。"""
        detail = row["detail"]
        try:
            scores = await qa_judge.judge(detail["question"], detail["answer"],
                                          context=detail["gold_answer"])
            row["scores"] = {
                "relevance": scores.relevance, "accuracy": scores.accuracy,
                "completeness": scores.completeness, "helpfulness": scores.helpfulness,
                "mean": round((scores.relevance + scores.accuracy
                               + scores.completeness + scores.helpfulness) / 4, 4),
            }
            row["passed"] = True
        except Exception as ex:
            logger.warning("%s qa multi 判卷失败: %s", row["case_id"], ex)
            row["judge_failed"] = True
            row["failure_reason"] = f"qa multi 判卷失败: {ex}"

        traj = traj_index.get(case_conv_id(row["case_id"]))
        if traj is None:
            row["scores"]["generation_quality"] = {"unavailable": True}
        else:
            # 检索上下文取整场对话的全部轮次（rev3）：多轮问答常见「前几轮检索、
            # 最后一轮直接用前文作答」，只看最后一轮会把有依据的回答判成 0
            turns = self._conv_turns.get(case_conv_id(row["case_id"])) or [traj]
            await self._judge_generation_quality(row, traj, question=detail["question"],
                                                 context_trajs=turns)

    async def _judge_multihop(self, row: Dict[str, Any], traj_index: Dict[str, Trajectory],
                              direction_judge: CachingDirectionJudge) -> None:
        """多跳判分：实体链/步数/方向（score_multihop）+ 生成质量两指标；
        安全档顺手跑 score_safety_case（方向复用缓存，不重复调 judge）。"""
        from evaluation.multihop_scorer import score_multihop
        from evaluation.safety_scorer import score_safety_case

        case_id = row["case_id"]
        traj = traj_index.get(case_conv_id(case_id))
        case = self._find_case(case_id)
        if traj is None or case is None:
            row["failure_reason"] = "轨迹缺失，无法判分"
            return

        result = await score_multihop(traj, case, direction_judge=direction_judge)
        row["scores"].update(result.to_dict())
        row["passed"] = result.success
        row["failure_reason"] = result.failure_reason
        row["retrieved_chunk_ids"] = result.retrieved_chunk_ids
        row["judge_failed"] = result.judge_failed

        # 安全档：violation 进 headline「安全违规率」（方向缓存命中，零额外调用）
        if case.get("safety"):
            safety = await score_safety_case(
                case.get("question", ""), traj.final_answer() or row["detail"].get("answer", ""),
                case, direction_judge)
            row["scores"]["safety"] = safety.to_dict()
            row["scores"]["safety_rule_found"] = safety.rule_found
            row["scores"]["safety_hard_eligible"] = safety.hard_eligible
            row["scores"]["safety_violation"] = safety.violation
            if safety.judge_failed:
                row["judge_failed"] = True

        # 生成质量层（方案 C，进 report.md 不进 headline）
        await self._judge_generation_quality(row, traj, question=case.get("question", ""))

    async def _judge_long_session(self, row: Dict[str, Any], summary_judge: Any) -> None:
        """长会话：summary 四字段 vs gold_items（确定性 + judge 复核）。

        不评生成质量层（2026-10-13 用户拍板）：长会话考的是记忆留存，
        判分依据是 Redis summary 四字段；最终回答无检索上下文可对照，
        faithfulness 没有区分度（全量实测 10 场 train 9 场 0.0）。
        """
        from evaluation.multihop_scorer import score_long_session

        case = self._find_case(row["case_id"])
        if case is None:
            row["failure_reason"] = "考卷缺失，无法判分"
            return
        result = await score_long_session(row["detail"].get("summary", ""), case,
                                          judge=summary_judge)
        row["scores"].update(result.to_dict())
        row["passed"] = result.recall > 0
        if result.judge_failed:
            row["judge_failed"] = True

    async def _judge_generation_quality(self, row: Dict[str, Any], traj: Trajectory,
                                        question: str,
                                        context_trajs: Optional[List[Trajectory]] = None) -> None:
        """生成质量层（v2-plan 1.2 方案 C，2026-10-12 用户拍板）：
        faithfulness + answer_relevancy 只给 qa multi / 多跳两个集
        （long_session / emergency 不评，2026-10-13 用户拍板），
        结果放 scores.generation_quality，进 report.md 不进 headline。
        回答为空不调 LLM 白花钱。

        traj = 最后一轮轨迹（取最终回答）；context_trajs = 本次对话全部轮次
        （取检索上下文，不传就只用 traj——多跳是单轮题）。2026-10-15 rev3 口径：
          - 急症固定回复（最后一轮路由到 emergency）：不评，与急症集同口径
          - 全部轮次都没检索：faithfulness 不调 judge、记「未检索作答」，
            不进均值单独计数；answer_relevancy 照算（它不依赖检索内容）
        """
        from evaluation.ragas_metrics import judge_answer_relevancy, judge_faithfulness

        case_id = row["case_id"]
        # 急症门控/急症 Agent 给的是固定安全回复（不检索、不生成），评生成质量没有意义
        if (traj.meta or {}).get("primary_agent") == "emergency":
            row["scores"]["generation_quality"] = {"skipped": "急症固定回复"}
            return
        answer = traj.final_answer() or row["detail"].get("answer", "")
        if not answer.strip():
            row["scores"]["generation_quality"] = {"skipped": "回答为空"}
            return
        quality: Dict[str, Any] = {}
        # faithfulness：回答按句切分判「能否从本次对话检索到的内容推断」，
        # 多轮按片段 id 去重（同一片段前后两轮都检索到只算一份）
        contexts: List[str] = []
        seen_keys: set = set()
        for turn in context_trajs or [traj]:
            for item in turn.search_results():
                key = item.id or item.content
                if item.content and key not in seen_keys:
                    seen_keys.add(key)
                    contexts.append(item.content)
        if not contexts:
            # 没检索就作答：不调 judge（没有可对照的资料，判出来恒 0）、不进
            # faithfulness 均值，报告单独计数——「没检索」要教模型该搜就搜，
            # 「检索了没照着说」才是贴着检索回答的问题，混在一个均值里分不清
            quality["faithfulness"] = {"skipped": "未检索作答"}
        else:
            try:
                faith = await judge_faithfulness(question, answer, contexts,
                                                 self._judge_client, self._judge_model)
                quality["faithfulness"] = {
                    "score": faith.score, "supported": faith.supported,
                    "total": faith.total, "judge_failed": faith.judge_failed,
                }
                if faith.judge_failed:
                    row["judge_failed"] = True
            except Exception as ex:
                logger.warning("%s faithfulness 判分失败: %s", case_id, ex)
                quality["faithfulness"] = {"score": 0.5, "judge_failed": True}
                row["judge_failed"] = True
        # answer_relevancy：反向生成 3 问本地 embedding 余弦均值（零 API 成本）
        try:
            relevancy = await judge_answer_relevancy(
                question, answer, self._judge_client, self._judge_model,
                embedder=self._embedder)
            if relevancy is None:
                quality["answer_relevancy"] = {"score": None, "unavailable": True}
            else:
                quality["answer_relevancy"] = {
                    "score": relevancy.score, "judge_failed": relevancy.judge_failed,
                }
                if relevancy.judge_failed:
                    row["judge_failed"] = True
        except Exception as ex:
            logger.warning("%s answer_relevancy 判分失败: %s", case_id, ex)
            quality["answer_relevancy"] = {"score": 0.0, "judge_failed": True}
            row["judge_failed"] = True
        row["scores"]["generation_quality"] = quality

    def _find_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """判分阶段取 case 原文（gold 字段）：从跑题时 _row 存下的考卷快照取。"""
        return self._cases_by_id.get(case_id)

    # ── 行构造与落盘 ───────────────────────────────────────────────────────────

    def _row(self, case: Dict[str, Any], exam: str, passed: Optional[bool],
             scores: Dict[str, Any], detail: Dict[str, Any],
             failure_reason: str = "") -> Dict[str, Any]:
        """cases.jsonl 一行；顺手把 case 存进快照供判分阶段取 gold 字段。"""
        self._cases_by_id[str(case["id"])] = dict(case)
        return {
            "case_id": str(case["id"]),
            "exam": exam,
            "conv_id": case_conv_id(str(case["id"])),
            "split": case.get("split", ""),
            "passed": passed,
            "scores": dict(scores),
            "failure_reason": failure_reason,
            "judge_failed": False,
            "trajectory_id": "",
            "request_id": "",
            "retrieved_chunk_ids": [],
            "detail": detail,
        }

    def _load_existing_rows(self) -> Dict[str, Dict[str, Any]]:
        path = self._results_root / "cases.jsonl"
        if not path.exists():
            return {}
        rows: Dict[str, Dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["case_id"]] = row
        return rows

    def _save_rows(self, rows: List[Dict[str, Any]]) -> None:
        path = self._results_root / "cases.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # ── 聚合与产物 ─────────────────────────────────────────────────────────────

    def _split_rows(self, rows: List[Dict[str, Any]], held_out: bool) -> List[Dict[str, Any]]:
        return [r for r in rows if (r["split"] == "held_out") == held_out]

    def _build_baseline(self, rows: List[Dict[str, Any]],
                        manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        from evaluation.report_v2 import (
            aggregate_dialog, aggregate_emergency,
            aggregate_intent, aggregate_long_session, aggregate_multihop,
            aggregate_safety, build_baseline,
        )

        def _of(exam: str) -> List[Dict[str, Any]]:
            return [r for r in rows if r["exam"] == exam]

        # headline 只算 train（纪律 2：held-out 指标单独列不进 headline）
        train = self._split_rows(rows, held_out=False)

        intent_train = _of_intent(train, "intent")
        forced_train = _of_intent(train, "intent_llm_failure")
        qa_train = [r for r in train if r["exam"] == "qa"]
        mh_train = [r for r in train if r["exam"] == "multihop"]
        ls_train = [r for r in train if r["exam"] == "long_session"]
        em_train = [r for r in train if r["exam"] == "emergency"]

        intent_metrics = aggregate_intent([
            {"case_id": r["case_id"], "expected": r["detail"]["expected"],
             "actual": r["detail"]["actual"]} for r in intent_train])
        voting_correct = sum(1 for r in forced_train if r["scores"].get("voting_correct"))
        routing_correct = sum(1 for r in forced_train if r["scores"].get("routing_correct"))
        forced_total = len(forced_train)
        forced_metrics = {
            "cases": forced_total,
            "voting_accuracy": round(voting_correct / forced_total, 4) if forced_total else 0.0,
            "routing_accuracy": round(routing_correct / forced_total, 4) if forced_total else 0.0,
        }
        if self._skip_model_independent and not forced_total:
            # 本次没跑（LLM 路被短路，结果与被测模型无关）：标记未跑，别让 0.0 被误读
            forced_metrics = {"cases": 0, "voting_accuracy": None, "routing_accuracy": None,
                              "skipped": True}

        single_stats = [r["scores"] for r in qa_train if r.get("detail", {}).get("gold_chunks") is not None]
        multi_scores = [r["scores"]["mean"] for r in qa_train
                        if "mean" in r.get("scores", {})]
        dialog_metrics = aggregate_dialog(
            single_stats, judge_scores=multi_scores or None,
            single_skipped=self._skip_model_independent and not single_stats)

        mh_results = [self._multihop_result(r) for r in mh_train]
        multihop_metrics = aggregate_multihop([m for m in mh_results if m is not None])

        ls_results = [self._long_session_result(r) for r in ls_train]
        long_session_metrics = aggregate_long_session([m for m in ls_results if m is not None])

        safety_results = [self._safety_result(r) for r in mh_train
                          if r.get("scores", {}).get("safety") is not None]
        safety_metrics = aggregate_safety([m for m in safety_results if m is not None])

        emergency_metrics = aggregate_emergency([
            {"case_id": r["case_id"], "expected_escalated": r["detail"]["expected_escalated"],
             "actual_escalated": r["detail"]["actual_escalated"]} for r in em_train])
        # 模拟收件箱收到的告警条数（webhook 链路抽验，与漏检率互相印证）
        emergency_metrics["webhooks_received"] = len(self._webhook_payloads)

        retrieval = {
            # single 档检索统计（MRR / hit@1 / hit@5；没跑时为 {"skipped": True}）
            **dialog_metrics["retrieval"],
            "trajectory_quality": {
                "total": self._quality_report["total"],
                "violations": len(self._quality_report["violations"]),
                "missing": len(self._quality_report["missing"]),
            },
        }

        exam_metrics = {
            "intent": intent_metrics,
            "intent_llm_failure": forced_metrics,
            "dialog": dialog_metrics,
            "multihop": multihop_metrics,
            "long_session": long_session_metrics,
        }

        config = {
            # 被测系统与判卷分开记（2026-10-15）：rev2 及以前的 config.model 记的其实是判卷模型
            "system_model": self._system_model,
            "judge_model": self._judge_model,
            "llm_protocol": os.getenv("OPENMED_LLM_PROTOCOL", "anthropic").strip().lower(),
            "llm_thinking": os.getenv("OPENMED_LLM_THINKING", "").strip().lower(),
            "skip_model_independent": self._skip_model_independent,
            "eval_mode": True,
            "corpus_version": (manifest or {}).get("corpus_version", ""),
            "rules_version": (manifest or {}).get("rules_version", ""),
            "exams": (manifest or {}).get("exams", {}),
        }

        return build_baseline(
            run_id=self.run_id, config=config, exam_metrics=exam_metrics,
            safety=safety_metrics, emergency=emergency_metrics,
            retrieval=retrieval, results=rows,
        )

    @staticmethod
    def _multihop_result(row: Dict[str, Any]):
        from evaluation.multihop_scorer import MultihopResult
        s = row.get("scores", {})
        if "entity_chain_hit" not in s:
            return None  # 轨迹缺失没判成，不进聚合
        return MultihopResult(
            case_id=row["case_id"], expected_steps=s.get("expected_steps", 0),
            actual_steps=s.get("actual_steps", 0), chain_total=0, chain_hit=0,
            entity_chain_hit=s["entity_chain_hit"], ordered=s.get("ordered", False),
            understep=s.get("understep", False), direction=s.get("direction"),
            success=s.get("success", False), judge_failed=row.get("judge_failed", False),
            retrieved_chunk_ids=list(row.get("retrieved_chunk_ids") or []),
            failure_reason=row.get("failure_reason", ""),
        )

    @staticmethod
    def _long_session_result(row: Dict[str, Any]):
        from evaluation.multihop_scorer import LongSessionResult
        s = row.get("scores", {})
        if "recall" not in s:
            return None
        return LongSessionResult(
            case_id=row["case_id"], total=s.get("total", 0), hit=s.get("hit", 0),
            judge_hit=s.get("judge_hit", 0), recall=s.get("recall", 0.0),
            loss_rate=s.get("loss_rate", 0.0), judge_failed=s.get("judge_failed", 0),
            items=list(s.get("items") or []),
        )

    @staticmethod
    def _safety_result(row: Dict[str, Any]):
        from evaluation.safety_scorer import SafetyResult
        s = row.get("scores", {}) or {}
        safety = s.get("safety") or {}
        if not safety:
            return None
        return SafetyResult(
            case_id=row["case_id"], rule_found=s.get("safety_rule_found", False),
            hard_eligible=s.get("safety_hard_eligible", False),
            severity=safety.get("severity", ""),
            expected_direction=safety.get("expected_direction", ""),
            combo_mentioned=safety.get("combo_mentioned", False),
            direction=safety.get("direction", "unclear"),
            violation=s.get("safety_violation", False),
            judge_failed=safety.get("judge_failed", False),
            failure_reason=safety.get("failure_reason", ""),
        )

    def _save_outputs(self, baseline: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
        from evaluation.report_v2 import save_baseline

        # 1. baseline.json（results/<run_id>/ 内 + 复制到 data/eval/baseline.json）
        save_baseline(self._results_root / "baseline.json", baseline)
        data_eval = self._results_root.parent.parent / "baseline.json"
        save_baseline(data_eval, baseline)

        # 2. 轨迹质量报告
        with (self._results_root / "trajectory_quality_report.json").open("w", encoding="utf-8") as f:
            json.dump(self._quality_report, f, ensure_ascii=False, indent=2)

        # 3. 急症收件箱（每个 run 目录自带完整一份：重聚合/部分重跑时收件箱
        #    只是内存里合并过，不落盘的话下一次 --merge 读不到旧告警）
        with (self._results_root / "emergency_webhooks.jsonl").open("w", encoding="utf-8") as f:
            for payload in self._webhook_payloads:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

        # 4. report.md（人读报告）
        (self._results_root / "report.md").write_text(
            self._render_report(baseline, rows), encoding="utf-8")
        logger.info("基线产物已落盘: %s", self._results_root)

    def _render_report(self, baseline: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
        """人读报告：headline 五数字 + 各集明细（含生成质量层）+ 急症告警
        人话摘要 + held-out 单列 + 失败清单。"""
        h = baseline["headline"]
        exams = baseline["exams"]
        cfg = baseline["config"]

        def _num(value: Any) -> str:
            """None = 本次没跑（与被测模型无关的部分被跳过），不写成 0。"""
            return "未跑（与被测模型无关）" if value is None else str(value)

        lines = [
            f"# 基线评测报告 {self.run_id}",
            "",
            f"- 生成时间：{baseline['generated_at']}",
            f"- 被测模型：{cfg.get('system_model', '')}（协议 {cfg.get('llm_protocol', 'anthropic')}，"
            f"思考开关 {cfg.get('llm_thinking') or '默认'}）",
            f"- 判卷模型：{cfg.get('judge_model', '')}",
            f"- 语料：{cfg['corpus_version']}",
            f"- 规则库：{cfg['rules_version']}",
            "",
            "## 五个核心数字（只算 train）",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 单跳 MRR（问答集 single） | {_num(h['single_hop_mrr'])} |",
            f"| 多跳成功率（多跳集） | {h['multihop_success_rate']} |",
            f"| 长会话丢失率（长会话集） | {h['long_session_loss_rate']} |",
            f"| 安全违规率（多跳安全档） | {h['safety_violation_rate']} |",
            f"| 急症漏检率（急症集红旗题） | {h['emergency_miss_rate']} |",
            "",
            "## 各集明细（train）",
            "",
        ]
        train = self._split_rows(rows, held_out=False)
        intent = exams["intent"]
        lines += [
            f"### 意图集（{intent['total']} 题）",
            f"- 准确率：{intent['accuracy']}　macro-F1：{intent['macro_f1']}",
            "",
        ]
        if intent.get("per_class"):
            lines += ["| 意图 | 期望 | 实测 | 正确 | precision | recall |", "| --- | --- | --- | --- | --- | --- |"]
            for label, sc in sorted(intent["per_class"].items()):
                lines.append(f"| {label} | {sc['expected']} | {sc['actual']} | {sc['correct']} "
                             f"| {sc['precision']} | {sc['recall']} |")
            lines.append("")
        forced = exams["intent_llm_failure"]
        if forced.get("skipped"):
            lines += ["### LLM 失败降级子集", "- 本次未跑（LLM 路被短路，结果与被测模型无关）", ""]
        else:
            lines += [
                f"### LLM 失败降级子集（{forced['cases']} 题）",
                f"- 降级投票意图准确率：{forced['voting_accuracy']}",
                f"- 降级路由准确率：{forced['routing_accuracy']}",
                "",
            ]
        single = exams["dialog"]["retrieval"]
        lines.append("### 问答集")
        if single.get("skipped"):
            lines.append("- single 档检索：本次未跑（直测检索链路，不经过被测模型）")
        else:
            lines.append(f"- single 档检索（{single['cases']} 题）：MRR {single['mrr']}　"
                         f"hit@1 {single['hit_at_1']}　hit@5 {single['hit_at_5']}")
        if "avg_judge_scores" in exams["dialog"]:
            lines.append(f"- multi 档判卷均分（0-1 四维均值）：{exams['dialog']['avg_judge_scores']}")
        # 生成质量层只覆盖 multi 档（single 直测检索链路，没有最终回答）
        lines += self._generation_quality_lines(
            [r for r in train if r["exam"] == "qa" and r["detail"].get("gold_answer") is not None])
        lines += ["", f"### 多跳集（{exams['multihop']['cases']} 题）"]
        mh = exams["multihop"]
        lines += [
            f"- 成功率：{mh['success_rate']}　实体链命中率：{mh['entity_chain_hit_rate']}",
            f"- 有序链率：{mh['ordered_chain_rate']}　步数效率：{mh['step_efficiency']}",
        ]
        lines += self._generation_quality_lines([r for r in train if r["exam"] == "multihop"])
        lines += [
            "",
        ]
        safety = baseline["safety"]
        lines += [
            "#### 安全档（多跳安全题）",
            f"- 违规率：{safety['overall_violation_rate']}（分母 {safety['denominator']}，"
            f"违规 {safety['violations']}）　未覆盖率：{safety['uncovered_rate']}",
        ]
        for sev, sc in sorted((safety.get("by_severity") or {}).items()):
            lines.append(f"- {sev}：{sc['violations']}/{sc['cases']} 违规")
        lines += [
            "",
            f"### 长会话集（{exams['long_session']['cases']} 场）",
            f"- 关键节点存活率：{exams['long_session']['recall']}　丢失率：{exams['long_session']['loss_rate']}",
        ]
        lines += [
            "",
            "### 急症集",
        ]
        em = baseline["emergency"]
        lines += [
            f"- 漏检率：{em['missed_escalation_rate']}（红旗 {em['red_flag_cases']} 题）",
            f"- 误报率：{em['false_escalation_rate']}（阴性 {em['negative_cases']} 题）",
            f"- 模拟收件箱收到告警：{em.get('webhooks_received', 0)} 条",
        ]
        lines += [
            "",
            "## 急症告警（模拟收件箱）",
            "",
        ]
        if self._webhook_payloads:
            for payload in self._webhook_payloads:
                lines.append(f"- {render_webhook_summary(payload)}")
        else:
            lines.append("- 无告警（红旗题全漏检，或本 run 未含急症题）")
        lines += [
            "",
            "## held-out 单列（不进 headline）",
            "",
        ]
        held = self._split_rows(rows, held_out=True)
        if held:
            for exam_name, label in (("intent", "意图集"), ("qa", "问答集"),
                                     ("multihop", "多跳集"), ("long_session", "长会话集"),
                                     ("emergency", "急症集"), ("intent_llm_failure", "LLM 失败子集")):
                sub = [r for r in held if r["exam"] == exam_name]
                judged = [r for r in sub if r["passed"] is not None]
                if sub:
                    passed = sum(1 for r in judged if r["passed"])
                    lines.append(f"- {label}：{len(sub)} 题（held_out），判分完成 {len(judged)}，"
                                 f"通过 {passed}")
        else:
            lines.append("- 无 held-out 题（pilot 或 --cases 部分跑）")
        lines += [
            "",
            "## 轨迹质量",
            "",
        ]
        tq = baseline["retrieval"]["trajectory_quality"]
        lines += [
            f"- 轨迹总数：{tq['total']}　违规：{tq['violations']}　缺失：{tq['missing']}",
            "",
            "## 失败清单",
            "",
        ]
        failed = [r for r in rows if r.get("failure_reason")]
        if failed:
            for r in failed:
                lines.append(f"- {r['case_id']}（{r['exam']}）：{r['failure_reason']}")
        else:
            lines.append("- 无失败记录")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _generation_quality_lines(exam_rows: List[Dict[str, Any]]) -> List[str]:
        """某集生成质量层（方案 C）的 report 行：faithfulness 与
        answer_relevancy 各一行均值，没有可算数字就不打印。

        rev3 口径的两类跳过单独计数、不进均值：急症固定回复（整题不评）、
        未检索作答（只跳 faithfulness，answer_relevancy 照算）。"""
        faiths: List[float] = []
        rels: List[float] = []
        no_retrieval = 0
        emergency = 0
        for r in exam_rows:
            quality = (r.get("scores") or {}).get("generation_quality") or {}
            if quality.get("skipped") == "急症固定回复":
                emergency += 1
                continue
            faith = quality.get("faithfulness") or {}
            if faith.get("skipped") == "未检索作答":
                no_retrieval += 1
            elif faith.get("score") is not None:
                faiths.append(faith["score"])
            relv = quality.get("answer_relevancy") or {}
            if relv.get("score") is not None:
                rels.append(relv["score"])
        lines: List[str] = []
        if faiths:
            lines.append(f"- 生成质量 faithfulness 均值：{round(sum(faiths) / len(faiths), 4)}"
                         f"（{len(faiths)} 题）")
        if no_retrieval:
            lines.append(f"- 整场未检索就作答：{no_retrieval} 题（不进 faithfulness 均值）")
        if rels:
            lines.append(f"- 生成质量 answer_relevancy 均值：{round(sum(rels) / len(rels), 4)}"
                         f"（{len(rels)} 题）")
        if emergency:
            lines.append(f"- 急症固定回复：{emergency} 题（不评生成质量）")
        return lines


def _of_intent(rows: List[Dict[str, Any]], exam: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["exam"] == exam]
