"""
v2 阶段 1 第 3 步：考卷生成流水线。

五个子命令（与用户确认的多 agent 分工对应）：
  intent        —— 主 agent 跑：474 条旧素材导入 + 1/3 LLM 改写措辞 + 补生成
                   ~30 条扩展问法 + llm_failure 子集标记（唯一调 DeepSeek 的子命令）
  qa            —— 子 agent A 写 drafts 后组装：问答集 draft → 确定性组装
  multihop      —— 子 agent A 写 drafts 后组装：多跳集 draft → 确定性组装
  long_session  —— 子 agent A 写 drafts 后组装：长会话集 draft → 确定性组装
  manifest      —— 四套定稿后跑：确定性生成 MANIFEST.json + held_out/held_out_ids.txt
                   （数字全部从考卷/语料/规则库实测统计，不写死——手写必漂移）

铁律（与用户确认）：
  1. 语料有据：gold_chunks/entity_chain/safety 必须从语料与规则库实录复制，
     本流水线不代写内容字段——它只做「导入/改写/扩展」这类 LLM 生成 + 确定性
     组装（id 规范化、split 分配、字面重叠检查），内容由出题 agent 对着语料写；
  2. split 分配统一走 assign_split()（id 的 SHA256 哈希 ~25% 落 held_out），
     固化进 JSON 后不再变（held-out 是评测隔离底线，杜绝事后挪动）；
  3. 组装完成后自动调 scripts/validate_exams.py 做全量确定性断言，
     任一 error 即失败退出——考卷写坏在这里拦下，不进人工审核环节。

断点续跑：intent 子命令的 LLM 产物按批缓存到 drafts/（默认
data/knowledge/_archive/exam_drafts/），重跑时已有缓存且未给 --force 则跳过；
qa/multihop/long_session 的组装是纯确定性操作，天然幂等。

用法（在项目根目录）：
    .venv/Scripts/python.exe scripts/build_exams.py intent
    .venv/Scripts/python.exe scripts/build_exams.py qa
    .venv/Scripts/python.exe scripts/build_exams.py multihop
    .venv/Scripts/python.exe scripts/build_exams.py long_session
    .venv/Scripts/python.exe scripts/build_exams.py qa --draft <其他路径>
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.llm_utils import create_llm_client, extract_text_content  # noqa: E402

LEGACY_FILE = ROOT / "data" / "knowledge" / "_archive" / "medical_eval_dataset.json"
CORPUS_FILE = ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"
DRAFTS_DIR = ROOT / "data" / "knowledge" / "_archive" / "exam_drafts"
EXAMS_DIR = ROOT / "data" / "eval" / "exams"
CHUNK_INDEX_FILE = ROOT / "data" / "eval" / "chunk_index.json"

# 五套正式考卷文件名（与 exam_loader 的固定约定一致）
EXAM_FILES = {
    "intent": "intent_v1.json",
    "qa": "qa_v1.json",
    "multihop": "multihop_v1.json",
    "long_session": "long_session_v1.json",
    "emergency": "emergency_v1.json",
}

# 意图集规模设计（实测 legacy 474 条，19 类，每类 10-57 条，无真空弱类）：
#   archive+rewritten ~130（每类按 27% 比例抽，最少 3 条）+ extended ~28 = ~158，
#   落在「精选 150-200」区间内；extended 每类 1 条 + legacy<15 的 7 类各 +1 +
#   易混组（medication_interaction / symptom_check）各 +1。
ARCHIVE_TARGET = 130
REWRITE_RATIO = 1 / 3          # 抽中素材里 1/3 走 LLM 改写措辞（source=rewritten）
LLM_FAILURE_TARGET = 35        # llm_failure 子集条数（分层抽，第 6 步 forced-fail 跑这批）
MIN_PER_CLASS = 3

# 每类扩展问法条数：默认 1 条/类；弱类（legacy 计数 <15）与易混组多给 1 条。
# 依据：扩展目的是措辞多样性覆盖 + 混淆边界密度（appointment 四兄弟、
# medication 三兄弟之间的混淆是意图识别的主要错误源，多给边界样本）。
EXTENDED_EXTRA_CLASSES = {"complaint", "feedback", "query", "medical_receipt", "request",
                          "other", "human_handoff", "medication_interaction", "symptom_check"}

# 19 类语义说明（扩展生成时贴给 LLM 的类别定义，与 IntentCategory 枚举注释一致）
INTENT_CLASS_DESC = {
    "query": "通用医疗咨询（泛泛的医学问题，如『XX病是什么』）",
    "complaint": "投诉不满（对服务/医生/流程的不满表达）",
    "request": "请求操作（要求执行某个动作，如『帮我查一下』）",
    "greeting": "问候（寒暄、开场白）",
    "emergency": "急症/紧急情况（剧烈疼痛、大出血、意识障碍等危重描述）",
    "medication": "用药咨询（某药是治什么的、能不能吃某药）",
    "appointment": "预约挂号（要挂某个科室/医生的号）",
    "feedback": "正面反馈（感谢、表扬）",
    "symptom_check": "症状自查/描述症状（叙述身体不适，想知道怎么回事）",
    "department_guide": "科室指引（不知道挂什么科，求科室建议）",
    "appointment_manage": "查询/修改预约（查看已挂号信息）",
    "appointment_reschedule": "预约改期（更换就诊时间）",
    "appointment_cancel": "取消/退号",
    "medical_receipt": "就诊证明/医疗票据（开证明、打印票据）",
    "appointment_payment_issue": "挂号费支付异常（支付失败、重复扣款）",
    "medication_dosage": "用法用量咨询（吃多少、怎么吃、漏服怎么办）",
    "medication_interaction": "药物相互作用/不良反应（两种药能否同服、副作用）",
    "human_handoff": "转人工/转诊（要求人工客服或转诊）",
    "other": "其他无法归类的输入",
}


# ── 通用工具 ───────────────────────────────────────────────────────────────────

def assign_split(case_id: str) -> str:
    """按 id 的 SHA256 哈希分配 split：~25% 落 held_out。

    哈希分配而不是随机数：同一 id 永远落同一侧（重跑流水线不挪动分区），
    且分布对生成顺序无关（不因「先写的一批都是 train」造成系统性偏差）。
    """
    digest = hashlib.sha256(case_id.encode()).hexdigest()
    return "held_out" if int(digest[:8], 16) % 4 == 0 else "train"


def sample_per_class(items: list, ratio: float, min_per: int) -> list:
    """按 expected_intent 分层抽样：每类取 ceil(类数 × ratio)，至少 min_per 条。

    用哈希打散而不是 random.sample——确定性（可复现）且无状态，
    重跑结果一致，审核清单的题目编号不会漂移。
    """
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for it in items:
        buckets[it["expected_intent"]].append(it)
    picked = []
    for cls, bucket in sorted(buckets.items()):
        n = max(min_per, int(len(bucket) * ratio + 0.999))  # ceil 语义
        bucket_sorted = sorted(bucket, key=lambda it: hashlib.sha256(it["message"].encode()).hexdigest())
        picked += bucket_sorted[:n]
    return picked


def load_legacy() -> List[dict]:
    """旧素材 474 条意图样例（{message, expected_intent}，19 类标签已实测与枚举同构）。"""
    data = json.loads(LEGACY_FILE.read_text(encoding="utf-8"))
    return data["intent_cases"]


def load_corpus_by_id() -> dict:
    """多跳语料按 doc_id 索引（组装 multihop 时查实体链实录）。"""
    corpus = {}
    for line in CORPUS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            corpus[row["doc_id"]] = row
    return corpus


# ── intent 子命令：导入 + LLM 改写 + LLM 扩展 ──────────────────────────────────

INTENT_SYSTEM_PROMPT = """你是医疗问诊系统意图识别评测集的构建助手。输出严格 JSON（数组），不输出任何 Markdown 代码块或解释文字。

改写铁律：
1. 只改措辞不改意图——输入的 expected_intent 是答案，改完后意图类别必须仍与它一致；
2. 改写后仍像真实用户打字：口语化、简短（5-40 字）、可以有口语连接词与错别字；
3. 至少一半用词或句式与原文不同（同义替换/换语序/换称呼），禁止只改标点。

扩展铁律：
1. 每类生成指定条数的全新消息，不得与输入样例重复；
2. 消息必须能被唯一归入该类别（避免与相近类别混淆的模糊表述）；
3. 口吻与真实问诊对话一致，5-40 字。"""


def parse_json_array(text: str):
    """把 LLM 输出解析成 JSON 数组。失败返回 (None, 错误信息) 供重试。"""
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}（输出前 80 字: {text[:80]}...）"
    if not isinstance(data, list):
        return None, f"期望 JSON 数组，得到 {type(data).__name__}"
    return data, None


async def generate_batch(client, model: str, task: str, max_retries: int = 2) -> list:
    """一次 LLM 批生成：请求 → 解析 → 失败带错误重试（与 build_multihop_corpus 同款）。"""
    messages = [{"role": "user", "content": task}]
    for attempt in range(max_retries + 1):
        resp = await client.messages.create(
            model=model, max_tokens=8000, temperature=0.7,
            system=INTENT_SYSTEM_PROMPT, messages=messages,
        )
        text = extract_text_content(resp.content)
        data, err = parse_json_array(text)
        if err is None:
            return data
        messages.append({"role": "assistant", "content": text[:2000]})
        messages.append({"role": "user", "content": f"解析失败：{err}。请重新只输出 JSON 数组。"})
        print(f"  retry {attempt + 1}/{max_retries}: {err}", flush=True)
    raise RuntimeError(f"LLM 批生成失败（重试 {max_retries} 次后仍无法解析）")


async def rewrite_items(client, model: str, items: list, cache_file: Path, force: bool) -> List[dict]:
    """1/3 改写：把抽中的素材按 15 条/批交给 LLM 改写措辞（意图类别不变）。

    断点续跑：每批结果立即追加缓存文件，重跑时按 message 跳过已改写的。
    """
    done: dict = {}
    if cache_file.exists() and not force:
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["orig_message"]] = row["message"]

    todo = [it for it in items if it["message"] not in done]
    if not todo:
        print(f"rewrite: all {len(items)} cached, skip", flush=True)
    with cache_file.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), 15):
            batch = todo[i:i + 15]
            task = "改写任务：逐条改写下列用户消息的措辞，输出 JSON 数组 [{id, message}]（id 用输入的序号，message 是改写后的文本）。\n" + "\n".join(
                f"{j}: {it['message']}（意图: {it['expected_intent']}）" for j, it in enumerate(batch))
            print(f"rewrite batch {i // 15 + 1}/{(len(todo) - 1) // 15 + 1} ({len(batch)} 条)...", flush=True)
            rows = await generate_batch(client, model, task)
            for row in rows:
                idx = int(row.get("id", -1))
                if 0 <= idx < len(batch):
                    it = batch[idx]
                    line = json.dumps({"orig_message": it["message"], "message": row["message"]}, ensure_ascii=False)
                    f.write(line + "\n")
                    done[it["message"]] = row["message"]
                    print(f"  [{it['expected_intent']}] {it['message'][:20]}... -> {row['message'][:20]}...", flush=True)
    return apply_rewrites(items, done)


def apply_rewrites(items: list, done: dict) -> list:
    """把改写缓存应用到素材列表：命中缓存的取改写文本，未命中的保留原文。

    纯函数（无 I/O）——单独可测。教训：初版在这里「source 判断查了 done、
    message 取值却忘了查 done」，47 条 rewritten 全是 legacy 原文（子 agent B
    逐条比对抓到）；validate_exams 的 rewritten 断言是第二道防线。
    """
    return [{"message": done.get(it["message"], it["message"]),
             "expected_intent": it["expected_intent"],
             "source": "rewritten" if it["message"] in done else "archive"}
            for it in items]


async def extend_classes(client, model: str, plan: dict, cache_file: Path, force: bool) -> List[dict]:
    """补生成扩展问法：plan = {类名: 条数}。断点续跑按类跳过已生成。"""
    done: dict = {}
    if cache_file.exists() and not force:
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.setdefault(row["intent"], []).append(row["message"])

    out: List[dict] = []
    for cls, n in sorted(plan.items()):
        have = done.get(cls, [])
        if len(have) >= n:
            print(f"extend {cls}: cached {len(have)}/{n}, skip", flush=True)
            out += [{"message": m, "expected_intent": cls} for m in have[:n]]
            continue
        need = n - len(have)
        desc = INTENT_CLASS_DESC[cls]
        task = (f"扩展任务：为意图类别「{cls}」（{desc}）生成 {need} 条全新的用户消息，"
                f"输出 JSON 数组 [{{intent, message}}]。不得与下列已有样例重复：\n" +
                "\n".join(f"- {m}" for m in have))
        print(f"extend {cls}: generating {need}...", flush=True)
        rows = await generate_batch(client, model, task)
        new_msgs = [r["message"] for r in rows if r.get("intent") == cls][:need]
        with cache_file.open("a", encoding="utf-8") as f:
            for m in new_msgs:
                f.write(json.dumps({"intent": cls, "message": m}, ensure_ascii=False) + "\n")
        have += new_msgs
        out += [{"message": m, "expected_intent": cls} for m in have[:n]]
    return out


def pick_llm_failure(cases: list, target: int) -> set:
    """从全部意图题里按类分层抽 target 条标记 llm_failure（第 6 步 forced-fail 跑这批）。

    分层依据：意图识别的主要错误源是混淆边界，llm_failure 子集要覆盖每类
    至少 1 条（弱类全量），其余按类占比分配——子集测的是「LLM 路挂了之后
    降级投票/路由还准不准」，覆盖均匀比数量堆大类更重要。
    """
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for c in cases:
        buckets[c["expected_intent"]].append(c)
    picked: set = set()
    # 第一轮：每类 1 条（哈希打散取第一条）
    for cls, bucket in buckets.items():
        first = sorted(bucket, key=lambda c: hashlib.sha256(c["message"].encode()).hexdigest())[0]
        picked.add(first["message"])
    # 第二轮：剩余名额按类占比分配
    remaining = target - len(picked)
    for cls, bucket in buckets.items():
        rest = [c for c in bucket if c["message"] not in picked]
        n = int(len(bucket) / len(cases) * remaining)
        picked.update(c["message"] for c in rest[:n])
    # 兜底：仍不足就继续按哈希顺序补
    if len(picked) < target:
        all_rest = [c for c in cases if c["message"] not in picked]
        all_rest.sort(key=lambda c: hashlib.sha256(c["message"].encode()).hexdigest())
        picked.update(c["message"] for c in all_rest[: target - len(picked)])
    return picked


async def run_intent(client, model: str) -> None:
    """意图集全流程：导入抽样 → 1/3 改写 → 扩展 → llm_failure 标记 → 写 JSON → 校验。"""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    rewrite_cache = DRAFTS_DIR / "intent_rewrite_cache.jsonl"
    extend_cache = DRAFTS_DIR / "intent_extend_cache.jsonl"

    legacy = load_legacy()
    picked = sample_per_class(legacy, ARCHIVE_TARGET / len(legacy), MIN_PER_CLASS)
    print(f"legacy {len(legacy)} -> sampled {len(picked)}", flush=True)

    # 1/3 改写：按哈希打散抽（与分层抽样同款确定性，重跑一致）
    picked_sorted = sorted(picked, key=lambda it: hashlib.sha256(it["message"].encode()).hexdigest())
    n_rewrite = int(len(picked) * REWRITE_RATIO)
    rewrite_todo = picked_sorted[:n_rewrite]
    keep = picked_sorted[n_rewrite:]
    rewritten = await rewrite_items(client, model, rewrite_todo, rewrite_cache, force=False)
    print(f"rewritten {len(rewrite_todo)} / keep-as-archive {len(keep)}", flush=True)

    # 扩展计划：每类 1 条 + EXTENDED_EXTRA_CLASSES 各 +1
    plan = {cls: 1 + (1 if cls in EXTENDED_EXTRA_CLASSES else 0) for cls in INTENT_CLASS_DESC}
    extended = await extend_classes(client, model, plan, extend_cache, force=False)

    # 组装 cases：archive（未改写）+ rewritten + extended
    cases = []
    for it in keep:
        cases.append({"message": it["message"], "expected_intent": it["expected_intent"], "source": "archive"})
    cases += [{"message": r["message"], "expected_intent": r["expected_intent"], "source": r["source"]}
              for r in rewritten]
    cases += [{"message": e["message"], "expected_intent": e["expected_intent"], "source": "extended"} for e in extended]

    # llm_failure 子集标记（分层抽 35 条）
    failure_msgs = pick_llm_failure(cases, LLM_FAILURE_TARGET)

    # id 与 split：id 按类内哈希顺序编号（确定性），split 由 id 哈希分配
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for c in cases:
        buckets[c["expected_intent"]].append(c)
    final: List[dict] = []
    n = 0
    for cls in sorted(buckets):
        for c in sorted(buckets[cls], key=lambda c: hashlib.sha256(c["message"].encode()).hexdigest()):
            n += 1
            case_id = f"INT-{n:03d}"
            final.append({
                "id": case_id,
                "message": c["message"],
                "expected_intent": c["expected_intent"],
                "source": c["source"],
                "split": assign_split(case_id),
                "llm_failure": c["message"] in failure_msgs,
            })
    write_and_validate("intent", final)
    n_held = sum(1 for c in final if c["split"] == "held_out")
    print(f"intent exam: {len(final)} cases ({n_held} held_out, {len(failure_msgs)} llm_failure)", flush=True)


# ── qa / multihop / long_session 子命令：draft 组装 ───────────────────────────

def normalize_draft_cases(drafts: List[dict], exam_name: str) -> List[dict]:
    """确定性组装：id 缺省时补前缀+序号；split 一律重分配（draft 里即使写了也忽略）。

    split 必须在流水线统一分配（铁律 2）：出题 agent 不接触分区逻辑，
    杜绝「出题时随手指定 train」造成 held-out 系统性偏移。
    """
    # emergency 用 ESC 前缀：EMR 前缀哈希分配只落 4/30 held_out（13.3% < 20% 铁律），
    # 改名重摇哈希分区（ESC = escalation，27% 最接近 25% 期望）——先例见 intent 集
    # 的「改名重摇哈希分区（held_out 24→30）」
    prefix = {"qa": "QA", "multihop": "MH", "long_session": "LS", "emergency": "ESC"}[exam_name]
    cases = []
    for i, d in enumerate(drafts):
        case_id = d.get("id") or f"{prefix}-{i + 1:03d}"
        case = dict(d)
        case["id"] = case_id
        case["split"] = assign_split(case_id)
        cases.append(case)
    return cases


def write_and_validate(exam_name: str, cases: List[dict]) -> None:
    """写正式考卷 JSON + 跑 validate_exams.py 全量断言（任一 error 即失败退出）。"""
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXAMS_DIR / EXAM_FILES[exam_name]
    out_file.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {out_file.name}", flush=True)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_exams.py")],
        capture_output=True, text=True, encoding="utf-8",
    )
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)  # validate 自身崩了时 stderr 有 traceback（stdout 为空）
    if result.returncode != 0:
        print("ERROR: validate_exams 未通过，考卷未定稿（报告见 _archive/exam_validate_report.txt）", flush=True)
        sys.exit(1)


def run_assemble(exam_name: str, args) -> None:
    """读 draft JSON → 组装 → 写正式考卷。draft 路径默认 _archive/exam_drafts/<exam>_draft.json。"""
    draft_file = Path(args.draft) if args.draft else DRAFTS_DIR / f"{exam_name}_draft.json"
    if not draft_file.exists():
        print(f"ERROR: draft 不存在: {draft_file}", flush=True)
        sys.exit(1)
    drafts = json.loads(draft_file.read_text(encoding="utf-8"))
    if not isinstance(drafts, list) or not drafts:
        print(f"ERROR: draft 必须是非空 JSON 数组: {draft_file}", flush=True)
        sys.exit(1)
    print(f"assembling {exam_name} from {draft_file} ({len(drafts)} drafts)", flush=True)
    cases = normalize_draft_cases(drafts, exam_name)
    write_and_validate(exam_name, cases)


def run_manifest() -> None:
    """五套定稿后生成 MANIFEST.json + held_out_ids.txt（数字全实测，不写死）。

    MANIFEST 记录「这批题是在哪版语料/规则上生成的」——基线可复现性的根；
    held_out_ids.txt 与各考卷 JSON 内 split=="held_out" 互为双保险，
    阶段 2.1 捞取失败样本时硬编码拒绝这些 id（测试锁定）。
    """
    from datetime import date

    from core.drug_rules import load_rules
    from evaluation.exam_loader import load_all_exams

    loaded = load_all_exams(EXAMS_DIR)  # 五套全量校验，任一失败抛 ValueError
    rules = load_rules(ROOT / "data" / "eval" / "rules" / "drug_interactions_v1.json")
    chunk_index = json.loads(CHUNK_INDEX_FILE.read_text(encoding="utf-8"))
    corpus_rows = [l for l in CORPUS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    hospital_files = [f for f in sorted((ROOT / "data" / "knowledge" / "hospital").glob("*.json"))]

    manifest = {
        "schema_version": "1",
        "generated_at": date.today().isoformat(),
        "corpus_version": (
            f"{CORPUS_FILE.name} {len(corpus_rows)} 篇（切片 {chunk_index['total_chunks']} 片段）"
            f"+ hospital {len(hospital_files)} 文件"
        ),
        "rules_version": f"drug_interactions_v1.json {len(rules)} 组",
        "exams": {
            name: {
                "file": EXAM_FILES[name],
                "count": len(cases),
                "held_out": sum(1 for c in cases if c.split == "held_out"),
            }
            for name, cases in loaded.items()
        },
    }
    (EXAMS_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # held-out 审计快照：每行「套名 id」，注释头说明用途
    held_dir = ROOT / "data" / "eval" / "held_out"
    held_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OpenMed v2 考卷 held-out 审计快照（build_exams.py manifest 子命令生成，勿手改）",
        "# 与各考卷 JSON 内 split==\"held_out\" 字段互为双保险；",
        "# 阶段 2.1 捞取失败样本时硬编码拒绝这些 id（held-out 是评测隔离底线）。",
        f"# generated_at: {manifest['generated_at']}",
    ]
    for name, cases in loaded.items():
        for c in cases:
            if c.split == "held_out":
                lines.append(f"{name} {c.id}")
    (held_dir / "held_out_ids.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_held = sum(sum(1 for c in cases if c.split == "held_out") for cases in loaded.values())
    print(f"manifest -> MANIFEST.json（{sum(len(c) for c in loaded.values())} 题 / {n_held} held_out）"
          f"+ held_out/held_out_ids.txt", flush=True)


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()  # 读项目根 .env（intent 子命令调 DeepSeek 用）
    parser = argparse.ArgumentParser(description="v2 阶段 1 第 3 步：考卷生成流水线")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("intent", help="意图集：导入 474 + 1/3 改写 + 扩展 + llm_failure 标记（调 LLM）")
    sub.add_parser("manifest", help="五套定稿后：生成 MANIFEST.json + held_out/held_out_ids.txt（不调 LLM）")
    for name in ("qa", "multihop", "long_session", "emergency"):
        p = sub.add_parser(name, help=f"{name} 集：读 draft → 确定性组装（不调 LLM）")
        p.add_argument("--draft", help="draft JSON 路径（默认 _archive/exam_drafts/<name>_draft.json）")
    args = parser.parse_args()

    if args.command == "manifest":
        run_manifest()
        return
    if args.command in ("qa", "multihop", "long_session", "emergency"):
        run_assemble(args.command, args)
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env", flush=True)
        sys.exit(1)
    # 0.5 的双协议工厂：DeepSeek 走 anthropic 协议零变化
    client = create_llm_client(api_key=api_key, base_url=os.getenv("ANTHROPIC_BASE_URL"))
    model = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")
    asyncio.run(run_intent(client, model))


if __name__ == "__main__":
    main()
