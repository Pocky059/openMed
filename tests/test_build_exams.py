"""build_exams.py 核心函数的单测（阶段 1 第 3 步生成流水线）。

锁定四条纪律：
  1. split 分配确定性 + ~25% 落 held_out（分区纪律，id 哈希当场分配）；
  2. 分层抽样每类保底 + 按比例（意图集覆盖全枚举）；
  3. llm_failure 子集每类至少 1 条（第 6 步 forced-fail 跑批的覆盖要求）；
  4. 字面重叠检查对「照抄语料」敏感、对「正常措辞」宽容（防字面检索命中）。

全部离线：只 import scripts/build_exams 的函数，不触发 LLM 客户端创建。
"""
import hashlib

from scripts.build_exams import (
    apply_rewrites,
    assign_split,
    normalize_draft_cases,
    pick_llm_failure,
    sample_per_class,
)
from scripts.validate_exams import _literal_overlap


def test_apply_rewrites_uses_rewritten_text():
    """回归锁（子 agent B 抓到的 bug）：命中改写缓存必须取改写文本，不是原文。

    初版实现「source 判断查了 done、message 取值却忘了查 done」，
    47 条 rewritten 全是 legacy 原文——这里锁住 done.get 的行为。
    """
    items = [
        {"message": "原始问法A", "expected_intent": "greeting"},
        {"message": "原始问法B", "expected_intent": "query"},
    ]
    done = {"原始问法A": "改写后问法A"}   # B 未改写
    out = apply_rewrites(items, done)
    assert out[0]["message"] == "改写后问法A" and out[0]["source"] == "rewritten"
    assert out[1]["message"] == "原始问法B" and out[1]["source"] == "archive"


def test_assign_split_deterministic():
    """同一 id 的 split 永远一致（重跑流水线不挪动分区）。"""
    assert assign_split("MH-001") == assign_split("MH-001")
    assert assign_split("QA-007") == assign_split("QA-007")


def test_assign_split_roughly_25_percent():
    """1000 个固定 id 的 held_out 占比落在 25%±5%（哈希分配，样本固定不 flaky）。"""
    held = sum(1 for i in range(1, 1001) if assign_split(f"MH-{i:03d}") == "held_out")
    assert 200 <= held <= 300, f"held_out {held}/1000 偏离 25% 太多"


def test_sample_per_class_min_and_ratio():
    """每类至少 min_per 条；总量接近按比例抽样；确定性（两次结果相同）。"""
    items = []
    for cls in ("a", "b", "c", "d"):
        for i in range(100):
            items.append({"message": f"{cls}-msg-{i}", "expected_intent": cls})
    picked = sample_per_class(items, ratio=0.3, min_per=3)
    from collections import Counter

    dist = Counter(p["expected_intent"] for p in picked)
    assert all(v >= 3 for v in dist.values()), f"存在低于保底的类: {dict(dist)}"
    assert sum(dist.values()) == 120, f"4 类 × 100 × 30% 应抽 120 条，实际 {sum(dist.values())}"
    # 确定性：哈希打散排序，重跑结果一致
    assert [p["message"] for p in picked] == [
        p["message"] for p in sample_per_class(items, ratio=0.3, min_per=3)]


def test_pick_llm_failure_every_class_covered():
    """llm_failure 子集：数量达标 + 每类至少 1 条。"""
    cases = []
    for cls in ("a", "b", "c", "d", "e"):
        for i in range(20):
            cases.append({"message": f"{cls}-{i}", "expected_intent": cls})
    picked = pick_llm_failure(cases, target=35)
    assert len(picked) == 35
    from collections import defaultdict

    by_cls = defaultdict(int)
    for c in cases:
        if c["message"] in picked:
            by_cls[c["expected_intent"]] += 1
    assert set(by_cls) == {"a", "b", "c", "d", "e"}, f"存在未覆盖的类: {dict(by_cls)}"


def test_literal_overlap_verbatim_vs_unrelated():
    """照抄语料 = 1.0；完全无关 = 0（5 字窗口对连续照抄敏感）。"""
    contents = ["银杏叶提取物与华法林合用时可能增加出血风险，建议加强 INR 监测。"]
    assert _literal_overlap("银杏叶提取物与华法林合用时可能增加出血风险", contents) == 1.0
    assert _literal_overlap("请问今天门诊什么时候开门", contents) == 0.0


def test_literal_overlap_legitimate_rewrite_stays_low():
    """正常改写（换词、口语化）不应触发 >60% 拦截——检查的是字面重叠不是语义重叠。"""
    contents = ["银杏叶提取物与华法林合用时可能增加出血风险，建议加强 INR 监测。"]
    question = "我最近在吃银杏叶片，还吃着华法林，会不会容易出血？"
    assert _literal_overlap(question, contents) < 0.6


def test_normalize_draft_cases_reassigns_split():
    """组装时：id 缺省补前缀+序号；draft 里的 split 一律被重分配（分区只归流水线）。"""
    drafts = [
        {"id": "MH-001", "question": "q1", "split": "held_out"},
        {"question": "q2", "split": "train"},  # 缺 id
    ]
    cases = normalize_draft_cases(drafts, "multihop")
    assert cases[0]["id"] == "MH-001"
    assert cases[1]["id"] == "MH-002"
    for case in cases:
        # split 必须等于哈希分配结果（draft 写什么都作废）
        assert case["split"] == assign_split(case["id"])


def test_assign_split_uses_sha256():
    """实现细节锁：split 依赖 id 的 SHA256 前 8 位 hex % 4 == 0（与文档一致）。"""
    assert assign_split("X") == (
        "held_out" if int(hashlib.sha256(b"X").hexdigest()[:8], 16) % 4 == 0 else "train")
