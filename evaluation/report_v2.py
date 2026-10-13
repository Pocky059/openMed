"""
基线报告聚合（v2 阶段 1 第 5 步模块 5）。

职责：把各集判分结果聚合成 baseline.json v2（schema 见 v2-plan 1.2）：
  - 各聚合函数是确定性纯函数（同一输入同一数字，指标可复算可审计）
  - build_baseline 组装完整 schema；headline 五数字从各集指标里抽，
    不是单独维护的另一份数字（防两处漂移）
  - 空输入聚合返回 0 而非报错（防除零：某集 0 题时报告仍要能生成）

与 evaluation/evaluator.py（legacy）的关系：本模块只产 v2 结构；
legacy 字段由基线 CLI 用旧 EvalReport 填，回归检测 _detect_regressions
继续读它——两套数字并存，谁也不覆盖谁。
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.multihop_scorer import LongSessionResult, MultihopResult
from evaluation.safety_scorer import SafetyResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"


# ── 各集聚合（确定性纯函数）───────────────────────────────────────────────────

def aggregate_intent(pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    """意图集聚合：accuracy / macro_f1 / per_class。

    pairs 每项 {"case_id", "expected", "actual"}（expected 是考卷枚举，
    actual 是 /chat 响应里的 intent）。per_class 对每个出现过的类别算
    P/R/F1，macro_f1 是所有类别 F1 的宏平均（类别从未被判出也算 0）。
    """
    classes = sorted({p["expected"] for p in pairs} | {p["actual"] for p in pairs})
    per_class: Dict[str, Any] = {}
    f1s: List[float] = []
    correct = 0
    for cls in classes:
        tp = sum(1 for p in pairs if p["expected"] == cls and p["actual"] == cls)
        pred = sum(1 for p in pairs if p["actual"] == cls)
        gold = sum(1 for p in pairs if p["expected"] == cls)
        precision = tp / pred if pred else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {
            "expected": gold, "actual": pred, "correct": tp,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)
        correct += tp
    total = len(pairs)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "total": total,
        "per_class": per_class,
    }


def aggregate_dialog_single(stats: List[Dict[str, float]]) -> Dict[str, float]:
    """对话集单轮检索统计聚合：P@k/R@k/F1@k/MRR/hit@k 的均值。

    stats 每项是单题检索统计（基线 CLI 用 ragas_metrics 的确定性函数算）。
    """
    if not stats:
        return {"f1_at_5": 0.0, "precision_at_5": 0.0, "recall_at_5": 0.0,
                "mrr": 0.0, "hit_at_5": 0.0, "cases": 0}
    keys = ("precision_at_5", "recall_at_5", "f1_at_5", "mrr", "hit_at_5")
    out = {key: round(sum(s.get(key, 0.0) for s in stats) / len(stats), 4) for key in keys}
    out["cases"] = len(stats)
    return out


def aggregate_dialog(single_stats: List[Dict[str, float]],
                     judge_scores: Optional[List[float]] = None) -> Dict[str, Any]:
    """对话（问答）集聚合：single_hop_f1 = 单轮检索 F1@5 均值（headline），
    avg_judge_scores = 现有四维 judge 的均值（legacy 语义延续，可选）。"""
    retrieval = aggregate_dialog_single(single_stats)
    out: Dict[str, Any] = {"single_hop_f1": retrieval["f1_at_5"], "retrieval": retrieval}
    if judge_scores:
        out["avg_judge_scores"] = round(sum(judge_scores) / len(judge_scores), 4)
    return out


def aggregate_multihop(results: List[MultihopResult]) -> Dict[str, Any]:
    """多跳集聚合：成功率 / 链命中率 / 有序率 / 步数效率 / 少步解决统计。"""
    total = len(results)
    if not total:
        return {"success_rate": 0.0, "entity_chain_hit_rate": 0.0,
                "ordered_chain_rate": 0.0, "step_efficiency": 0.0,
                "avg_actual_steps": 0.0, "understep_cases": 0, "cases": 0}
    success = sum(1 for r in results if r.success)
    chain_hit = sum(1 for r in results if r.entity_chain_hit)
    ordered = sum(1 for r in results if r.ordered)
    understep = sum(1 for r in results if r.understep)

    def efficiency(r: MultihopResult) -> float:
        """单题步数效率：预期步数/实际步数，封顶 1.0（少步不算浪费）。

        实际步数 0（一次都没搜）记 0——完全没干活谈不上效率。
        """
        if r.actual_steps <= 0:
            return 0.0
        return min(1.0, r.expected_steps / r.actual_steps)

    return {
        "success_rate": round(success / total, 4),
        "entity_chain_hit_rate": round(chain_hit / total, 4),
        "ordered_chain_rate": round(ordered / total, 4),
        "step_efficiency": round(sum(efficiency(r) for r in results) / total, 4),
        "avg_actual_steps": round(sum(r.actual_steps for r in results) / total, 4),
        "understep_cases": understep,
        "cases": total,
    }


def aggregate_long_session(results: List[LongSessionResult]) -> Dict[str, float]:
    """长会话集聚合：recall 与 loss_rate 的均值（丢失率是 headline）。"""
    total = len(results)
    if not total:
        return {"recall": 0.0, "loss_rate": 0.0, "cases": 0}
    return {
        "recall": round(sum(r.recall for r in results) / total, 4),
        "loss_rate": round(sum(r.loss_rate for r in results) / total, 4),
        "cases": total,
    }


def aggregate_safety(results: List[SafetyResult]) -> Dict[str, Any]:
    """安全判分聚合：违规率 / 未覆盖率 / severity 分层。

    违规率分母 = hard_eligible 题数（三道预检全过才进分母，坏数据
    不污染 headline）；uncovered_rate = 规则库查不到的题 / 全部题。
    """
    total = len(results)
    if not total:
        return {"overall_violation_rate": 0.0, "uncovered_rate": 0.0,
                "denominator": 0, "violations": 0, "cases": 0, "by_severity": {}}
    denominator = sum(1 for r in results if r.hard_eligible)
    violations = sum(1 for r in results if r.hard_eligible and r.violation)
    uncovered = sum(1 for r in results if not r.rule_found)
    by_severity: Dict[str, Any] = {}
    for severity in ("major", "moderate", "minor"):
        group = [r for r in results if r.hard_eligible and r.severity == severity]
        by_severity[severity] = {
            "cases": len(group),
            "violations": sum(1 for r in group if r.violation),
            "rate": round(sum(1 for r in group if r.violation) / len(group), 4) if group else 0.0,
        }
    return {
        "overall_violation_rate": round(violations / denominator, 4) if denominator else 0.0,
        "uncovered_rate": round(uncovered / total, 4),
        "denominator": denominator,
        "violations": violations,
        "cases": total,
        "by_severity": by_severity,
    }


def aggregate_emergency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """急症集聚合：漏检率（红旗题未升级）与误报率（阴性题被升级）。

    判分纯确定性（v2-plan 1.0.10）：rows 每项 {"case_id", "expected_escalated",
    "actual_escalated"}——期望值由考卷词表确定性得出，实际值读 /chat
    响应的 escalated 字段。红旗漏检理论上恒 0，任何 >0 都是真漏检。
    """
    red = [r for r in rows if r.get("expected_escalated")]
    negative = [r for r in rows if not r.get("expected_escalated")]
    missed = sum(1 for r in red if not r.get("actual_escalated"))
    false_alarm = sum(1 for r in negative if r.get("actual_escalated"))
    return {
        "missed_escalation_rate": round(missed / len(red), 4) if red else 0.0,
        "false_escalation_rate": round(false_alarm / len(negative), 4) if negative else 0.0,
        "red_flag_cases": len(red),
        "negative_cases": len(negative),
        "missed": missed,
        "false_alarms": false_alarm,
    }


# ── v2 schema 组装 ────────────────────────────────────────────────────────────

def build_baseline(
    run_id: str,
    config: Dict[str, Any],
    exam_metrics: Dict[str, Any],
    safety: Dict[str, Any],
    emergency: Dict[str, Any],
    retrieval: Dict[str, Any],
    results: List[Dict[str, Any]],
    legacy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组装 baseline.json v2 完整结构。

    headline 五数字从 exam_metrics/safety/emergency 里抽——报告里不维护
    第二份数字，改口径只改聚合函数一处。
    """
    headline = {
        "single_hop_f1": exam_metrics.get("dialog", {}).get("single_hop_f1", 0.0),
        "multihop_success_rate": exam_metrics.get("multihop", {}).get("success_rate", 0.0),
        "long_session_loss_rate": exam_metrics.get("long_session", {}).get("loss_rate", 0.0),
        "safety_violation_rate": safety.get("overall_violation_rate", 0.0),
        # 漏检率（红旗题未升级）：理论恒 0，任何 >0 都是真漏检（门控链路坏）
        "emergency_miss_rate": emergency.get("missed_escalation_rate", 0.0),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "config": config,
        "legacy": legacy if legacy is not None else {},
        "headline": headline,
        "exams": exam_metrics,
        "safety": safety,
        "emergency": emergency,
        "retrieval": retrieval,
        "results": results,
    }


def save_baseline(path: Any, data: Dict[str, Any]) -> None:
    """落盘 baseline.json（UTF-8、ensure_ascii=False，中文可读）。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_baseline(path: Any) -> Dict[str, Any]:
    """读 baseline.json 并校验 schema 版本（版本不认识的报告拒绝处理）。"""
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"baseline schema 版本不匹配: {data.get('schema_version')!r}（期望 {SCHEMA_VERSION}）")
    return data
