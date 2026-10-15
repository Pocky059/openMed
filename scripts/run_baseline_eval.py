"""
基线评测 CLI（v2 阶段 1 第 7 步）。

为什么是 CLI 而不是 HTTP 端点（v2-plan 1.3 第 7 步理由）：
  1. 全量 90-120 分钟，同步 HTTP 过不了 nginx 60s 超时链
  2. 断点续跑/部分重跑（--resume/--cases/--merge）要能随时中断恢复
  3. 不占 HTTP 连接；跑题必须串行（编排器有实例级状态），
     CLI 一个进程按顺序跑天然满足

跑法（容器内，与生产同环境；OPENMED_EVAL_MODE=1 禁工具缓存，
急症告警默认进模拟收件箱，绝不外发）：
    docker compose exec openmed env OPENMED_EVAL_MODE=1 python scripts/run_baseline_eval.py

子集跑法（pilot 校准 / 部分重跑）：
    docker compose exec openmed env OPENMED_EVAL_MODE=1 \
        python scripts/run_baseline_eval.py --pilot
    docker compose exec openmed env OPENMED_EVAL_MODE=1 \
        python scripts/run_baseline_eval.py --cases QA-001,MH-002 --merge

判分口径修订后刷新 baseline.json（不全量重跑）：
    docker compose exec openmed env OPENMED_EVAL_MODE=1 \
        python scripts/run_baseline_eval.py --run-id baseline-20261013-rev2 \
            --rejudge-from baseline-20261013 --cases LS-006,LS-007,LS-008

换被测模型跑基线（2026-10-15 Qwen3 基线准备；vLLM 起在 AutoDL、SSH 隧道
转到宿主机 8000 端口）：被测系统改走 OpenAI 协议，判卷单独配回 DeepSeek
（不单独配会被 BaselineRunner 拦下，防止被测模型给自己打分），qa single 与
强制 LLM 失败子集与模型无关，跳过：
    docker compose exec openmed env OPENMED_EVAL_MODE=1 \
        OPENMED_LLM_PROTOCOL=openai OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
        OPENAI_API_KEY=EMPTY ANTHROPIC_MODEL=<vLLM 的 served-model-name> \
        OPENMED_LLM_THINKING=off \
        OPENMED_JUDGE_MODEL=deepseek-v4-pro \
        OPENMED_JUDGE_BASE_URL=https://api.deepseek.com/anthropic \
        python scripts/run_baseline_eval.py --run-id qwen3-base-<日期> --skip-model-independent
（OPENMED_JUDGE_API_KEY 不设时沿用 .env 的 ANTHROPIC_API_KEY，即 DeepSeek key。）

产出 results/<run_id>/ 四件：baseline.json / cases.jsonl /
trajectory_quality_report.json / report.md，急症告警另落
emergency_webhooks.jsonl；baseline.json 同时复制到 data/eval/。
"""
import argparse
import asyncio
import logging
import os
from dataclasses import asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("baseline_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMed 基线评测（五套考卷，全量或子集）")
    parser.add_argument("--pilot", action="store_true",
                        help="每集只跑前 3 题（校准时长/成本用）")
    parser.add_argument("--cases", default="",
                        help="只跑指定题号，逗号分隔（如 QA-001,MH-002），配合 --merge 部分重跑")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：跳过 cases.jsonl 已完成的行")
    parser.add_argument("--merge", action="store_true",
                        help="配合 --cases 部分重跑：新行覆盖旧行，其余保留")
    parser.add_argument("--run-id", default="",
                        help="指定 run_id（默认按天 baseline-YYYYMMDD）")
    parser.add_argument("--rejudge-from", default="",
                        help="从旧 run 产物重聚合出新 run（不全量重跑）：读 results/<旧run_id>/cases.jsonl，"
                             "配合 --cases 重判指定题，其余行保留，按当前判分口径聚合。"
                             "用于判分口径修订后快速刷新 baseline.json")
    parser.add_argument("--no-progress", action="store_true",
                        help="关闭进度条（把输出重定向到文件时用，动态刷新行会糊成一片）")
    parser.add_argument("--allow-emergency-webhook", action="store_true",
                        help="急症告警允许发往容器配置的真实 URL（默认模拟收件箱，绝不外发）")
    parser.add_argument("--skip-model-independent", action="store_true",
                        help="换被测模型时用：跳过 qa single（直测检索）与强制 LLM 失败子集"
                             "（LLM 路被短路）——两者都不经过被测模型，结果与模型无关")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # 评测模式守卫：禁工具缓存靠这个 env（不设会污染检索轨迹、判分失真）
    if os.environ.get("OPENMED_EVAL_MODE") != "1":
        logger.warning("未设置 OPENMED_EVAL_MODE=1：工具缓存未禁用，"
                       "检索轨迹可能命中缓存导致判分失真")

    from core.app_factory import assemble
    from evaluation.baseline_runner import BaselineRunner
    from evaluation.exam_loader import load_all_exams, load_manifest

    # start_monitor=False：容器里生产 API 已占 Prometheus 端口，CLI 再起会崩
    ctx = await assemble(start_monitor=False)
    # loader 返回 dataclass，跑批器要纯 dict（asdict 保持字段名与 schema 一致）
    exams = {name: [asdict(c) for c in cases] for name, cases in load_all_exams().items()}
    manifest = asdict(load_manifest())

    runner = BaselineRunner(
        ctx,
        run_id=args.run_id or None,
        allow_webhook=args.allow_emergency_webhook,
        progress=not args.no_progress,
    )
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()] or None
    if args.rejudge_from:
        await runner.run_rejudge(
            args.rejudge_from, exams, manifest=manifest, case_ids=case_ids,
        )
        return
    await runner.run_all(
        exams,
        manifest=manifest,
        pilot=args.pilot,
        case_ids=case_ids,
        resume=args.resume,
        merge=args.merge,
        skip_model_independent=args.skip_model_independent,
    )


if __name__ == "__main__":
    asyncio.run(main())
