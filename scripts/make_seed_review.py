"""生成黄金种子人审稿（阶段 2.1 第 2 步，交用户审核用）

读 seeds_draft.json 生成一个人类易读的 markdown：
  1. 题面（用户要判断像不像真实用户提问）
  2. 实体链（机器先验验证过的检索目标）
  3. 两步查询（用户重点审：查询词是否自然、两步是否合理）
  4. 最终答案（用户审医疗结论方向，已按语料「方向/机制/处置建议」撰写）

可选参数 --verify <验证输出文件>：把 verify_seed_queries.py 的 stdout
结果合并进审核稿，每条种子标 [OK]/[FAIL] 和每步命中数，方便对照。

用法：
  .venv/Scripts/python.exe scripts/make_seed_review.py \
      --verify _tmp_verify50.txt \
      > data/sft/multihop/seeds/review_draft.md
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "sft" / "multihop" / "seeds" / "seeds_draft.json"


def parse_verify(text: str) -> dict[str, tuple[str, list[str]]]:
    """解析验证脚本输出：seed_id -> (状态, 每步描述行列表)。"""
    out: dict[str, tuple[str, list[str]]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"\[(OK|FAIL)\] (SEED-\d+|MH-\d+)", line)
        if m:
            current = m.group(2)
            out[current] = (m.group(1), [])
        elif current and line.startswith("   "):
            out[current][1].append(line.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", help="verify_seed_queries.py 的输出文件（可选）")
    args = ap.parse_args()

    seeds = __import__("json").loads(SEEDS.read_text(encoding="utf-8"))
    verdicts: dict = {}
    if args.verify:
        # 验证脚本跑在 Windows 控制台，重定向输出是 GBK 编码，先试 UTF-8 再回退 GBK
        raw = Path(args.verify).read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="replace")
        verdicts = parse_verify(text)

    # 审核稿写到项目内 UTF-8 文件，控制台只打行数（避免 Windows GBK 乱码）
    lines: list[str] = []
    lines.append("# 黄金种子审核稿（50 条）")
    lines.append("")
    lines.append("结构：题面 → 实体链 → 两步查询 → 最终答案。")
    lines.append("审核重点：① 题面像不像真实用户提问；② 查询词是否自然、两步是否合理；")
    lines.append("③ 答案的医疗结论方向是否正确（依据是语料的方向/机制/处置建议）。")
    if verdicts:
        lines.append("带 [OK]/[FAIL] 标记的是检索机器先验结果（实体链 chunk 是否被查询结果覆盖）。")
    lines.append("")

    source_titles = {"failed": "失败轨迹改写", "new": "新场景"}
    for i, seed in enumerate(seeds, 1):
        sid = seed["seed_id"]
        verdict = verdicts.get(sid)
        head = f"## {i}. {sid}"
        if verdict:
            status, notes = verdict
            head += f" [{status}]"
        head += f"（{source_titles.get(seed.get('source', ''), seed.get('source', ''))}）"
        lines.append(head)
        lines.append("")
        lines.append(f"**题面**：{seed['question']}")
        lines.append("")
        lines.append(f"**实体链**：{' → '.join(seed['entity_chain'])}")
        lines.append("")
        lines.append("**两步查询**：")
        lines.append("")
        for plan in seed["search_plan"]:
            obs = plan.get("observation") or "（回填后写观察）"
            lines.append(f"- 第 {plan['step']} 步：`{plan['query']}` — 观察：{obs}")
        lines.append("")
        lines.append("**最终答案**：")
        lines.append("")
        for para in seed["final_answer"].split("\n"):
            lines.append(para if para.strip() else "")
        lines.append("")
        if verdict and verdict[1]:
            lines.append("**机器先验明细**：")
            for note in verdict[1]:
                lines.append(f"- {note}")
            lines.append("")

    out = ROOT / "data" / "sft" / "multihop" / "seeds" / "review_draft.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = sum(1 for v in verdicts.values() if v[0] == "OK")
    print(f"审核稿已生成：{out.relative_to(ROOT)}，共 {len(seeds)} 条"
          + (f"，机器先验 {ok}/{len(verdicts)} 通过" if verdicts else ""))


if __name__ == "__main__":
    main()
