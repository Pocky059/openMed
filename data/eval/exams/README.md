# 考卷目录

本目录存放五套评测考卷，**由第 3 步生成流水线产出**（`scripts/build_exams.py`），不手写：

| 文件 | 内容 | 加载入口 |
|---|---|---|
| `MANIFEST.json` | 考卷清单（schema 版本/生成日期/语料与规则版本/各套数量与 held-out 数） | `load_manifest()` |
| `intent_v1.json` | 意图集 150-200 条 | `load_intent_exam()` |
| `qa_v1.json` | 问答集 80-120 条（single 走 /search 直测检索，multi 走 agent 链路） | `load_qa_exam()` |
| `multihop_v1.json` | 多跳集 50-80 条（安全档/常规档/组件跳档） | `load_multihop_exam()` |
| `long_session_v1.json` | 长会话集 15-20 条（10-20 轮剧本） | `load_long_session_exam()` |
| `emergency_v1.json` | 急症集 30 条（红旗 20 / 阴性 10，测漏检率与误报率） | `load_emergency_exam()` |

Schema 定义、校验规则与字段说明见 [evaluation/exam_loader.py](../../../evaluation/exam_loader.py)（校验失败抛 ValueError）。

**铁律（语料有据）**：每道题的 gold_chunks / entity_chain / safety 从语料与规则库实录提取，校验脚本 `scripts/validate_exams.py` 断言实存；`split` 在生成流水线里按 id 的 SHA256 哈希当场分配（~25% 落 held_out）并固化进 JSON。

**判分纪律**：考卷是「答案册」的一部分（rule_ref/safety/gold_* / expected_* 字段只供判分器读）；被测系统视野里只有题干（message/question/turns），绝无答案字段。
