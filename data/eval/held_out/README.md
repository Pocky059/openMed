# held-out 审计快照目录

`held_out_ids.txt` 由第 3 步生成流水线产出：每行一个 held-out case 的 id，与考卷 JSON 内的 `split: "held_out"` 字段互为双保险。

- 阶段 2.1 捞取失败样本时硬编码拒绝 `split=="held_out"`（单测锁定）——held-out 是评测隔离的底线，被训练数据碰过就失去意义
- held-out 指标单独列出，不进 headline
