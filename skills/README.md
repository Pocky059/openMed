# OpenMed Skills 文档

OpenMed 启动时会从 `OPENMED_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护症状分诊规范、用药咨询话术、预约挂号处理规则、升级条件和禁止事项。

当前内置三类 Skills：

```text
skills/symptom_triage/SKILL.md      # 症状分诊：接待、澄清、分流和转人工/急诊
skills/medication_consult/SKILL.md  # 用药咨询：用法用量、相互作用、不良反应和升级边界
skills/appointment_booking/SKILL.md # 预约挂号：预约、改期取消、挂号费核实和升级规则
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 用药咨询处理规范
description: 适用于 MedicationAgent 的用药安全核查、剂量说明和相互作用提示规范
keywords: 吃药,用药,剂量,用法用量,副作用,相互作用,过敏
agents: medication
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `symptom_triage`、`medication`、`appointment`、`emergency`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把症状分诊、用药、预约规则混在一个文件里。
- 必须包含"角色定位""处理流程""升级条件""禁止事项"等稳定章节。
- 对用户隐私、身份证号、支付密码、验证码等敏感信息必须写明禁止收集或禁止公开。
- 对无法保证的医疗结论使用保守措辞，例如"通常""建议""需要核验或就医确认"。
- 涉及红旗症状、严重不良反应、需要人工或急诊处理的场景要明确写出升级条件。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
