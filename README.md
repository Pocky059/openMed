# OpenMed

OpenMed 是一个面向医疗问诊分流场景的多 Agent 编排智能问诊系统，覆盖症状分诊、预约挂号、用药咨询、急诊四大场景。它不是单纯的聊天机器人，而是把以下能力串成闭环：

- 三路融合的细粒度意图识别（LLM 语义理解 0.7 + Embedding 相似度 0.2 + 关键词兜底 0.1）
- 路由驱动的多 Agent 编排（意图 + 关键词 + 实体累积打分，支持主 Agent + 辅助 Agent 并行协作）
- 医疗 RAG 知识库：BM25 + 向量混合召回 → RRF 融合排序 → Top-K → Cross-Encoder 精排
- Redis + ChromaDB 三级记忆（工作记忆 / 情景记忆 / 用户画像）
- 动态 Skills 注入（按 Agent 类型和关键词匹配业务处理规范）
- 在线监控与路由降权（Agent/工具成功率、延迟反馈到路由分数）
- LLM-as-Judge 端到端评测（意图识别准确率、回复质量多维评分、回归检测）

## 业务场景与 Agent

| 场景 | Agent | 说明 |
|------|-------|------|
| 症状分诊 | `SymptomTriageAgent` | 首诊接待、症状澄清、科室分流，不给确定性诊断结论 |
| 用药咨询 | `MedicationAgent` | 用法用量、药物相互作用、不良反应核查 |
| 预约挂号 | `AppointmentAgent` | 新预约、改期、取消、挂号费核验 |
| 急诊 | `EmergencyAgent` | 非 LLM 直接交接节点，识别到高危症状/转人工时跳过模型调用，直接标准化交接 |

意图识别覆盖 19 类细粒度意图（如 `symptom_check`、`medication_dosage`、`medication_interaction`、`appointment_reschedule`、`appointment_payment_issue`、`emergency`、`human_handoff` 等），归一化后映射到上述 4 个 Agent。复合问题（例如同时问症状和用药相互作用）会触发主 Agent + 辅助 Agent 并行处理。


## 快速开始

### 1. 准备环境

- Docker + Docker Compose（起 Redis / ChromaDB / Prometheus / Nginx 等依赖服务）
- Python 3.11 或 3.12（本地跑单测/CLI 时用；不建议用 3.13+，部分依赖暂无预编译 wheel）
- `ANTHROPIC_API_KEY`

如果使用兼容 Anthropic 协议的第三方模型服务，也可以配置：

```env
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-v4-pro
ANTHROPIC_API_KEY=your_key
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

最少确认这些变量可用：

```env
ANTHROPIC_API_KEY=your_api_key
REDIS_PASSWORD=your_redis_password
```

### 3. 启动服务

推荐直接启动全栈：

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
```

看日志：

```bash
docker compose logs -f openmed
```

只想在本地跑代码、用 Docker 起依赖服务方便调试时：

```bash
docker compose up -d redis chromadb
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m api.main --cli
```

### 4. 访问入口

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Nginx: `http://localhost`
- Health: `http://localhost:8000/health`

## 核心功能

### 对话主链路

`POST /chat`

流程是：

```text
读取记忆 -> 意图识别 -> 知识检索 -> Agent 路由 -> 回复生成 -> 写回记忆
```

请求示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "布洛芬和阿莫西林能一起吃吗？", "user_id": "u1"}'
```

响应关键字段：`response` / `intent` / `intent_group` / `agent_type` / `primary_agent` / `supporting_agents` / `tools_used` / `routing_reason` / `escalated` / `knowledge_used` / `entities`。

急诊/转人工场景验证：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "胸口剧烈疼痛，喘不上气", "user_id": "u1"}'
```

预期 `agent_type: "emergency"`、`escalated: true`，且不产生 LLM 调用（`EmergencyAgent` 是非 LLM 的确定性交接节点）。

### 知识库

- `POST /search` — 单独调试 BM25+向量混合召回 → RRF 融合 → Cross-Encoder 精排效果
- `POST /knowledge/add` / `POST /knowledge/upload` — 写入医疗文档（药品说明书、科室指引、预约流程等）
- `GET /knowledge/stats` — 查看知识库文档统计

### Skills

- `GET /skills`
- `POST /skills/reload`

当前内置三类 Skills：`症状分诊接待规范`、`用药咨询处理规范`、`预约挂号处理规范`，按 `agents` + `keywords` 动态注入对应 Agent 的 system prompt。

### 监控与评测

- `GET /monitor`
- `POST /eval/run`

## 项目结构

```text
api/main.py                  FastAPI 入口
agents/agent_orchestrator.py 多 Agent 编排（路由、并行协作、降级）
core/intent_recognizer.py    三路融合意图识别
core/reranker.py             Cross-Encoder 精排（带词重叠降级）
core/skill_loader.py         动态 Skills 加载
memory/conversation_memory.py  Redis + ChromaDB 记忆
mcp/tool_manager.py          工具层、缓存、熔断、精排调用
mcp/knowledge_base.py        BM25 + 向量混合检索知识库
monitor/performance_monitor.py 在线监控
evaluation/evaluator.py      端到端评测（LLM-as-Judge）
skills/                      动态业务规则
data/                        持久化数据
tests/                       pytest 单元测试
```

## 运行时架构

```text
用户请求
  -> /chat
  -> MemoryManager 读取工作记忆、情景记忆、用户画像
  -> IntentRecognizer 输出 intent / intent_group / urgency / entities
  -> 按意图决定是否检索知识库
  -> AgentOrchestrator 路由到 SymptomTriage / Medication / Appointment / Emergency
  -> Skills 注入、工具调用、回复生成
  -> 写回 Redis 和 ChromaDB
  -> Monitor 采集在线指标
  -> Evaluator 做意图识别和回复质量评测
```

## 主要端口

| 服务 | 端口 |
|---|---:|
| OpenMed API | 8000 |
| ChromaDB | 8001 |
| Redis | 6379 |
| Prometheus | 9090 |
| Nginx | 80 |

## 测试

单元测试（不依赖 Docker/API Key，mock LLM client）：

```bash
pytest -q
```

覆盖 Agent Profile 契约、路由决策（单 Agent / 主辅并行）、工具白名单隔离、工具调用往返、急诊非 LLM 交接节点等场景。

想验证真实端到端链路（`/chat` + Redis + ChromaDB + 真实模型调用），需要先 `docker compose up -d --build` 并配置好 `ANTHROPIC_API_KEY`，再通过 Swagger UI 或 curl 调用。

## 开发和调试

常用顺序：

```text
1. /health
2. /chat
3. /skills
4. /monitor
5. /eval/run
```

