// 后端接口封装。所有请求走相对路径：
// - 开发模式：Vite proxy 转发到 http://localhost:80（nginx → FastAPI）
// - 生产模式：build 产物由 FastAPI 静态挂载，同源直连

/** 发一条消息走 /chat 完整链路，返回 ChatResponse。 */
export async function sendChat(message, userId, convId) {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      user_id: userId,
      ...(convId ? { conv_id: convId } : {}),
    }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`后端返回 ${res.status}：${text.slice(0, 200)}`)
  }
  return res.json()
}

/** 拉取某次请求的工具调用轨迹（每步检索的查询词、状态、耗时）。 */
export async function fetchToolTrace(requestId) {
  if (!requestId) return null
  const res = await fetch(`/trace/tool/${requestId}`)
  if (!res.ok) return null
  const data = await res.json()
  return data.found ? data.trace : null
}

/** 拉取某次请求的完整轨迹（/trace/full，读落盘文件）：
 * meta.intent_detail（意图识别输入/输出）、meta.input（编排器输入）、
 * messages（拼给 LLM 的完整对话，含 system prompt）。 */
export async function fetchFullTrace(requestId) {
  if (!requestId) return null
  const res = await fetch(`/trace/full/${requestId}`)
  if (!res.ok) return null
  const data = await res.json()
  return data.found ? data.trajectory : null
}

/** 复现一次检索：用同样的查询词打 /search，看命中了哪些文档。 */
export async function searchKnowledge(query, topK = 5) {
  const res = await fetch(`/search?query=${encodeURIComponent(query)}&top_k=${topK}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(`检索失败：${res.status}`)
  return res.json()
}

/** 健康检查：返回服务状态与各 agent 统计。 */
export async function fetchHealth() {
  const res = await fetch('/health')
  if (!res.ok) throw new Error(`健康检查失败：${res.status}`)
  return res.json()
}

/** 重置演示用户的全部记忆（后端按 user_id 全清 Redis + ChromaDB）。 */
export async function resetDemoMemory(userId = 'web-demo') {
  const res = await fetch(`/demo/reset?user_id=${encodeURIComponent(userId)}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(`重置失败：${res.status}`)
  return res.json()
}

// ── 监控 ─────────────────────────────────────────────────────────────────────

/** 监控摘要：Agent 统计 / 工具统计 / 活动告警 / 优化建议（/monitor）。 */
export async function fetchMonitor() {
  const res = await fetch('/monitor')
  if (!res.ok) throw new Error(`监控接口失败：${res.status}`)
  return res.json()
}

/** Prometheus 原始指标文本（/metrics，与 Prometheus 容器抓取同一份数据）。 */
export async function fetchMetrics() {
  const res = await fetch('/metrics')
  if (!res.ok) throw new Error(`指标接口失败：${res.status}`)
  return res.text()
}

// ── 知识库管理 ───────────────────────────────────────────────────────────────

/** 知识库文档列表（按标题聚合片段数，source=builtin/user）。 */
export async function fetchKnowledgeDocs() {
  const res = await fetch('/knowledge/docs')
  if (!res.ok) throw new Error(`文档列表失败：${res.status}`)
  return res.json()
}

/** 按标题取一篇文档的全部片段（只读阅览）。标题含中文，需 URL 编码。 */
export async function fetchKnowledgeDoc(title) {
  const res = await fetch(`/knowledge/doc/${encodeURIComponent(title)}`)
  if (!res.ok) throw new Error(`文档内容读取失败：${res.status}`)
  return res.json()
}

/** 批量导入文档（每篇自动切片入库）。documents: [{title, content}, ...] */
export async function addKnowledge(documents) {
  const res = await fetch('/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`导入失败（${res.status}）：${text.slice(0, 200)}`)
  }
  return res.json()
}

/** 上传文件导入知识库（.txt/.md 整篇、.json 文档数组）。 */
export async function uploadKnowledgeFile(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/knowledge/upload', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`上传失败（${res.status}）：${text.slice(0, 200)}`)
  }
  return res.json()
}

// ── Agent 与 Skill 管理 ──────────────────────────────────────────────────────

/** Agent 目录：每个 Agent 的角色契约与工具白名单（含参数 schema）。 */
export async function fetchAgentsCatalog() {
  const res = await fetch('/agents/catalog')
  if (!res.ok) throw new Error(`Agent 目录失败：${res.status}`)
  return res.json()
}

/** Skill 列表（full=true 附带完整内容，Agent 管理页展示用）。 */
export async function fetchSkills(full = false) {
  const res = await fetch(`/skills${full ? '?full=1' : ''}`)
  if (!res.ok) throw new Error(`Skill 列表失败：${res.status}`)
  return res.json()
}

/** 上传新 Skill（.md/.txt/.json；与内置同名会被后端拒绝）。
 * agents：适用 Agent 列表；空数组 = 公用（适用所有 Agent）。 */
export async function uploadSkillFile(file, agents = []) {
  const form = new FormData()
  form.append('file', file)
  form.append('agents', agents.join(',')) // 空串 = 公用，后端语义一致
  const res = await fetch('/skills/upload', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`上传失败（${res.status}）：${text.slice(0, 200)}`)
  }
  return res.json()
}
