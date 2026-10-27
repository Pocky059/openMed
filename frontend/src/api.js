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
