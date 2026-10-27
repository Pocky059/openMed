import { useCallback, useEffect, useRef, useState } from 'react'
import ChatPanel from './components/ChatPanel.jsx'
import TracePanel from './components/TracePanel.jsx'
import { sendChat, fetchHealth, resetDemoMemory } from './api.js'

// 演示用的固定用户 id（同一个「患者」连续对话，记忆系统会累积病史）。
const DEMO_USER_ID = 'web-demo'

let seq = 0
const nextId = () => `msg-${++seq}`

export default function App() {
  // 消息流：{id, role: 'user'|'assistant', text, meta?, error?, loading?}
  const [messages, setMessages] = useState([])
  // 多轮对话：第一次 /chat 响应里返回 conv_id，之后每轮都带上
  const [convId, setConvId] = useState(null)
  // 右侧面板选中的消息 id（默认跟随最新助手消息）
  const [selectedId, setSelectedId] = useState(null)
  const [busy, setBusy] = useState(false)
  // 服务健康状态
  const [health, setHealth] = useState(null)
  // 重置演示记忆：进行中标记 + 完成提示（几秒后消失）
  const [resetting, setResetting] = useState(false)
  const [resetMsg, setResetMsg] = useState(null)

  const checkHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth())
    } catch {
      setHealth({ status: 'offline' })
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const t = setInterval(checkHealth, 15000)
    return () => clearInterval(t)
  }, [checkHealth])

  const send = useCallback(
    async (text) => {
      if (busy || !text.trim()) return
      const userMsg = { id: nextId(), role: 'user', text }
      setMessages((prev) => [...prev, userMsg])
      setSelectedId(userMsg.id)
      setBusy(true)

      const loadingMsg = { id: nextId(), role: 'assistant', text: '', loading: true }
      setMessages((prev) => [...prev, loadingMsg])

      try {
        // convId 用函数式更新拿最新值，避免闭包拿到旧的
        const meta = await sendChat(text, DEMO_USER_ID, convId)
        if (!convId) setConvId(meta.conv_id)

        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id
              ? { ...m, loading: false, text: meta.response, meta }
              : m,
          ),
        )
        setSelectedId(loadingMsg.id)
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id
              ? { ...m, loading: false, error: String(err.message || err) }
              : m,
          ),
        )
      } finally {
        setBusy(false)
      }
    },
    [busy, convId],
  )

  // 重置演示记忆：调后端 /demo/reset 清空 web-demo 的全部记忆，
  // 然后本地清空聊天区、丢弃 conv_id（后端旧会话的记忆已删，直接开新会话）。
  const handleReset = async () => {
    if (busy || resetting) return
    if (!window.confirm('确定重置演示用户的全部记忆吗？对话历史和长期记忆（过敏史等）都会清空。')) return
    setResetting(true)
    try {
      const r = await resetDemoMemory(DEMO_USER_ID)
      setConvId(null)
      setMessages([])
      setSelectedId(null)
      const n = r.deleted
        ? Object.values(r.deleted).reduce((a, b) => a + b, 0)
        : 0
      setResetMsg(n > 0 ? `已重置，清空 ${n} 条记忆` : '已重置（记忆本来就是空的）')
      setTimeout(() => setResetMsg(null), 4000)
    } catch (err) {
      window.alert(`重置失败：${err.message || err}`)
    } finally {
      setResetting(false)
    }
  }

  const selected = messages.find((m) => m.id === selectedId) || null

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">＋</span>
          <h1>OpenMed 智能问诊分流系统</h1>
        </div>
        <div className="topbar-right">
          {resetMsg && <span className="reset-msg">{resetMsg}</span>}
          {health && (
            <span className={`health ${health.status === 'ok' ? 'ok' : 'bad'}`}>
              ● {health.status === 'ok' ? '服务正常' : '服务离线'}
            </span>
          )}
          <button
            className="reset-btn"
            onClick={handleReset}
            disabled={resetting || busy}
            title="清空演示用户 web-demo 的全部记忆（Redis + ChromaDB）"
          >
            {resetting ? '重置中…' : '重置记忆'}
          </button>
          <span className="hint">右侧面板可查看系统内部过程</span>
        </div>
      </header>

      <main className="layout">
        <ChatPanel
          messages={messages}
          busy={busy}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onSend={send}
        />
        <TracePanel message={selected} />
      </main>
    </div>
  )
}
