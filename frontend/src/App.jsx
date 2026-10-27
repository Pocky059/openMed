import { useCallback, useEffect, useRef, useState } from 'react'
import ChatPanel from './components/ChatPanel.jsx'
import TracePanel from './components/TracePanel.jsx'
import Sidebar from './components/Sidebar.jsx'
import ToolRagTab from './components/ToolRagTab.jsx'
import MonitorTab from './components/MonitorTab.jsx'
import KnowledgeTab from './components/KnowledgeTab.jsx'
import AgentsTab from './components/AgentsTab.jsx'
import { sendChat, fetchHealth, resetDemoMemory } from './api.js'

// 顶部整页选项卡：主界面 = 聊天 + 检查面板；其余页面整页展示。
// 侧边栏（会话列表）只属于主界面，其他页面没有它。
const PAGES = [
  { key: 'main', label: '主界面' },
  { key: 'tools', label: '工具 & RAG' },
  { key: 'monitor', label: '监控' },
  { key: 'knowledge', label: '知识库' },
  { key: 'agents', label: 'Agent 管理' },
]

// 演示用的固定用户 id（同一个「患者」连续对话，记忆系统会累积病史）。
const DEMO_USER_ID = 'web-demo'

let seq = 0
const nextId = () => `msg-${++seq}`

/** 时间戳 HH:MM:SS（精确到秒，消息与会话列表都用）。 */
const fmtTime = (d) => {
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** 会话标题：首条用户消息截 14 字（调试工具，够区分就行）。 */
const clipTitle = (t) => (t.length > 14 ? t.slice(0, 14) + '…' : t)

/** 新建一个会话对象：convId 先空着，首轮 /chat 响应后回填。
 * 「新会话」不重置记忆——工作记忆按 (user, conv) 隔离，情景记忆按 user 跨会话。 */
const newSession = () => ({
  id: nextId(),
  title: '新会话',
  createdAt: fmtTime(new Date()),
  convId: null,
  messages: [],
})

export default function App() {
  // 会话列表：默认开一个空会话（与旧版「空聊天界面」行为一致）
  const [sessions, setSessions] = useState(() => [newSession()])
  const [currentSid, setCurrentSid] = useState(sessions[0].id)
  // 边栏收起标记 / 左栏宽度百分比（inspector 吃剩余宽度）
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [chatPct, setChatPct] = useState(60)
  // 右侧面板选中的消息 id（默认跟随最新助手消息）
  const [selectedId, setSelectedId] = useState(null)
  const [busy, setBusy] = useState(false)
  // 服务健康状态
  const [health, setHealth] = useState(null)
  // 重置演示记忆：进行中标记 + 完成提示（几秒后消失）
  const [resetting, setResetting] = useState(false)
  const [resetMsg, setResetMsg] = useState(null)
  // 顶部页面选项卡：非主界面页整页渲染（侧边栏 + 聊天 + 检查面板只属于主界面）
  const [page, setPage] = useState('main')

  const current = sessions.find((s) => s.id === currentSid) || sessions[0]
  const messages = current?.messages || []
  const convId = current?.convId || null

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
      const userMsg = { id: nextId(), role: 'user', text, ts: fmtTime(new Date()) }
      const loadingMsg = { id: nextId(), role: 'assistant', text: '', loading: true, ts: fmtTime(new Date()) }
      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSid
            ? {
                ...s,
                // 首条消息顺带定会话标题（默认「新会话」）
                title: s.title === '新会话' ? clipTitle(text) : s.title,
                messages: [...s.messages, userMsg, loadingMsg],
              }
            : s,
        ),
      )
      setSelectedId(userMsg.id)
      setBusy(true)

      try {
        // convId 从当前会话对象读（闭包），首轮为空 → 后端新建会话并返回
        const meta = await sendChat(text, DEMO_USER_ID, convId)
        setSessions((prev) =>
          prev.map((s) =>
            s.id === currentSid
              ? {
                  ...s,
                  convId: s.convId || meta.conv_id,
                  messages: s.messages.map((m) =>
                    m.id === loadingMsg.id ? { ...m, loading: false, text: meta.response, meta } : m,
                  ),
                }
              : s,
          ),
        )
        setSelectedId(loadingMsg.id)
      } catch (err) {
        setSessions((prev) =>
          prev.map((s) =>
            s.id === currentSid
              ? {
                  ...s,
                  messages: s.messages.map((m) =>
                    m.id === loadingMsg.id ? { ...m, loading: false, error: String(err.message || err) } : m,
                  ),
                }
              : s,
          ),
        )
      } finally {
        setBusy(false)
      }
    },
    [busy, convId, currentSid],
  )

  // 新建会话：不调后端、不清记忆，只是换一个新 conv_id 开局
  const handleNewSession = () => {
    if (busy) return
    const s = newSession()
    setSessions((prev) => [...prev, s])
    setCurrentSid(s.id)
    setSelectedId(null)
  }

  // 切换会话：busy 时禁止（响应要写回发起它的那个会话，中途切走会丢上下文）
  const handleSwitchSession = (sid) => {
    if (busy || sid === currentSid) return
    setCurrentSid(sid)
    setSelectedId(null)
  }

  // 重置演示记忆：调后端 /demo/reset 清空 web-demo 的全部记忆（含所有会话），
  // 本地会话列表也一并清掉、回到单个新会话。
  const handleReset = async () => {
    if (busy || resetting) return
    if (!window.confirm('确定重置演示用户的全部记忆吗？所有会话的对话历史和长期记忆（过敏史等）都会清空。')) return
    setResetting(true)
    try {
      const r = await resetDemoMemory(DEMO_USER_ID)
      const s = newSession()
      setSessions([s])
      setCurrentSid(s.id)
      setSelectedId(null)
      const n = r.deleted ? Object.values(r.deleted).reduce((a, b) => a + b, 0) : 0
      setResetMsg(n > 0 ? `已重置，清空 ${n} 条记忆` : '已重置（记忆本来就是空的）')
      setTimeout(() => setResetMsg(null), 4000)
    } catch (err) {
      window.alert(`重置失败：${err.message || err}`)
    } finally {
      setResetting(false)
    }
  }

  // 拖拽分栏：按住中间把手左右拖，chatPct 限制在 30%–75%
  const layoutRef = useRef(null)
  const startDrag = useCallback(
    (e) => {
      e.preventDefault()
      const layout = layoutRef.current
      if (!layout) return
      const startX = e.clientX
      const startPct = chatPct
      const onMove = (ev) => {
        const w = layout.getBoundingClientRect().width
        const pct = Math.min(75, Math.max(30, startPct + ((ev.clientX - startX) / w) * 100))
        setChatPct(pct)
      }
      const onUp = () => {
        document.removeEventListener('mousemove', onMove)
        document.removeEventListener('mouseup', onUp)
        document.body.classList.remove('resizing')
      }
      document.addEventListener('mousemove', onMove)
      document.addEventListener('mouseup', onUp)
      document.body.classList.add('resizing')
    },
    [chatPct],
  )

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
            title="清空演示用户 web-demo 的全部记忆（Redis + ChromaDB，所有会话）"
          >
            {resetting ? '重置中…' : '重置记忆'}
          </button>
        </div>
      </header>

      {/* 页面级选项卡：主界面 = 聊天 + 检查面板；其余页面整页展示，空间更宽 */}
      <nav className="pagetabs">
        {PAGES.map((p) => (
          <button key={p.key} className={page === p.key ? 'active' : ''} onClick={() => setPage(p.key)}>
            {p.label}
          </button>
        ))}
      </nav>

      {page !== 'main' ? (
        <div className="page-view">
          {page === 'tools' && <ToolRagTab requestId={selected?.meta?.request_id} />}
          {page === 'monitor' && <MonitorTab />}
          {page === 'knowledge' && <KnowledgeTab />}
          {page === 'agents' && <AgentsTab />}
        </div>
      ) : (
        <main className="layout" ref={layoutRef}>
          {sidebarCollapsed ? (
            <div className="sidebar-rail">
              <button
                className="sidebar-expand"
                onClick={() => setSidebarCollapsed(false)}
                title="展开会话列表"
              >
                ☰
              </button>
            </div>
          ) : (
            <Sidebar
              sessions={sessions}
              currentSid={currentSid}
              onNew={handleNewSession}
              onSwitch={handleSwitchSession}
              onToggle={() => setSidebarCollapsed(true)}
            />
          )}
          {/* 左栏宽度用 CSS 变量传，媒体查询可以正常覆盖（inline flex 值会压过媒体查询） */}
          <div className="chat-wrap" style={{ '--chat-pct': `${chatPct}%` }}>
            <ChatPanel
              messages={messages}
              busy={busy}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onSend={send}
            />
          </div>
          <div className="resizer" onMouseDown={startDrag} title="拖动调整两栏宽度" />
          <TracePanel message={selected} />
        </main>
      )}
    </div>
  )
}
