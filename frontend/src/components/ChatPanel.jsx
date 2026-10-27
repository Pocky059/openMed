import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

// 意图显示名（后端枚举值 → 中文，方便演示时看懂）
const INTENT_NAMES = {
  medication_interaction: '药物相互作用',
  medication_info: '药品信息',
  symptom_check: '症状咨询',
  department_guide: '科室导诊',
  appointment: '预约挂号',
  medical_receipt: '凭证票据',
  emergency: '急症求助',
  human_handoff: '转人工',
  complaint: '投诉建议',
  greeting: '问候寒暄',
  query: '流程咨询',
  request: '业务请求',
  other: '其他',
}

export default function ChatPanel({ messages, busy, selectedId, onSelect, onSend }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // 新消息到达时滚到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = (e) => {
    e.preventDefault()
    if (busy || !input.trim()) return
    onSend(input.trim())
    setInput('')
    inputRef.current?.focus()
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>欢迎使用 OpenMed</h2>
            <p>试着问我：</p>
            <ul>
              <li>「在吃华法林，能喝银杏叶茶吗？」（多步检索 + 药物相互作用）</li>
              <li>「头痛三天了，该挂什么科？」（意图识别 + 科室导诊）</li>
              <li>「胸口压榨性疼痛、喘不上气」（急症升级，观察右侧面板）</li>
            </ul>
          </div>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={`msg-row ${m.role} ${selectedId === m.id ? 'selected' : ''}`}
            onClick={() => onSelect(m.id)}
          >
            <div className={`bubble ${m.error ? 'error' : ''}`}>
              {m.loading ? (
                <div className="thinking">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                  <span className="thinking-text">系统处理中（检索 + 生成，约 5-20 秒）</span>
                </div>
              ) : m.error ? (
                <p className="error-text">⚠️ {m.error}</p>
              ) : m.role === 'assistant' ? (
                <ReactMarkdown>{m.text}</ReactMarkdown>
              ) : (
                <p>{m.text}</p>
              )}
            </div>

            {/* 助手消息下方的元信息行：意图 / 路由 / 耗时 / 升级 */}
            {m.role === 'assistant' && m.meta && (
              <div className="msg-meta">
                <span className="chip">{INTENT_NAMES[m.meta.intent] || m.meta.intent}</span>
                <span className="chip">{m.meta.agent_type}</span>
                {m.meta.escalated && <span className="chip danger">已升级转人工</span>}
                {m.meta.urgency && m.meta.urgency !== 'LOW' && (
                  <span className={`chip ${m.meta.urgency === 'CRITICAL' ? 'danger' : 'warn'}`}>
                    紧急度 {m.meta.urgency}
                  </span>
                )}
                <span className="latency">{Math.round(m.meta.latency_ms)}ms</span>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form className="inputbar" onSubmit={submit}>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="描述你的症状或问题，回车发送…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? '处理中…' : '发送'}
        </button>
      </form>
    </section>
  )
}
