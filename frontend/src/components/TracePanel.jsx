import { useEffect, useState } from 'react'
import { fetchToolTrace, searchKnowledge } from '../api.js'

// 三路投票的权重（与后端 intent_recognizer._vote 一致：LLM 70% / 语义 20% / 关键词 10%）
const SOURCE_META = [
  { key: 'llm', label: 'LLM 判断', weight: '70%' },
  { key: 'embedding', label: '语义向量', weight: '20%' },
  { key: 'pattern', label: '关键词', weight: '10%' },
]

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

/** 单个工具调用的「复现检索」：用同样的查询词打 /search，展示命中文档。 */
function ReplaySearch({ query }) {
  const [open, setOpen] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const toggle = async () => {
    setOpen((prev) => !prev)
    if (open || data || loading) return
    setLoading(true)
    try {
      setData(await searchKnowledge(query, 5))
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="replay">
      <button className="replay-btn" onClick={toggle}>
        {open ? '收起检索结果' : '复现检索（看命中了什么）'}
      </button>
      {open && loading && <p className="muted">检索中…</p>}
      {open && error && <p className="muted">⚠️ {error}</p>}
      {open && data && (
        <ol className="hits">
          {data.results.map((doc, i) => (
            <li key={doc.id || i}>
              <div className="hit-head">
                <span className="hit-rank">#{i + 1}</span>
                <strong>{doc.title}</strong>
                <span className="hit-score">{Number(doc.score).toFixed(3)}</span>
              </div>
              <p className="hit-snippet">
                {doc.content ? doc.content.slice(0, 120) + '…' : ''}
              </p>
            </li>
          ))}
          {data.results.length === 0 && <p className="muted">没有命中任何文档</p>}
        </ol>
      )}
    </div>
  )
}

/** 分数条：一条横向进度条 + 数值。 */
function ScoreBar({ label, sub, value, color }) {
  return (
    <div className="score-row">
      <div className="score-label">
        <span>{label}</span>
        {sub && <span className="score-sub">{sub}</span>}
      </div>
      <div className="score-bar">
        <div
          className="score-fill"
          style={{ width: `${Math.min(100, Math.round((value || 0) * 100))}%`, background: color }}
        />
      </div>
      <span className="score-num">{(value ?? 0).toFixed(3)}</span>
    </div>
  )
}

/** 内部过程面板：展示选中消息的系统内部行为。 */
export default function TracePanel({ message }) {
  const [trace, setTrace] = useState(null)
  const [traceLoading, setTraceLoading] = useState(false)

  const meta = message?.meta

  // 选中消息变化时拉取工具轨迹
  useEffect(() => {
    setTrace(null)
    if (!meta?.request_id) return
    setTraceLoading(true)
    fetchToolTrace(meta.request_id).then((t) => {
      setTrace(t)
      setTraceLoading(false)
    })
  }, [message?.id, meta?.request_id])

  if (!message) {
    return (
      <aside className="inspector empty">
        <h2>内部过程面板</h2>
        <p>发送一条消息，点选左侧的助手回复，这里会展示系统这次请求的内部过程：</p>
        <ul>
          <li>意图识别：三路投票（LLM 70% / 语义 20% / 关键词 10%）</li>
          <li>路由决策：派给哪个 Agent、置信度多少</li>
          <li>工具轨迹：每步检索的查询词、状态、耗时，可复现检索</li>
        </ul>
      </aside>
    )
  }

  if (message.role === 'user') {
    return (
      <aside className="inspector empty">
        <h2>内部过程面板</h2>
        <p>这是你的消息「{message.text}」。点选右侧这条消息对应的<strong>助手回复</strong>，即可看到系统处理它的内部过程。</p>
      </aside>
    )
  }

  if (!meta) {
    return (
      <aside className="inspector empty">
        <h2>内部过程面板</h2>
        {message.error ? <p className="error-text">⚠️ 请求失败：{message.error}</p> : <p>（处理中…）</p>}
      </aside>
    )
  }

  const toolCalls = trace?.tool_calls || []

  return (
    <aside className="inspector">
      <h2>内部过程 <span className="muted">request {meta.request_id.slice(0, 8)}</span></h2>

      {/* 意图识别 */}
      <div className="card">
        <h3>① 意图识别</h3>
        <div className="intent-line">
          <span className="intent-badge">{INTENT_NAMES[meta.intent] || meta.intent}</span>
          <span className="muted">分组 {meta.intent_group}</span>
          <span className="muted">融合置信度 {(meta.intent_confidence ?? 0).toFixed(3)}</span>
        </div>
        {meta.intent_source_scores &&
          SOURCE_META.map((s) => (
            <ScoreBar
              key={s.key}
              label={s.label}
              sub={`权重 ${s.weight}`}
              value={meta.intent_source_scores[s.key]}
              color={s.key === 'llm' ? '#4cc9f0' : s.key === 'embedding' ? '#7bd88f' : '#f4a261'}
            />
          ))}
      </div>

      {/* 路由决策 */}
      <div className="card">
        <h3>② 路由决策</h3>
        <p className="agent-line">
          <strong>{meta.agent_type}</strong>
          {meta.supporting_agents?.length > 0 && (
            <span className="muted"> + 协作 {meta.supporting_agents.join(', ')}</span>
          )}
        </p>
        <ScoreBar label="路由置信度" value={meta.routing_confidence} color="#4cc9f0" />
        {meta.routing_reason && <p className="muted reason">{meta.routing_reason}</p>}
        {meta.entities && Object.keys(meta.entities).length > 0 && (
          <p className="muted reason">
            结构化实体：
            {Object.entries(meta.entities).map(([k, v]) => `${k}=${v.join('/')}`).join('，')}
          </p>
        )}
      </div>

      {/* 安全 / 紧急度 */}
      {(meta.escalated || (meta.urgency && meta.urgency !== 'LOW')) && (
        <div className={`card ${meta.escalated ? 'alert' : 'warncard'}`}>
          <h3>③ 安全门控</h3>
          <p>
            紧急级别 <strong>{meta.urgency}</strong>
            {meta.escalated ? '，已升级转人工（触发 webhook 工单）' : '，未升级'}
          </p>
        </div>
      )}

      {/* 工具轨迹 */}
      <div className="card">
        <h3>{meta.escalated || (meta.urgency && meta.urgency !== 'LOW') ? '④' : '③'} 工具调用轨迹</h3>
        {traceLoading && <p className="muted">加载轨迹中…</p>}
        {!traceLoading && trace === null && <p className="muted">本次请求没有工具轨迹（未检索）</p>}
        {!traceLoading && toolCalls.length === 0 && trace !== null && (
          <p className="muted">工具轨迹为空</p>
        )}
        {toolCalls.map((call, i) => (
          <div key={call.tool_use_id || i} className="tool-step">
            <div className="tool-head">
              <span className="step-num">{i + 1}</span>
              <strong>{call.tool_name}</strong>
              <span className="muted">{call.latency_ms}ms</span>
              <span className={`flag ${call.success && call.result_success ? 'ok' : 'bad'}`}>
                {call.error ? '失败' : '成功'}
              </span>
              {call.cached && <span className="flag warn">缓存命中</span>}
              {call.reranked && <span className="flag info">精排</span>}
            </div>
            {call.input?.query && (
              <p className="tool-query">查询词：{call.input.query}</p>
            )}
            {call.input?.query && <ReplaySearch query={call.input.query} />}
          </div>
        ))}
      </div>

      {/* 汇总 */}
      <div className="card summary">
        <h3>本次请求</h3>
        <p className="muted">
          总耗时 {Math.round(meta.latency_ms)}ms
          {meta.knowledge_used ? ' · 使用了知识库' : ' · 未检索知识库'}
        </p>
      </div>
    </aside>
  )
}
