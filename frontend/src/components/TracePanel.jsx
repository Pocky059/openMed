import { useEffect, useState } from 'react'
import { fetchToolTrace, fetchFullTrace, searchKnowledge } from '../api.js'

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

// 轨迹消息 kind 的中文名与徽标色（与后端 classify_messages 的 kind 规则一一对应）
const KIND_NAMES = {
  system: '系统提示词',
  context: '背景注入',
  ack: '背景确认',
  query: '用户提问',
  tool_call: '调用工具',
  tool_result: '工具返回',
  final: '最终回答',
}
const KIND_FLAG = {
  system: 'info',
  context: 'info',
  ack: 'warn',
  query: 'ok',
  tool_call: 'info',
  tool_result: 'warn',
  final: 'ok',
}

/** [背景信息]/[结构化实体]/[角色输入契约] 三个占位前缀在 classify_messages 里
 * 都标为 kind=context，但语义完全不同——按内容前缀细分标签，别都叫「背景注入」。 */
function contextLabel(content) {
  const s = String(content ?? '')
  if (s.startsWith('[背景信息]')) return '记忆上下文'
  if (s.startsWith('[结构化实体]')) return '实体注入'
  if (s.startsWith('[角色输入契约]')) return '角色输入契约'
  return KIND_NAMES.context
}

/** 截断长文本：system prompt / 检索结果动辄上千字，调试面板里看头尾即可。 */
function clip(text, max = 600) {
  const s = String(text ?? '')
  if (s.length <= max) return s
  return s.slice(0, max) + ` …（已截断，共 ${s.length} 字）`
}

/** 长文本折叠：默认收起只显示前 max 字，点击「展开/收起」看全文。
 * 与 clip 的区别：内容完整可查（clip 直接砍掉后半段），面板又不会被大段文案占满。 */
function LongText({ text, max = 240, cls = 'msg-text' }) {
  const [open, setOpen] = useState(false)
  const s = String(text ?? '')
  const long = s.length > max
  const shown = long && !open ? s.slice(0, max) + ' …' : s
  return (
    <div className="long-text">
      <div className={`${cls}${long && open ? ' open' : ''}`}>{shown}</div>
      {long && (
        <button className="expand-btn" onClick={() => setOpen((v) => !v)}>
          {open ? '收起 ▲' : `点击展开（共 ${s.length} 字）▼`}
        </button>
      )}
    </div>
  )
}

/** 对话历史小列表（意图识别与编排器的输入里都有 history）。
 * 结构化记忆（{role, content} 列表）单独成块，与记忆上下文的块对齐，
 * 避免 user:/assistant: 裸行堆在外面。 */
function HistoryList({ history, title = '结构化记忆' }) {
  if (!history || history.length === 0) return <p className="muted">无对话历史</p>
  return (
    <div className="ctx-section mem-block">
      <p className="ctx-title">{title}（最近 {history.length} 条）</p>
      <ul className="history-list">
        {history.map((h, i) => (
          <li key={i}>
            <span className={`msg-role role-${h.role}`}>{h.role}</span> {h.content}
          </li>
        ))}
      </ul>
    </div>
  )
}

/** JSON 字符串还原成缩进格式（用户画像是 json.dumps 单行，读不动）。 */
function prettyIfJson(s) {
  try {
    return JSON.stringify(JSON.parse(s), null, 1)
  } catch {
    return s
  }
}

/** [背景信息] 记忆上下文按段拆分：会话摘要/相关历史/用户画像/最近对话
 * （与 memory/conversation_memory.py to_prompt_text 的四段一一对应）。
 * 后端本来存的就是全文——默认折叠显示、点击展开看完整内容，不丢信息。 */
function ContextSections({ content }) {
  const text = String(content ?? '').replace(/^\[背景信息\]\s*\n*/, '')
  const parts = text.split(/\n(?=\[)/) // 每段以行首 [ 开头，按此切分
  return (
    <div className="ctx-sections">
      {parts.map((p, i) => {
        const m = p.match(/^\[([^\]]+)\]\s*/)
        const title = m ? m[1] : `片段 ${i + 1}`
        const body = p.replace(/^\[[^\]]+\]\s*/, '')
        return (
          <div className="ctx-section" key={i}>
            <p className="ctx-title">{title}</p>
            <LongText text={prettyIfJson(body)} cls="ctx-body" max={240} />
          </div>
        )
      })}
    </div>
  )
}

/** 消息内容块：纯文本直接展示；tool_use / tool_result 结构化展示。 */
function ContentBlock({ block }) {
  const type = block?.type
  if (type === 'text') return <LongText text={block.text} />
  if (type === 'tool_use') {
    return (
      <div className="msg-block">
        <p className="msg-text">调用工具 <strong>{block.name}</strong></p>
        <LongText text={JSON.stringify(block.input ?? {}, null, 1)} cls="msg-json" max={240} />
      </div>
    )
  }
  if (type === 'tool_result') {
    let text = block.content
    if (typeof text === 'string') {
      try {
        text = JSON.stringify(JSON.parse(text), null, 1) // 落盘时是 JSON 字符串，还原成格式化 JSON
      } catch {
        /* 不是 JSON 就原样展示 */
      }
    } else {
      text = JSON.stringify(text ?? {}, null, 1)
    }
    return (
      <div className="msg-block">
        <p className="msg-text muted">
          工具返回{block.tool_use_id ? `（${block.tool_use_id}）` : ''}
        </p>
        <LongText text={text} cls="msg-json" max={400} />
      </div>
    )
  }
  return <LongText text={JSON.stringify(block)} />
}

/** LLM 对话里的一条消息：序号 + role + agent_type + kind 徽标 + 内容。
 * 按 kind 上色条（system 橙 / query 绿 / final 蓝），层级一眼可辨。 */
function ConversationEntry({ msg, idx }) {
  const blocks = Array.isArray(msg.content) ? msg.content : null
  const label = msg.kind === 'context' ? contextLabel(msg.content) : KIND_NAMES[msg.kind] || msg.kind
  return (
    <div className={`msg-entry kind-${msg.kind || 'unknown'}`}>
      <div className="msg-head">
        {idx != null && <span className="msg-seq">{idx}</span>}
        <span className={`msg-role role-${msg.role}`}>{msg.role}</span>
        {msg.agent_type && <span className="msg-agent">{msg.agent_type}</span>}
        {msg.kind && (
          <span className={`flag ${KIND_FLAG[msg.kind] || 'info'}`}>{label}</span>
        )}
      </div>
      {blocks ? (
        blocks.map((b, i) => <ContentBlock key={i} block={b} />)
      ) : typeof msg.content === 'string' ? (
        label === '记忆上下文' ? (
          <ContextSections content={msg.content} />
        ) : (
          <LongText text={msg.content} max={300} />
        )
      ) : (
        <p className="muted">（无内容）</p>
      )}
    </div>
  )
}

/** 取 messages 里最后一条 kind=final 的文本（编排器输出的最终回答）。 */
function finalAnswerText(messages) {
  for (let i = (messages?.length || 0) - 1; i >= 0; i--) {
    const m = messages[i]
    if (m?.kind !== 'final') continue
    if (typeof m.content === 'string') return m.content
    if (Array.isArray(m.content)) {
      return m.content
        .filter((b) => b?.type === 'text')
        .map((b) => b.text)
        .join('')
    }
  }
  return ''
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

/** 检查面板（主界面右侧）：展示选中消息的系统内部行为。 */
export default function TracePanel({ message }) {
  const [trace, setTrace] = useState(null)
  const [traceLoading, setTraceLoading] = useState(false)
  const [full, setFull] = useState(null)
  const [fullLoading, setFullLoading] = useState(false)
  const [fullOpen, setFullOpen] = useState(true)

  const meta = message?.meta

  // 选中消息变化时拉取工具轨迹 + 完整轨迹（落盘文件，重启不丢）
  useEffect(() => {
    setTrace(null)
    setFull(null)
    if (!meta?.request_id) return
    setTraceLoading(true)
    setFullLoading(true)
    fetchToolTrace(meta.request_id).then((t) => {
      setTrace(t)
      setTraceLoading(false)
    })
    fetchFullTrace(meta.request_id).then((t) => {
      setFull(t)
      setFullLoading(false)
    })
  }, [message?.id, meta?.request_id])

  if (!message) {
    return (
      <aside className="inspector empty">
        <h2>内部过程面板</h2>
        <p>发送一条消息，点选左侧的助手回复，这里会展示系统这次请求的内部过程：</p>
        <ul>
          <li>完整轨迹：意图识别输入/输出、编排器输入/输出、拼给 LLM 的完整对话</li>
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

      {/* 完整轨迹（/trace/full 落盘文件）：放最下方，需要深挖时展开 */}
      <div className="card">
        <h3 className="full-trace-head" onClick={() => setFullOpen((v) => !v)}>
          完整轨迹（落盘）
          <span className="muted toggle">{fullOpen ? '收起 ▲' : '展开 ▼'}</span>
        </h3>
        {fullOpen && (
          <>
            {fullLoading && <p className="muted">加载中…</p>}
            {!fullLoading && full === null && (
              <p className="muted">未找到落盘轨迹（发消息后等几秒再点选，后台写盘是异步的）</p>
            )}
            {!fullLoading && full && (
              <div className="full-trace">
                {/* 意图识别：输入（消息 + 历史）与输出（8 字段全量） */}
                <div className="sub">
                  <h4>意图识别 · 输入</h4>
                  <p className="msg-text">用户消息：{clip(full.meta?.intent_detail?.input?.message, 300)}</p>
                  <HistoryList history={full.meta?.intent_detail?.input?.history} />
                  <h4>意图识别 · 输出</h4>
                  {full.meta?.intent_detail?.output ? (
                    <div className="kv">
                      <p>
                        意图 <strong>{INTENT_NAMES[full.meta.intent_detail.output.intent] || full.meta.intent_detail.output.intent}</strong>
                        <span className="muted"> · 分组 {full.meta.intent_detail.output.intent_group}</span>
                        <span className="muted"> · 置信度 {(full.meta.intent_detail.output.confidence ?? 0).toFixed(3)}</span>
                        <span className="muted"> · 紧急度 {full.meta.intent_detail.output.urgency}</span>
                        <span className="muted"> · 识别耗时 {full.meta.intent_detail.output.latency_ms}ms</span>
                      </p>
                      {full.meta.intent_detail.output.reasoning && (
                        <p className="muted reason">推理：{full.meta.intent_detail.output.reasoning}</p>
                      )}
                      {full.meta.intent_detail.output.entities &&
                        Object.keys(full.meta.intent_detail.output.entities).length > 0 && (
                          <p className="muted">
                            实体：
                            {Object.entries(full.meta.intent_detail.output.entities)
                              .map(([k, v]) => `${k}=${v.join('/')}`)
                              .join('，')}
                          </p>
                        )}
                      {full.meta.intent_detail.output.source_scores &&
                        Object.keys(full.meta.intent_detail.output.source_scores).length > 0 && (
                          <div className="kv-scores">
                            {SOURCE_META.filter((s) => s.key in full.meta.intent_detail.output.source_scores).map((s) => (
                              <ScoreBar
                                key={s.key}
                                label={s.label}
                                sub={`权重 ${s.weight}`}
                                value={full.meta.intent_detail.output.source_scores[s.key]}
                                color={s.key === 'llm' ? '#4cc9f0' : s.key === 'embedding' ? '#7bd88f' : '#f4a261'}
                              />
                            ))}
                          </div>
                        )}
                    </div>
                  ) : (
                    <p className="muted">无意图识别明细（调用方预设了 intent，跳过识别）</p>
                  )}
                </div>

                {/* 编排器：输入快照与路由输出 */}
                <div className="sub">
                  <h4>编排器 · 输入</h4>
                  <p className="msg-text">用户消息：{clip(full.meta?.input?.message, 300)}</p>
                  {full.meta?.input?.context && (
                    <ContextSections content={full.meta.input.context} />
                  )}
                  {full.meta?.input?.entities && Object.keys(full.meta.input.entities).length > 0 && (
                    <p className="muted">
                      实体：
                      {Object.entries(full.meta.input.entities)
                        .map(([k, v]) => `${k}=${v.join('/')}`)
                        .join('，')}
                    </p>
                  )}
                  <HistoryList history={full.meta?.input?.history} />
                  <h4>编排器 · 输出</h4>
                  <div className="kv">
                    <p>
                      主 Agent <strong>{full.meta?.primary_agent}</strong>
                      {full.meta?.agent_types?.length > 1 && (
                        <span className="muted"> · 全部参与 {full.meta.agent_types.join(', ')}</span>
                      )}
                      <span className="muted"> · 路由置信度 {(full.meta?.routing_confidence ?? 0).toFixed(3)}</span>
                      <span className="muted"> · 升级 {full.meta?.escalated ? '是' : '否'}</span>
                      <span className="muted"> · 总耗时 {Math.round(full.meta?.latency_ms)}ms</span>
                    </p>
                    {full.meta?.routing_reason && (
                      <p className="muted reason">路由理由：{full.meta.routing_reason}</p>
                    )}
                  </div>
                  {/* 最终回答独立成块（与记忆上下文/结构化记忆同款视觉），
                      不再和其他路由字段堆在 kv 里 */}
                  <div className="ctx-section final-block">
                    <p className="ctx-title">最终回答（Agent 输出）</p>
                    <LongText text={finalAnswerText(full.messages)} cls="ctx-body" max={240} />
                  </div>
                </div>

                {/* 拼给 LLM 的完整对话（含 system prompt，按时间顺序） */}
                <div className="sub">
                  <h4>拼给 LLM 的完整对话（{full.messages?.length || 0} 条消息）</h4>
                  {(full.messages || []).map((m, i) => (
                    <ConversationEntry key={i} msg={m} idx={i + 1} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  )
}
