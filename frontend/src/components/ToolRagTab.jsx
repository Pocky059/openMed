import { useEffect, useState } from 'react'
import { fetchToolTrace, fetchFullTrace } from '../api.js'

/** 长文本折叠（与 TracePanel 同款交互，组件自持一份）。 */
function LongText({ text, max = 200, cls = 'msg-text' }) {
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

/** 解析落盘轨迹里 tool_result 块的 content：后端落盘时是 JSON 字符串。 */
function parseToolResult(content) {
  if (typeof content === 'string') {
    try {
      return JSON.parse(content)
    } catch {
      return { raw: content }
    }
  }
  return content ?? {}
}

/** 一次 search_knowledge_base 调用的 RAG 链路结果：召回排名 + 三种分数。
 * score=混合召回融合分（RRF 后），rrf_score/rerank_score 为精排阶段的分数，
 * 字段缺失说明该次调用没走到对应阶段（如未精排），原样留空不造假。 */
function RagResults({ result }) {
  const results = result?.results || []
  if (results.length === 0) {
    return <p className="muted">（本次检索无命中结果）</p>
  }
  return (
    <div className="rag-hits">
      {results.map((doc, i) => (
        <div key={doc.id || i} className="rag-hit">
          <div className="rag-hit-head">
            <span className="hit-rank">#{i + 1}</span>
            <strong>{doc.title}</strong>
            {doc.chunk != null && <span className="muted">chunk {doc.chunk}</span>}
            <span className="rag-scores">
              <span title="混合召回融合分">融合 {Number(doc.score ?? 0).toFixed(3)}</span>
              {doc.rrf_score != null && <span title="RRF 融合分">RRF {Number(doc.rrf_score).toFixed(3)}</span>}
              {doc.rerank_score != null && <span title="CrossEncoder 精排分">精排 {Number(doc.rerank_score).toFixed(3)}</span>}
            </span>
          </div>
          <LongText text={doc.content} max={120} cls="rag-snippet" />
        </div>
      ))}
    </div>
  )
}

/** 单次工具调用卡片：状态徽章（成功/延迟/缓存/精排）+ 输入 + RAG 结果排名。 */
function ToolCallCard({ call, resultDetail, index }) {
  const failed = !call.success || call.error
  return (
    <div className={`tool-call-card ${failed ? 'failed' : ''}`}>
      <div className="tool-call-head">
        <span className="step-num">{index + 1}</span>
        <strong>{call.tool_name}</strong>
        <span className="muted">by {call.agent_type}</span>
        <span className="muted">{call.latency_ms}ms</span>
        <span className={`flag ${failed ? 'bad' : 'ok'}`}>{failed ? '失败' : '成功'}</span>
        {call.cached && <span className="flag warn">缓存命中</span>}
        {call.reranked && <span className="flag info">精排</span>}
        {call.rerank_degraded && <span className="flag bad">精排降级</span>}
      </div>
      {call.input?.query && (
        <p className="tool-query">查询词：<strong>{call.input.query}</strong>
          {call.input.top_k != null && <span className="muted"> · top_k={call.input.top_k}</span>}
        </p>
      )}
      {call.error && <p className="error-text">⚠️ {call.error}</p>}
      {resultDetail ? (
        resultDetail.results ? (
          <RagResults result={resultDetail} />
        ) : (
          <LongText text={JSON.stringify(resultDetail, null, 1)} cls="msg-json" max={200} />
        )
      ) : (
        <p className="muted">（轨迹里没有这次调用的返回明细——工具轨迹按 tool_use_id 关联落盘对话，非 RAG 工具可能只有输入）</p>
      )}
    </div>
  )
}

/** 工具 & RAG 选项卡：本次请求的每次工具调用 + RAG 链路的召回/精排明细。
 *
 * 两个数据源按 tool_use_id 关联：
 * - /trace/tool（内存）：latency/success/cached/reranked 等元信息
 * - /trace/full（落盘）：messages 里 tool_result 块的完整返回（含 results 排名与分数）
 */
export default function ToolRagTab({ requestId }) {
  const [toolTrace, setToolTrace] = useState(null)
  const [full, setFull] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setToolTrace(null)
    setFull(null)
    setError(null)
    if (!requestId) return
    fetchToolTrace(requestId)
      .then(setToolTrace)
      .catch((err) => setError(String(err.message || err)))
    fetchFullTrace(requestId)
      .then(setFull)
      .catch(() => {}) // 落盘轨迹拿不到不阻塞工具轨迹展示
  }, [requestId])

  if (!requestId) {
    return <p className="muted">先回「主界面」点选一条<strong>助手回复</strong>，这里会展示那次请求的工具调用与 RAG 检索链路。</p>
  }

  // 从完整轨迹里按 tool_use_id 索引 tool_result 明细
  const resultById = {}
  for (const msg of full?.messages || []) {
    for (const block of Array.isArray(msg.content) ? msg.content : []) {
      if (block?.type === 'tool_result' && block.tool_use_id) {
        resultById[block.tool_use_id] = parseToolResult(block.content)
      }
    }
  }

  const calls = toolTrace?.tool_calls || []

  return (
    <div className="tab-body">
      {error && <p className="error-text">⚠️ {error}</p>}
      <p className="muted">
        本次请求共 <strong>{calls.length}</strong> 次工具调用。
        RAG 链路 = 混合召回（BM25 + 向量）→ RRF 融合 → CrossEncoder 精排，
        每个状态徽章与三种分数都能在下面逐一核对。
      </p>
      {calls.length === 0 && <p className="muted">本次请求没有工具调用（未走检索）。</p>}
      {calls.map((call, i) => (
        <ToolCallCard key={call.tool_use_id || i} call={call} resultDetail={resultById[call.tool_use_id]} index={i} />
      ))}
    </div>
  )
}
