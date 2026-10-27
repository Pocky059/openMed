import { useCallback, useEffect, useState } from 'react'
import { fetchMonitor, fetchMetrics } from '../api.js'

/** 告警级别 → 色条样式。severity 来自后端 Alert（info/warn/critical 等）。 */
const SEVERITY_CLS = {
  critical: 'sev-critical',
  error: 'sev-critical',
  warning: 'sev-warn',
  warn: 'sev-warn',
  info: 'sev-info',
}

/** Agent 统计表（后端 get_stats：每实例 total/成功率/均耗/路由评分/监控惩罚）。 */
function AgentStatsTable({ agentStats }) {
  const rows = Object.entries(agentStats || {})
  if (rows.length === 0) return <p className="muted">暂无 Agent 统计</p>
  return (
    <table className="mon-table">
      <thead>
        <tr><th>Agent</th><th>角色</th><th>调用数</th><th>成功率</th><th>平均耗时</th><th>路由评分</th><th>监控惩罚</th></tr>
      </thead>
      <tbody>
        {rows.map(([key, s]) => (
          <tr key={key}>
            <td><strong>{key}</strong></td>
            <td>{s.role}</td>
            <td>{s.total}</td>
            <td className={s.success_rate < 0.8 ? 'num bad' : 'num'}>{(s.success_rate * 100).toFixed(1)}%</td>
            <td className="num">{s.avg_ms}ms</td>
            <td className="num">{s.routing_score}</td>
            <td className="num">{s.monitor_penalty}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** 工具统计表（后端 get_stats：按工具名分组的调用统计 + 熔断器状态）。 */
function ToolStatsTable({ toolStats }) {
  const rows = Object.entries(toolStats || {})
  if (rows.length === 0) return <p className="muted">暂无工具统计</p>
  return (
    <table className="mon-table">
      <thead>
        <tr><th>工具</th><th>调用数</th><th>成功率</th><th>平均延迟</th><th>连续失败</th><th>熔断器</th></tr>
      </thead>
      <tbody>
        {rows.map(([name, s]) => (
          <tr key={name}>
            <td><strong>{name}</strong></td>
            <td>{s.total}</td>
            <td className={s.success_rate < 0.8 ? 'num bad' : 'num'}>{(s.success_rate * 100).toFixed(1)}%</td>
            <td className="num">{s.avg_latency_ms}ms</td>
            <td className={s.consecutive_fails > 0 ? 'num bad' : 'num'}>{s.consecutive_fails}</td>
            <td>{s.circuit_state}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** 监控选项卡：/monitor 摘要（10 秒轮询）+ /metrics 原始指标。
 * 给面试演示「监测链路」：业务层监控摘要与 Prometheus 抓取口径同屏。 */
export default function MonitorTab() {
  const [monitor, setMonitor] = useState(null)
  const [metrics, setMetrics] = useState('')
  const [metricsOpen, setMetricsOpen] = useState(false)
  const [error, setError] = useState(null)
  const [lastRefresh, setLastRefresh] = useState(null)

  const load = useCallback(async () => {
    try {
      const [m, raw] = await Promise.all([fetchMonitor(), fetchMetrics()])
      setMonitor(m)
      setMetrics(raw)
      setError(null)
    } catch (err) {
      setError(String(err.message || err))
    }
    setLastRefresh(new Date())
  }, [])

  useEffect(() => {
    load()
    const timer = setInterval(load, 10000) // 10 秒轮询，与后端 Monitor 采集间隔一致
    return () => clearInterval(timer)
  }, [load])

  const alerts = monitor?.active_alerts || []
  const suggestions = monitor?.suggestions || []

  return (
    <div className="tab-body">
      <div className="mon-head">
        <p className="muted">
          数据来自 <code>/monitor</code>（业务监控摘要，每 10 秒自动刷新）与
          <code> /metrics</code>（Prometheus 抓取口径的原始指标）。
          {lastRefresh && <span> · 上次刷新 {lastRefresh.toLocaleTimeString()}</span>}
        </p>
        <button className="mini-btn" onClick={load}>立即刷新</button>
      </div>
      {error && <p className="error-text">⚠️ {error}</p>}

      {/* 活动告警：超过阈值（成功率/延迟/连续失败）才会出现 */}
      <div className="card">
        <h3>活动告警（{alerts.length}）</h3>
        {alerts.length === 0 && <p className="muted ok-text">✓ 当前没有触发阈值的告警</p>}
        {alerts.map((a, i) => (
          <div key={i} className={`alert-card ${SEVERITY_CLS[a.severity] || 'sev-info'}`}>
            <span className="flag">{a.severity}</span>
            <strong>{a.message}</strong>
            <span className="muted">
              指标 {a.metric} = {a.value}（阈值 {a.threshold}）
            </span>
          </div>
        ))}
      </div>

      {/* Agent 运行统计（Monitor 每 10 秒采集一次编排器指标） */}
      <div className="card">
        <h3>Agent 运行统计</h3>
        <AgentStatsTable agentStats={monitor?.agent_stats} />
      </div>

      {/* 工具统计（含缓存命中与熔断器状态） */}
      <div className="card">
        <h3>工具统计</h3>
        <ToolStatsTable toolStats={monitor?.tool_stats} />
      </div>

      {/* 优化建议（Monitor 对指标的自动化解读） */}
      {suggestions.length > 0 && (
        <div className="card">
          <h3>优化建议</h3>
          {suggestions.map((s, i) => (
            <div key={i} className="suggestion">
              <p><strong>{s.title}</strong> <span className={`flag ${s.priority > 0.6 ? 'bad' : 'warn'}`}>优先级 {Math.round(s.priority * 100)}%</span></p>
              <p className="muted">{s.action}</p>
            </div>
          ))}
        </div>
      )}

      {/* Prometheus 原始指标：默认折叠，需要时展开 */}
      <div className="card">
        <h3 className="collapse-head" onClick={() => setMetricsOpen((v) => !v)}>
          Prometheus 原始指标 <span className="muted toggle">{metricsOpen ? '收起 ▲' : '展开 ▼'}</span>
        </h3>
        {metricsOpen && <pre className="metrics-pre">{metrics}</pre>}
      </div>
    </div>
  )
}
