import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchAgentsCatalog, fetchSkills, uploadSkillFile } from '../api.js'

/** 长文本折叠（组件自持一份，与 TracePanel 同款交互）。 */
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

/** 契约条目列表（workflow/输入契约/输出契约/升级条件/工具范围共用）。 */
function ContractList({ title, items }) {
  if (!items || items.length === 0) return null
  return (
    <div className="contract">
      <p className="contract-title">{title}</p>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

/** 单个工具定义：名字 + 描述 + 参数 schema（JSON，折叠）。 */
function ToolDef({ tool }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="tool-def">
      <button className="expand-btn" onClick={() => setOpen((v) => !v)}>
        {open ? '▼' : '▶'} <strong>{tool.name}</strong>
      </button>
      <p className="muted tool-desc">{tool.description}</p>
      {open && <LongText text={JSON.stringify(tool.input_schema, null, 1)} cls="msg-json" max={240} />}
    </div>
  )
}

/** 单个 Agent 卡片：角色契约 + system prompt + 工具白名单（含参数 schema）。 */
function AgentCard({ agent }) {
  const [showPrompt, setShowPrompt] = useState(false)
  return (
    <div className="agent-card">
      <div className="agent-head">
        <span className="agent-badge">{agent.agent_type}</span>
        <strong>{agent.role}</strong>
        <span className="muted">
          {agent.model} · temperature {agent.temperature} · max_tokens {agent.max_tokens}
        </span>
      </div>
      <p className="agent-mission">{agent.mission}</p>

      <div className="contracts">
        <ContractList title="处理流程 workflow" items={agent.workflow} />
        <ContractList title="输入契约" items={agent.input_contract} />
        <ContractList title="输出契约" items={agent.output_contract} />
        <ContractList title="升级条件" items={agent.handoff_conditions} />
        <ContractList title="工具范围声明" items={agent.tool_scope} />
      </div>

      {/* system prompt：每次请求都拼进 LLM 对话的固定人设，默认折叠 */}
      <div className="agent-prompt">
        <button className="expand-btn" onClick={() => setShowPrompt((v) => !v)}>
          {showPrompt ? '收起 system prompt ▲' : '查看 system prompt ▼'}
        </button>
        {showPrompt && <LongText text={agent.system_prompt} max={400} cls="msg-json" />}
      </div>

      {/* 工具白名单：真实可调用的工具（急症 Agent 只有共享 RAG 工具） */}
      <div className="agent-tools">
        <p className="contract-title">工具（{agent.tools.length}）</p>
        {agent.tools.length === 0 && <p className="muted">无工具（急症 Agent 硬编码响应，不走 LLM）</p>}
        {agent.tools.map((tool) => (
          <ToolDef key={tool.name} tool={tool} />
        ))}
      </div>
    </div>
  )
}

/** Skill 卡片：来源徽章（内置不可改动/用户上传）+ 完整内容查看。 */
function SkillCard({ skill, agentTypes }) {
  const [open, setOpen] = useState(false)
  const targetAgents = skill.agents?.length
    ? skill.agents
    : agentTypes // agents 为空 = 适用于所有 Agent
  return (
    <div className="skill-card">
      <div className="skill-head">
        <span className={`flag ${skill.source === 'builtin' ? 'info' : 'warn'}`}>
          {skill.source === 'builtin' ? '内置' : '用户上传'}
        </span>
        <strong>{skill.name}</strong>
        <span className="muted">适用：{targetAgents.join(', ')}</span>
        {!skill.enabled && <span className="flag bad">已停用</span>}
      </div>
      {skill.description && <p className="muted skill-desc">{skill.description}</p>}
      {skill.keywords?.length > 0 && (
        <p className="muted">触发关键词：{skill.keywords.join('、')}</p>
      )}
      <button className="expand-btn" onClick={() => setOpen((v) => !v)}>
        {open ? '收起内容 ▲' : '查看内容 ▼'}
      </button>
      {open && <LongText text={skill.content} max={400} cls="skill-content" />}
    </div>
  )
}

/** Agent 管理选项卡：Agent 定义与工具白名单 + Skill 列表与上传。 */
export default function AgentsTab() {
  const [catalog, setCatalog] = useState(null)
  const [skillsData, setSkillsData] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  // 上传 Skill 时选择的适用 Agent：空 = 公用（适用所有 Agent）
  const [uploadAgents, setUploadAgents] = useState([])
  const fileRef = useRef(null)

  const load = useCallback(async () => {
    try {
      const [agents, skills] = await Promise.all([fetchAgentsCatalog(), fetchSkills(true)])
      setCatalog(agents)
      setSkillsData(skills)
      setError(null)
    } catch (err) {
      setError(String(err.message || err))
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const toggleUploadAgent = (t) =>
    setUploadAgents((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]))

  const submitUpload = async () => {
    const file = fileRef.current?.files?.[0]
    if (busy || !file) return
    setBusy(true)
    try {
      const r = await uploadSkillFile(file, uploadAgents)
      const scope = uploadAgents.length ? `（仅 ${uploadAgents.join(', ')}）` : '（公用）'
      setNotice(`Skill「${r.skill?.name}」上传成功，已热加载生效 ${scope}`)
      fileRef.current.value = ''
      setUploadAgents([])
      await load()
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setBusy(false)
    }
  }

  const agents = catalog?.agents || []
  const agentTypes = agents.map((a) => a.agent_type)
  const skills = skillsData?.skills || []

  return (
    <div className="tab-body">
      {error && <p className="error-text">⚠️ {error}</p>}
      {notice && <p className="notice-text">✓ {notice}</p>}

      <div className="card">
        <h3>Agent 列表（{agents.length}）</h3>
        <p className="muted">
          多 Agent 编排：意图识别后路由到主 Agent（可带协作 Agent），
          每个 Agent 有独立角色契约、工具白名单与生成参数。
        </p>
        <div className="agents-grid">
          {agents.map((agent) => (
            <AgentCard key={agent.agent_type} agent={agent} />
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Skills（{skills.length}）</h3>
        <p className="muted">
          Skill 是热加载的业务能力说明：命中关键词/适用 Agent 时动态拼进 system prompt。
          内置 Skill 不可改动；上传同名 Skill 会被后端拒绝。
        </p>
        {skills.map((skill) => (
          <SkillCard key={`${skill.source}-${skill.name}`} skill={skill} agentTypes={agentTypes} />
        ))}
        {skills.length === 0 && <p className="muted">暂无 Skill</p>}

        <div className="skill-upload">
          <p className="contract-title">上传新 Skill（.md / .txt / .json）</p>
          {/* 适用范围：不选 = 公用；勾选 = 只注入对应 Agent（后端写进文件的 agents 字段） */}
          <div className="skill-agents">
            <span className="muted">适用 Agent：</span>
            {agentTypes.map((t) => (
              <label key={t} className="agent-check">
                <input type="checkbox" checked={uploadAgents.includes(t)} onChange={() => toggleUploadAgent(t)} />
                {t}
              </label>
            ))}
            <span className="muted">
              {uploadAgents.length === 0 ? '（不选 = 公用，适用所有 Agent）' : `（仅 ${uploadAgents.join('、')}）`}
            </span>
          </div>
          <input ref={fileRef} type="file" accept=".md,.txt,.json" onChange={submitUpload} disabled={busy} />
          <p className="muted">
            格式示例（.md 用 front matter，agents 一行可留空、由上面勾选决定）：
            <code>{`---\nname: 新话术\nkeywords: 关键词1, 关键词2\n---\n正文内容…`}</code>
          </p>
        </div>
      </div>
    </div>
  )
}
