/** 会话列表边栏：新建会话 / 切换当前会话。
 *
 * 「新会话」= 同一个 user_id（web-demo）换一个新 conv_id：
 * 后端工作记忆/会话摘要按 (user_id, conv_id) 隔离，新会话干净开局；
 * 情景记忆（ChromaDB）按 user_id 跨会话召回——上个会话的病历摘要
 * 会在语义相关时注入新会话（见 memory/conversation_memory.py _search_episodic）。
 */
export default function Sidebar({ sessions, currentSid, onNew, onSwitch, onToggle }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <span>会话列表</span>
        <div className="sidebar-actions">
          <button className="new-session-btn" onClick={onNew} title="同一用户开新会话（不重置记忆）">
            ＋ 新建
          </button>
          <button className="collapse-btn" onClick={onToggle} title="收起边栏">
            «
          </button>
        </div>
      </div>
      <ul className="session-list">
        {sessions.map((s) => (
          <li
            key={s.id}
            className={`session-item ${s.id === currentSid ? 'active' : ''}`}
            onClick={() => onSwitch(s.id)}
            title={s.title}
          >
            <div className="session-title">{s.title}</div>
            <div className="session-time">创建于 {s.createdAt}</div>
          </li>
        ))}
      </ul>
    </aside>
  )
}
