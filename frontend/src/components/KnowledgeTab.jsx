import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchKnowledgeDocs, fetchKnowledgeDoc, addKnowledge, uploadKnowledgeFile } from '../api.js'

/** 单篇文档条目：点「查看内容」懒加载全部片段（只读，没有修改/删除入口）。 */
function KbDoc({ doc }) {
  const [open, setOpen] = useState(false)
  const [detail, setDetail] = useState(null)
  const [loadError, setLoadError] = useState(null)

  const toggle = async () => {
    if (open) {
      setOpen(false)
      return
    }
    setOpen(true)
    if (detail) return // 已加载过，直接展开，不重复请求
    try {
      setDetail(await fetchKnowledgeDoc(doc.title))
      setLoadError(null)
    } catch (err) {
      setLoadError(String(err.message || err))
    }
  }

  return (
    <div className="kb-doc">
      <div className="kb-doc-head">
        <span className={`flag ${doc.source === 'builtin' ? 'info' : 'warn'}`}>
          {doc.source === 'builtin' ? '内置' : '用户上传'}
        </span>
        <strong>{doc.title}</strong>
        <span className="muted">{doc.chunks} 个片段</span>
        <button className="expand-btn" onClick={toggle}>
          {open ? '收起内容 ▲' : '查看内容 ▼'}
        </button>
      </div>
      {open && loadError && <p className="error-text">⚠️ {loadError}</p>}
      {open && detail && (
        <div className="kb-chunks">
          {detail.chunks.length === 0 && <p className="muted">该标题下没有片段</p>}
          {detail.chunks.map((c) => (
            <div key={c.index} className="kb-chunk">
              <span className="muted">片段 {c.index + 1}</span>
              <div className="kb-chunk-text">{c.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** 知识库选项卡：查看已有文档（内置/用户上传），添加新文档。
 * 内置文档只读（没有修改/删除入口——后端也没有这类端点，天然不可改动）。 */
export default function KnowledgeTab() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  // 文档来源筛选：全部 / 内置 / 用户上传
  const [filter, setFilter] = useState('all')
  const fileRef = useRef(null)

  const load = useCallback(async () => {
    try {
      setData(await fetchKnowledgeDocs())
      setError(null)
    } catch (err) {
      setError(String(err.message || err))
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const flash = (msg) => {
    setNotice(msg)
    setTimeout(() => setNotice(null), 4000)
  }

  const submitAdd = async () => {
    if (busy || !title.trim() || !content.trim()) return
    setBusy(true)
    try {
      const r = await addKnowledge([{ title: title.trim(), content: content.trim() }])
      flash(`已入库 ${r.added_chunks} 个片段`)
      setTitle('')
      setContent('')
      await load()
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setBusy(false)
    }
  }

  const submitUpload = async () => {
    const file = fileRef.current?.files?.[0]
    if (busy || !file) return
    setBusy(true)
    try {
      const r = await uploadKnowledgeFile(file)
      flash(`文件 ${file.name} 导入成功（${r.added_chunks} 个片段）`)
      fileRef.current.value = ''
      await load()
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setBusy(false)
    }
  }

  const docs = data?.docs || []
  const FILTERS = [
    { key: 'all', label: '全部' },
    { key: 'builtin', label: '内置' },
    { key: 'user', label: '用户上传' },
  ]
  const shown = docs.filter((d) => filter === 'all' || d.source === filter)

  return (
    <div className="tab-body">
      <p className="muted">
        文档入库时自动切片（500 字/片）并生成向量，检索时 BM25 + 向量混合召回。
        共 <strong>{data?.total_chunks ?? 0}</strong> 个片段 / <strong>{docs.length}</strong> 篇文档
        （内置 {data?.builtin_count ?? 0} · 用户上传 {data?.user_count ?? 0}）。
      </p>
      {error && <p className="error-text">⚠️ {error}</p>}
      {notice && <p className="notice-text">✓ {notice}</p>}

      {/* 添加文档：标题 + 正文（走 /knowledge/add 批量导入接口） */}
      <div className="card">
        <h3>添加文档</h3>
        <input
          className="kb-input"
          type="text"
          placeholder="文档标题（如：某药品说明书摘要）"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <textarea
          className="kb-textarea"
          rows={4}
          placeholder="文档正文（入库时自动切片）…"
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        <div className="kb-actions">
          <button className="mini-btn primary" onClick={submitAdd} disabled={busy || !title.trim() || !content.trim()}>
            添加并入库
          </button>
          <span className="muted">或上传文件：</span>
          <input ref={fileRef} type="file" accept=".txt,.md,.json" onChange={submitUpload} disabled={busy} />
        </div>
      </div>

      {/* 文档列表：内置只读、用户上传可辨认 */}
      <div className="card">
        <h3>已入库文档</h3>
        {/* 按标签筛选：点「内置」就只看内置文档，列表更整齐 */}
        <div className="kb-filters">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              className={`mini-btn ${filter === f.key ? 'primary' : ''}`}
              onClick={() => setFilter(f.key)}
            >
              {f.label}（{f.key === 'all' ? docs.length : docs.filter((d) => d.source === f.key).length}）
            </button>
          ))}
        </div>
        {shown.map((doc) => (
          <KbDoc key={doc.title} doc={doc} />
        ))}
        {shown.length === 0 && <p className="muted">{docs.length === 0 ? '知识库为空' : '该分类下暂无文档'}</p>}
      </div>
    </div>
  )
}
