import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type AskResponse, type DocumentOut } from './api'
import { formatBytes, parseCitations } from './citations'

const POLL_MS = 2000

export default function App() {
  const [docs, setDocs] = useState<DocumentOut[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.listDocuments())
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Poll only while something is still being ingested.
  const pending = docs.some((d) => d.status === 'queued' || d.status === 'processing')
  useEffect(() => {
    if (!pending) return
    const t = setInterval(refresh, POLL_MS)
    return () => clearInterval(t)
  }, [pending, refresh])

  async function onFiles(files: FileList | null) {
    if (!files?.length) return
    setUploading(true)
    setError(null)
    try {
      for (const file of Array.from(files)) await api.upload(file)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
      void refresh()
    }
  }

  async function onDelete(id: string) {
    await api.remove(id)
    setSelected((s) => {
      const next = new Set(s)
      next.delete(id)
      return next
    })
    void refresh()
  }

  function toggle(id: string) {
    setSelected((s) => {
      const next = new Set(s)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <main>
      <header>
        <h1>RAG DocSearch</h1>
        <p className="muted">Upload PDFs or Markdown, then ask questions answered only from them.</p>
      </header>

      {error && (
        <div role="alert" className="error">
          {error}
        </div>
      )}

      <section className="panel">
        <h2>Documents</h2>
        <UploadZone disabled={uploading} onFiles={onFiles} />
        {docs.length === 0 ? (
          <p className="muted">No documents yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th aria-label="Scope" />
                <th>File</th>
                <th>Status</th>
                <th>Pages</th>
                <th>Chunks</th>
                <th>Size</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Limit questions to ${d.filename}`}
                      checked={selected.has(d.id)}
                      disabled={d.status !== 'ready'}
                      onChange={() => toggle(d.id)}
                    />
                  </td>
                  <td>{d.filename}</td>
                  <td>
                    <span className={`badge ${d.status}`} title={d.error ?? undefined}>
                      {d.status}
                    </span>
                  </td>
                  <td>{d.page_count ?? '–'}</td>
                  <td>{d.chunk_count}</td>
                  <td>{formatBytes(d.size_bytes)}</td>
                  <td>
                    <button className="link" onClick={() => onDelete(d.id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <AskPanel scope={[...selected]} />
    </main>
  )
}

function UploadZone({ disabled, onFiles }: { disabled: boolean; onFiles: (f: FileList | null) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  return (
    <div
      className={`dropzone ${over ? 'over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        onFiles(e.dataTransfer.files)
      }}
    >
      <p>{disabled ? 'Uploading…' : 'Drop PDF, .md or .txt files here'}</p>
      <button disabled={disabled} onClick={() => input.current?.click()}>
        Choose files
      </button>
      <input
        ref={input}
        type="file"
        hidden
        multiple
        accept=".pdf,.md,.markdown,.txt"
        onChange={(e) => onFiles(e.target.files)}
      />
    </div>
  )
}

function AskPanel({ scope }: { scope: string[] }) {
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<AskResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [active, setActive] = useState<number | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!question.trim()) return
    setBusy(true)
    setError(null)
    setActive(null)
    try {
      setResult(await api.ask(question.trim(), scope))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const segments = useMemo(
    () =>
      result ? parseCitations(result.answer, new Set(result.citations.map((c) => c.index))) : [],
    [result],
  )

  return (
    <section className="panel">
      <h2>Ask</h2>
      <form onSubmit={submit} className="ask">
        <input
          aria-label="Question"
          placeholder="e.g. Who approves expense claims above LKR 50,000?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button disabled={busy || !question.trim()}>{busy ? 'Thinking…' : 'Ask'}</button>
      </form>
      {scope.length > 0 && (
        <p className="muted">Limited to {scope.length} selected document(s).</p>
      )}
      {error && (
        <div role="alert" className="error">
          {error}
        </div>
      )}
      {result && (
        <div className="answer">
          {result.mode === 'extractive' && (
            <p className="muted">Extractive mode: set ANTHROPIC_API_KEY for generated answers.</p>
          )}
          <p className="answer-text">
            {segments.map((s, i) =>
              s.kind === 'text' ? (
                <span key={i}>{s.text}</span>
              ) : (
                <button
                  key={i}
                  className="cite"
                  aria-label={`Show source ${s.index}`}
                  onClick={() => setActive(s.index)}
                >
                  {s.index}
                </button>
              ),
            )}
          </p>
          <ol className="sources">
            {result.citations.map((c) => (
              <li key={c.chunk_id} className={active === c.index ? 'active' : ''}>
                <strong>
                  [{c.index}] {c.filename}
                  {c.page ? `, p. ${c.page}` : ''}
                </strong>
                <span>{c.snippet}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  )
}
