export type DocumentStatus = 'queued' | 'processing' | 'ready' | 'failed'

export interface DocumentOut {
  id: string
  filename: string
  content_type: string
  size_bytes: number
  status: DocumentStatus
  error: string | null
  page_count: number | null
  chunk_count: number
  created_at: string
  updated_at: string
}

export interface Citation {
  index: number
  chunk_id: number
  document_id: string
  filename: string
  page: number | null
  snippet: string
}

export interface AskResponse {
  question: string
  answer: string
  mode: 'llm' | 'extractive'
  citations: Citation[]
}

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  listDocuments: () => request<DocumentOut[]>('/documents'),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<DocumentOut>('/documents', { method: 'POST', body: form })
  },
  remove: (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' }),
  ask: (question: string, documentIds?: string[]) =>
    request<AskResponse>('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, document_ids: documentIds?.length ? documentIds : null }),
    }),
}
