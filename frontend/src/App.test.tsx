import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const doc = {
  id: 'd1',
  filename: 'handbook.pdf',
  content_type: 'application/pdf',
  size_bytes: 2048,
  status: 'ready',
  error: null,
  page_count: 3,
  chunk_count: 4,
  created_at: '2026-11-07T00:00:00Z',
  updated_at: '2026-11-07T00:00:00Z',
}

function json(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
}

afterEach(() => vi.restoreAllMocks())

describe('App', () => {
  it('lists documents and renders an answer with clickable citations', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input)
      if (url.endsWith('/documents')) return json([doc])
      if (url.endsWith('/ask'))
        return json({
          question: 'q',
          answer: 'A department head approves them [1].',
          mode: 'llm',
          citations: [
            { index: 1, chunk_id: 7, document_id: 'd1', filename: 'handbook.pdf', page: 2, snippet: 'Claims above…' },
          ],
        })
      return Promise.reject(new Error(`unexpected ${url}`))
    })

    render(<App />)
    expect(await screen.findByText('handbook.pdf')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Who approves?' } })
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))

    const cite = await screen.findByRole('button', { name: 'Show source 1' })
    fireEvent.click(cite)
    await waitFor(() => expect(screen.getByText('[1] handbook.pdf, p. 2').closest('li')).toHaveClass('active'))
    expect(fetchMock).toHaveBeenCalledWith('/api/ask', expect.objectContaining({ method: 'POST' }))
  })

  it('shows API errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'database down' }), { status: 503 }),
    )
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('503: database down')
  })
})
