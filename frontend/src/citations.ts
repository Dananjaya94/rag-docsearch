export type Segment = { kind: 'text'; text: string } | { kind: 'cite'; index: number }

/** Split an answer like "14 days [1][3]." into text and citation segments.
 *  Only indices that exist in the citation list become links; anything else stays as text. */
export function parseCitations(answer: string, validIndices: Set<number>): Segment[] {
  const segments: Segment[] = []
  const pattern = /\[(\d{1,3})\]/g
  let last = 0
  for (const match of answer.matchAll(pattern)) {
    const index = Number(match[1])
    if (!validIndices.has(index)) continue
    const start = match.index ?? 0
    if (start > last) segments.push({ kind: 'text', text: answer.slice(last, start) })
    segments.push({ kind: 'cite', index })
    last = start + match[0].length
  }
  if (last < answer.length) segments.push({ kind: 'text', text: answer.slice(last) })
  return segments
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
