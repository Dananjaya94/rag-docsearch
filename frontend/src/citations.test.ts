import { describe, expect, it } from 'vitest'
import { formatBytes, parseCitations } from './citations'

describe('parseCitations', () => {
  it('splits text and valid citation markers', () => {
    expect(parseCitations('14 days [1][2].', new Set([1, 2]))).toEqual([
      { kind: 'text', text: '14 days ' },
      { kind: 'cite', index: 1 },
      { kind: 'cite', index: 2 },
      { kind: 'text', text: '.' },
    ])
  })

  it('leaves unknown indices and non-citation brackets as text', () => {
    expect(parseCitations('See [9] and [note].', new Set([1]))).toEqual([
      { kind: 'text', text: 'See [9] and [note].' },
    ])
  })

  it('handles an empty answer', () => {
    expect(parseCitations('', new Set())).toEqual([])
  })
})

describe('formatBytes', () => {
  it('picks a sensible unit', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })
})
