// Global test setup: install a fresh fake `chrome` API and reset all mocks
// before each test so suites don't leak state into one another.
import { afterEach, beforeEach, vi } from 'vitest'

function freshChrome() {
  return {
    storage: {
      local: {
        get: vi.fn(),
        set: vi.fn().mockResolvedValue(undefined),
      },
    },
    permissions: {
      request: vi.fn(),
      contains: vi.fn(),
    },
  }
}

beforeEach(() => {
  // @ts-expect-error — partial chrome stub, only what lib/ touches.
  globalThis.chrome = freshChrome()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})
