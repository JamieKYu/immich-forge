import { defineConfig } from 'vitest/config'

// Standalone Vitest config — deliberately NOT the Vite build config, which wires
// in the crx() plugin (it builds the whole extension and needs a manifest).
// Tests only exercise the pure lib/ modules, so a plain jsdom environment is all
// they need.
export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./test/setup.ts'],
    include: ['src/**/*.test.ts'],
  },
})
