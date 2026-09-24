/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The site is fully static. Vercel serves it from the domain root, so `base` defaults
// to "/". GitHub Pages serves project sites from "/<repo>/"; its workflow sets BASE_PATH.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
  build: {
    target: 'es2020',
    assetsInlineLimit: 0,
    // ~200 kB of the bundle is the precomputed research data itself (inlined JSON)
    chunkSizeWarningLimit: 700,
  },
  test: {
    environment: 'node',
  },
})
