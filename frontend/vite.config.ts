import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // ws: true so the voice WebSocket (ws://.../api/voice) proxies too,
      // not just plain HTTP requests under /api.
      '/api': { target: 'http://127.0.0.1:8000', ws: true },
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
