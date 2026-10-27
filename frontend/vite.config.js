import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发服务器配置：前端跑在 5173 端口，所有后端请求转发到 nginx（80 端口）。
// 这样开发时不用管 CORS、不用在后端改任何东西。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://localhost:80',
      '/trace': 'http://localhost:80',
      '/search': 'http://localhost:80',
      '/health': 'http://localhost:80',
      '/skills': 'http://localhost:80',
      '/monitor': 'http://localhost:80',
      '/metrics': 'http://localhost:80',
      '/knowledge': 'http://localhost:80',
      '/agents': 'http://localhost:80',
      '/demo': 'http://localhost:80',
    },
  },
})
