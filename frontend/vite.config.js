import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
    plugins: [react(), tailwindcss()],
    // .env 파일을 프로젝트 루트에서 읽도록 설정
    envDir: path.resolve(__dirname, '..'),
    envPrefix: 'VITE_',
    server: {
        proxy: {
            // ngrok URL을 프록시로 사용 (CORS 우회)
            '/api/colab': {
                target: 'https://cleopatra-palaeontological-impressibly.ngrok-free.dev',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/colab/, '/api'),
                configure: (proxy, _options) => {
                    proxy.on('proxyReq', (proxyReq, req, _res) => {
                        // ngrok 경고 페이지 우회 헤더
                        proxyReq.setHeader('ngrok-skip-browser-warning', 'true')
                    })
                }
            }
        }
    }
})