import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
    plugins: [react(), tailwindcss()],
    envDir: path.resolve(__dirname, '..'),  // 프로젝트 루트의 .env 파일 읽기
    envPrefix: 'VITE_',  // VITE_로 시작하는 환경 변수만 로드
})