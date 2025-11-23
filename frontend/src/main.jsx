import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'

import AppLayout from './layouts/AppLayout.jsx'
import Landing from './pages/Landing.jsx'
import Main from './pages/Main.jsx'

// 에러 핸들링
if (!document.getElementById('root')) {
  console.error('Root element not found!')
}

try {
  const router = createBrowserRouter([
    {
      path: '/',
      element: <AppLayout />,
      children: [
        { index: true, element: <Landing /> },
        { path: 'app', element: <Main /> },
        { path: 'app/about', element: <Main /> },
        { path: 'app/generation', element: <Main /> },
        { path: 'app/archive', element: <Main /> },
        { path: 'app/emotion', element: <Main /> },
        { path: 'app/settings', element: <Main /> },
      ],
    },
  ])

  const root = document.getElementById('root')
  if (root) {
    createRoot(root).render(
      <StrictMode>
        <RouterProvider router={router} />
      </StrictMode>,
    )
  } else {
    console.error('Root element not found!')
  }
} catch (error) {
  console.error('Error initializing app:', error)
  const root = document.getElementById('root')
  if (root) {
    root.innerHTML = `
      <div style="padding: 20px; font-family: sans-serif;">
        <h1>애플리케이션 초기화 오류</h1>
        <p>오류가 발생했습니다: ${error.message}</p>
        <p>브라우저 콘솔을 확인해주세요.</p>
      </div>
    `
  }
}
