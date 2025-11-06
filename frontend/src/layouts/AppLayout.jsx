import React from 'react'
import { Outlet } from 'react-router-dom'
import CursorStars from '../components/CursorStar.jsx'

function AppLayout() {
    return (
        <div className="min-h-screen bg-white text-slate-900">
            <CursorStars color="#79A9E6" density={1} />
            <Outlet />
        </div>
    )
}

export default AppLayout
