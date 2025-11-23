import React, { useState, useRef, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import PoemGeneration from './PoemGeneration.jsx'
import About from './About.jsx'
import Archive from './Archive.jsx'
import Settings from './Settings.jsx'
import EmotionTrend from './EmotionTrend.jsx'
import Footer from '../components/Footer.jsx'
import Logo from '../assets/logo.svg'
import Note1SVG from '../assets/note1.svg'
import Note2SVG from '../assets/note2.svg'
import Note3SVG from '../assets/note3.svg'
import CloverSVG from '../assets/clover.svg'

function NavLink({ children, active = false, onClick, textRef, to, isNavItem = false }) {
    if (isNavItem) {
        // 네비게이션 바 내부 아이템
        const activeClassName = active ? 'nav-link-active' : ''
        const baseClassName = `flex-1 text-center py-2.5 text-gray-800 font-medium transition-all duration-200 ${activeClassName}`
        
        if (to) {
            return (
                <Link to={to} onClick={onClick} className={baseClassName}>
                    <span ref={textRef}>{children}</span>
                </Link>
            )
        }
        
        return (
            <button onClick={onClick} className={baseClassName}>
                <span ref={textRef}>{children}</span>
            </button>
        )
    }
    
    // 설정 버튼 등
    const baseClassName = `px-4 py-2 text-gray-800 font-medium transition-all duration-300 relative ${active ? 'opacity-100' : 'opacity-80 hover:opacity-100'}`
    
    if (to) {
        return (
            <Link to={to} onClick={onClick} className={baseClassName}>
                <span ref={textRef} className="relative z-10">{children}</span>
            </Link>
        )
    }
    
    return (
        <button onClick={onClick} className={baseClassName}>
            <span ref={textRef} className="relative z-10">{children}</span>
        </button>
    )
}

function Main() {
    const location = useLocation()
    const [activeNav, setActiveNav] = useState('시 생성')
    const textRefs = useRef({})

    const navLinks = [
        { label: '시옷이란', key: 'about', to: '/app/about' },
        { label: '시 생성', key: 'generation', to: '/app/generation' },
        { label: '시 보관함', key: 'archive', to: '/app/archive' },
        { label: '감정 추이', key: 'emotion', to: '/app/emotion' },
    ]
    
    // URL에 따라 활성 탭 설정
    useEffect(() => {
        const path = location.pathname
        if (path === '/app/about') setActiveNav('시옷이란')
        else if (path === '/app/generation') setActiveNav('시 생성')
        else if (path === '/app/archive') setActiveNav('시 보관함')
        else if (path === '/app/emotion') setActiveNav('감정 추이')
        else if (path === '/app/settings') setActiveNav('설정')
        else if (path === '/app') setActiveNav('시 생성')
    }, [location.pathname])
    
    // 활성 네비게이션에 해당하는 컴포넌트 키 매핑
    const getActiveComponentKey = () => {
        if (activeNav === '시 생성') return 'generation'
        if (activeNav === '시옷이란') return 'about'
        if (activeNav === '시 보관함') return 'archive'
        if (activeNav === '감정 추이') return 'emotion'
        if (activeNav === '설정') return 'settings'
        return 'generation' // 기본값
    }
    
    // 각 네비게이션에 해당하는 컴포넌트
    const navComponents = {
        'generation': <PoemGeneration key="generation" />,
        'about': <About key="about" />,
        'archive': <Archive key="archive" />,
        'emotion': <EmotionTrend key="emotion" />,
        'settings': <Settings key="settings" />,
    }


    return (
        <div className="w-full min-h-screen relative overflow-hidden" style={{ background: 'linear-gradient(to bottom, #A4CCFF 0%, #FFFFFF 83%, #FFFFFF 100%)' }}>
            {/* 배경 장식 요소 - 화면 기준 고정 위치 */}
            {/* 상단 왼쪽 클로버 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="fixed hidden sm:block animate-float" 
                style={{
                    left: 'clamp(30px, 3vw, 60px)',
                    top: 'clamp(55%, 20vh, 60%)',
                    width: 'clamp(60px, 6vw, 80px)',
                    height: 'clamp(60px, 6vw, 80px)',
                    '--rotation': '-13deg',
                    opacity: 0.6,
                    animationDelay: '0s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* 중간 왼쪽 클로버 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="fixed hidden md:block animate-float-slow" 
                style={{
                    left: 'clamp(190px, 6vw, 200px)',
                    top: 'clamp(65%, 48vh, 70%)',
                    width: 'clamp(50px, 5vw, 70px)',
                    height: 'clamp(50px, 5vw, 70px)',
                    '--rotation': '12deg',
                    opacity: 0.6,
                    animationDelay: '0.5s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* 하단 왼쪽 큰 음표 */}
            <img 
                src={Note2SVG} 
                alt="" 
                className="fixed hidden sm:block animate-float" 
                style={{
                    left: 'clamp(50px, 2vw, 50px)',
                    bottom: 'clamp(60px, 8vh, 100px)',
                    width: 'clamp(90px, 9vw, 120px)',
                    height: 'clamp(110px, 11vw, 150px)',
                    '--rotation': '2deg',
                    opacity: 0.6,
                    animationDelay: '1s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* 하단 중간 왼쪽 작은 음표 */}
            <img 
                src={Note1SVG} 
                alt="" 
                className="fixed hidden sm:block animate-float-fast" 
                style={{
                    left: 'clamp(160px, 12vw, 180px)',
                    bottom: 'clamp(330px, 12vh, 400px)',
                    width: 'clamp(40px, 4vw, 50px)',
                    height: 'clamp(55px, 5.5vw, 75px)',
                    '--rotation': '15deg',
                    opacity: 0.6,
                    animationDelay: '1.5s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* 중앙 오른쪽 클로버 (위) */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="fixed hidden lg:block animate-float-slow" 
                style={{
                    right: 'clamp(200px, 8vw, 200px)',
                    top: 'clamp(70%, 42vh, 70%)',
                    width: 'clamp(60px, 6vw, 80px)',
                    height: 'clamp(60px, 6vw, 80px)',
                    '--rotation': '-9.61deg',
                    opacity: 0.6,
                    animationDelay: '0.8s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* 하단 중간 오른쪽 클로버 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="fixed hidden lg:block animate-float" 
                style={{
                    right: 'clamp(150px, 6vw, 150px)',
                    top: 'clamp(48%, 60vh, 48%)',
                    width: 'clamp(50px, 5vw, 70px)',
                    height: 'clamp(50px, 5vw, 70px)',
                    '--rotation': '11.07deg',
                    opacity: 0.6,
                    animationDelay: '1.2s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* 하단 오른쪽 높은음자리표 */}
            <img 
                src={Note3SVG} 
                alt="" 
                className="fixed hidden md:block animate-float-slow" 
                style={{
                    right: 'clamp(50px, 2vw, 50px)',
                    bottom: 'clamp(30px, 8vh, 50px)',
                    width: 'clamp(90px, 9vw, 140px)',
                    height: 'clamp(220px, 22vw, 350px)',
                    '--rotation': '0deg',
                    opacity: 0.6,
                    animationDelay: '0.3s',
                    zIndex: 0,
                    pointerEvents: 'none'
                }}
            />
            
            {/* Header */}
            <header className="w-full px-6 sm:px-8 md:px-10 py-6 relative z-10">
                <div className="flex items-center justify-between">
                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-2 cursor-pointer">
                        <img src={Logo} alt="시옷" className="h-10 w-auto brightness-0 opacity-70" />
                    </Link>
                    
                    {/* Center Navigation Links */}
                    <nav className="relative flex items-center border border-gray-600 rounded-full overflow-hidden min-w-[500px] px-2">
                        {navLinks.map((nav, index) => (
                            <NavLink
                                key={nav.key}
                                to={nav.to}
                                active={activeNav === nav.label}
                                onClick={() => setActiveNav(nav.label)}
                                textRef={(el) => (textRefs.current[nav.label] = el)}
                                isNavItem={true}
                            >
                                {nav.label}
                            </NavLink>
                        ))}
                    </nav>
                    
                    {/* Settings Link - Right aligned */}
                    <div>
                        <NavLink
                            to="/app/settings"
                            active={activeNav === '설정'}
                            onClick={() => setActiveNav('설정')}
                            textRef={(el) => (textRefs.current['설정'] = el)}
                        >
                            설정
                        </NavLink>
                    </div>
                </div>
            </header>
            
            {/* Main Content Area */}
            <main className="w-full min-h-[calc(100vh-80px)] relative z-10">
                {navComponents[getActiveComponentKey()]}
            </main>
            
            {/* Footer */}
            <Footer />
        </div>
    )
}

export default Main
