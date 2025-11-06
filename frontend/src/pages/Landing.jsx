import React from 'react'
import Typewriter from '../components/Typewriter.jsx'
import SegmentedTypewriter from '../components/SegmentedTypewriter.jsx'
import { Link } from 'react-router-dom'
import TitleSVG from '../assets/title.svg'
import Note1SVG from '../assets/note1.svg'
import Note2SVG from '../assets/note2.svg'
import Note3SVG from '../assets/note3.svg'
import CloverSVG from '../assets/clover.svg'

function Landing() {
    return (
        <main className="min-h-screen flex flex-col items-center justify-center px-8 sm:px-14 py-10 relative overflow-hidden" style={{ background: 'linear-gradient(to bottom, #A4CCFF 0%, #FFFFFF 83%, #FFFFFF 100%)' }}>
            {/* Decorative graphics */}
            {/* 상단 왼쪽 클로버 - 상단 왼쪽 사분면, 높고 왼쪽 가장자리에 가까움 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="absolute hidden sm:block animate-float" 
                style={{
                    left: 'clamp(30px, 3vw, 35px)',
                    top: 'clamp(55%, 20vh, 55%)',
                    width: 'clamp(60px, 6vw, 80px)',
                    height: 'clamp(60px, 6vw, 80px)',
                    '--rotation': '-13deg',
                    opacity: 1,
                    animationDelay: '0s'
                }}
            />
            
            {/* 중간 왼쪽 클로버 - 상단 왼쪽 클로버 아래, 약간 오른쪽, 수직 중앙 근처 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="absolute hidden md:block animate-float-slow" 
                style={{
                    left: 'clamp(190px, 6vw, 200px)',
                    top: 'clamp(65%, 48vh, 65%)',
                    width: 'clamp(50px, 5vw, 70px)',
                    height: 'clamp(50px, 5vw, 70px)',
                    '--rotation': '12deg',
                    opacity: 1,
                    animationDelay: '0.5s'
                }}
            />
            
            {/* 하단 왼쪽 큰 음표 (4분음표) - 하단 왼쪽 사분면, 페이지 하단에 가깝고 왼쪽 가장자리에 가까움 */}
            <img 
                src={Note2SVG} 
                alt="" 
                className="absolute hidden sm:block animate-float" 
                style={{
                    left: 'clamp(50px, 2vw, 50px)',
                    bottom: 'clamp(60px, 8vh, 60px)',
                    width: 'clamp(90px, 9vw, 120px)',
                    height: 'clamp(110px, 11vw, 150px)',
                    '--rotation': '2deg',
                    opacity: 1,
                    animationDelay: '1s'
                }}
            />
            
            {/* 하단 중간 왼쪽 작은 음표 (8분음표) - 큰 4분음표의 오른쪽, 약간 위에 위치 */}
            <img 
                src={Note1SVG} 
                alt="" 
                className="absolute hidden sm:block animate-float-fast" 
                style={{
                    left: 'clamp(160px, 12vw, 160px)',
                    bottom: 'clamp(330px, 12vh, 330px)',
                    width: 'clamp(40px, 4vw, 50px)',
                    height: 'clamp(55px, 5.5vw, 75px)',
                    '--rotation': '15deg',
                    opacity: 1,
                    animationDelay: '1.5s'
                }}
            />
            
            {/* 중앙 오른쪽 (시 텍스트 위): 꽃/별 모양 - 중앙 시 텍스트 블록의 오른쪽 중간, 텍스트의 상단 라인 근처 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="absolute hidden lg:block animate-float-slow" 
                style={{
                    right: 'clamp(200px, 8vw, 200px)',
                    top: 'clamp(70%, 42vh, 70%)',
                    width: 'clamp(60px, 6vw, 80px)',
                    height: 'clamp(60px, 6vw, 80px)',
                    '--rotation': '-9.61deg',
                    opacity: 1,
                    animationDelay: '0.8s'
                }}
            />
            
            {/* 하단 중간 오른쪽: 꽃/별 모양 - 중앙 시 텍스트 블록의 오른쪽 중간, 텍스트의 하단 라인 근처 */}
            <img 
                src={CloverSVG} 
                alt="" 
                className="absolute hidden lg:block animate-float" 
                style={{
                    right: 'clamp(150px, 6vw, 150px)',
                    top: 'clamp(48%, 60vh, 48%)',
                    width: 'clamp(50px, 5vw, 70px)',
                    height: 'clamp(50px, 5vw, 70px)',
                    '--rotation': '11.07deg',
                    opacity: 1,
                    animationDelay: '1.2s'
                }}
            />
            
            {/* 하단 오른쪽: 높은음자리표 (트레블 클레프) - 가장 오른쪽 하단 모서리 근처 */}
            <img 
                src={Note3SVG} 
                alt="" 
                className="absolute hidden md:block animate-float-slow" 
                style={{
                    right: 'clamp(50px, 2vw, 50px)',
                    bottom: 'clamp(50px, 8vh, 50px)',
                    width: 'clamp(90px, 9vw, 140px)',
                    height: 'clamp(220px, 22vw, 350px)',
                    '--rotation': '0deg',
                    opacity: 1,
                    animationDelay: '0.3s'
                }}
            />
            
            <div className="flex flex-col items-center space-y-8 max-w-4xl -mt-20 relative z-10">
                {/* 상단 "siot" 타이틀 */}
                <img 
                    src={TitleSVG} 
                    alt="시옷" 
                    className="h-[50px] sm:h-[60px] md:h-[70px] lg:h-[85px] w-auto animate-float-slow"
                    style={{
                        '--rotation': '0deg',
                        animationDelay: '0s'
                    }}
                />

                {/* 시 텍스트 블록 - 타이핑 효과 유지 */}
                <div className="text-center space-y-3 text-gray-400 text-[20px] font-medium leading-tight mt-12" style={{ letterSpacing: '0.23em' }}>
                    <p>
                        <SegmentedTypewriter
                            startDelay={100}
                            speedMs={40}
                            segments={[
                                { text: '시', className: 'text-[24px] text-gray-500' },
                                { text: ' 한 줄✑을 책갈피에', className: '' }
                            ]}
                        />
                    </p>
                    <p>
                        <Typewriter
                            text="잠시 끼워두고, 숨을 고릅니다"
                            startDelay={800}
                            speedMs={40}
                        />
                    </p>
                    <p>
                        <SegmentedTypewriter
                            startDelay={2000}
                            speedMs={40}
                            segments={[
                                { text: '바람은 ', className: '' },
                                { text: '옷', className: 'text-[24px] text-gray-500' },
                                { text: '깃을 살며시 스쳐가고,', className: '' }
                            ]}
                        />
                    </p>
                    <p>
                        <Typewriter
                            text="밤☾은 조용히 흘러갑니다"
                            startDelay={3600}
                            speedMs={40}
                        />
                    </p>
                    <p>
                        <Typewriter
                            text="말하지 않은 마음이"
                            startDelay={4800}
                            speedMs={40}
                        />
                    </p>
                    <p>
                        <Typewriter
                            text="✱고요 속에 자리를 잡습니다"
                            startDelay={6000}
                            speedMs={40}
                        />
                    </p>
                </div>

                {/* 하단 버튼 */}
                <Link 
                    to="/app" 
                    className="text-gray-700 hover:text-gray-800 transition-colors font-medium text-[20px] cursor-pointer mt-8"
                    style={{ letterSpacing: '0.23em' }}
                >
                    (시옷 사용하기)
                </Link>
            </div>
        </main>
    )
}

export default Landing
