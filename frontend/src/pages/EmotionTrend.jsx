import React, { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'

const STORAGE_KEY = 'saved_poems'
// API URL 설정: VITE_API_URL이 전체 경로일 수 있으므로 처리
const getApiBaseUrl = () => {
    const envUrl = import.meta.env.VITE_API_URL || ''
    if (envUrl.includes('/api/poem/generate')) {
        return envUrl.replace('/api/poem/generate', '')
    }
    if (envUrl && !envUrl.includes('/api/')) {
        return envUrl
    }
    return 'http://localhost:8000'
}
const API_URL = getApiBaseUrl()

// 감정 색상 매핑 (구체적인 감정들)
const EMOTION_COLORS = {
    '기쁨': '#4CAF50',
    '슬픔': '#2196F3',
    '분노': '#F44336',
    '놀람': '#FF9800',
    '두려움': '#9C27B0',
    '혐오': '#795548',
    '사랑': '#E91E63',
    '그리움': '#00BCD4',
    '평온': '#8BC34A',
    '불안': '#FF5722',
    '희망': '#FFC107',
    '실망': '#607D8B',
    '중립': '#9E9E9E'
}

function EmotionTrend() {
    const [poems, setPoems] = useState([])
    const [cuteAnalysis, setCuteAnalysis] = useState(null)
    const [loadingAnalysis, setLoadingAnalysis] = useState(false)
    const [analysisError, setAnalysisError] = useState(null)

    useEffect(() => {
        loadPoems()
    }, [])

    useEffect(() => {
        // 시 데이터가 로드되고 감정 분석이 아직 실행되지 않았을 때만 실행
        if (poems.length > 0 && !cuteAnalysis && !loadingAnalysis && !analysisError) {
            console.log('[EmotionTrend] 시 데이터 로드 완료, 감정 분석 시작')
            loadCuteAnalysis()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [poems.length])  // poems.length만 의존성으로 사용하여 무한 루프 방지

    const loadPoems = () => {
        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            console.log('[EmotionTrend] 로드된 시 개수:', savedPoems.length)
            console.log('[EmotionTrend] 샘플 데이터:', savedPoems[0])
            setPoems(savedPoems)
        } catch (err) {
            console.error('시 목록 불러오기 실패:', err)
        }
    }

    const loadCuteAnalysis = async () => {
        if (poems.length === 0) {
            console.log('[EmotionTrend] 시 데이터가 없어서 감정 분석을 건너뜁니다.')
            return
        }
        
        setLoadingAnalysis(true)
        setAnalysisError(null)
        
        const apiUrl = `${API_URL}/api/emotion/analyze-cute`
        console.log('[EmotionTrend] 감정 분석 시작:', apiUrl)
        console.log('[EmotionTrend] 전송할 시 개수:', poems.length)
        
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    poems: poems
                })
            })
            
            console.log('[EmotionTrend] 응답 상태:', response.status, response.statusText)
            
            if (response.ok) {
                const data = await response.json()
                console.log('[EmotionTrend] 감정 분석 결과:', data)
                setCuteAnalysis(data)
                setAnalysisError(null)
            } else {
                const errorText = await response.text()
                console.error('[EmotionTrend] 감정 분석 실패:', response.status, errorText)
                setAnalysisError(`서버 오류: ${response.status} ${response.statusText}`)
                setCuteAnalysis(null)
            }
        } catch (err) {
            console.error('[EmotionTrend] 감정 분석 오류:', err)
            setAnalysisError(`연결 오류: ${err.message}`)
            setCuteAnalysis(null)
        } finally {
            setLoadingAnalysis(false)
        }
    }

    // 주 단위로 날짜를 그룹화하는 헬퍼 함수
    const getWeekStart = (dateStr) => {
        const date = new Date(dateStr)
        const day = date.getDay() // 0(일요일) ~ 6(토요일)
        const diff = day === 0 ? -6 : 1 - day // 월요일을 주의 시작으로
        const weekStart = new Date(date)
        weekStart.setDate(date.getDate() + diff)
        return weekStart.toISOString().split('T')[0] // YYYY-MM-DD
    }

    // 주 라벨 포맷팅 함수
    const formatWeekLabel = (weekStartStr) => {
        const weekStart = new Date(weekStartStr)
        const weekEnd = new Date(weekStart)
        weekEnd.setDate(weekStart.getDate() + 6)
        
        const startMonth = weekStart.getMonth() + 1
        const startDay = weekStart.getDate()
        const endMonth = weekEnd.getMonth() + 1
        const endDay = weekEnd.getDate()
        
        if (startMonth === endMonth) {
            return `${startMonth}/${startDay}-${endDay}`
        } else {
            return `${startMonth}/${startDay}-${endMonth}/${endDay}`
        }
    }

    // 주 단위 감정 분포 데이터 계산
    const weeklyEmotionData = useMemo(() => {
        if (!poems.length) return []

        // 모든 감정 종류 수집
        const allEmotions = new Set()
        poems.forEach(poem => {
            if (poem.emotion) allEmotions.add(poem.emotion)
        })
        
        // 주 단위로 그룹화
        const weekMap = {}
        
        poems.forEach(poem => {
            if (!poem.createdAt || !poem.emotion) return
            
            const date = new Date(poem.createdAt).toISOString().split('T')[0] // YYYY-MM-DD
            const weekStart = getWeekStart(date)
            
            if (!weekMap[weekStart]) {
                // 모든 감정을 0으로 초기화
                weekMap[weekStart] = { 
                    date: weekStart,
                    weekLabel: formatWeekLabel(weekStart)
                }
                allEmotions.forEach(emotion => {
                    weekMap[weekStart][emotion] = 0
                })
            }
            
            if (weekMap[weekStart][poem.emotion] !== undefined) {
                weekMap[weekStart][poem.emotion]++
            }
        })

        // 날짜순으로 정렬
        return Object.values(weekMap).sort((a, b) => a.date.localeCompare(b.date))
    }, [poems])

    // 날짜별 감정 분포 데이터 계산 (최근 7일용)
    const dailyEmotionData = useMemo(() => {
        if (!poems.length) return []

        // 모든 감정 종류 수집
        const allEmotions = new Set()
        poems.forEach(poem => {
            if (poem.emotion) allEmotions.add(poem.emotion)
        })
        
        // 날짜별로 그룹화
        const dateMap = {}
        
        poems.forEach(poem => {
            if (!poem.createdAt || !poem.emotion) return
            
            const date = new Date(poem.createdAt).toISOString().split('T')[0] // YYYY-MM-DD
            
            if (!dateMap[date]) {
                // 모든 감정을 0으로 초기화
                dateMap[date] = { date }
                allEmotions.forEach(emotion => {
                    dateMap[date][emotion] = 0
                })
            }
            
            if (dateMap[date][poem.emotion] !== undefined) {
                dateMap[date][poem.emotion]++
            }
        })

        // 날짜순으로 정렬
        return Object.values(dateMap).sort((a, b) => a.date.localeCompare(b.date))
    }, [poems])

    // 감정별 총 개수
    const emotionCount = useMemo(() => {
        const count = {}
        
        poems.forEach(poem => {
            if (poem.emotion) {
                count[poem.emotion] = (count[poem.emotion] || 0) + 1
            }
        })

        return Object.entries(count)
            .map(([name, value]) => ({
                name,
                value,
                color: EMOTION_COLORS[name] || '#9E9E9E'
            }))
            .filter(item => item.value > 0)
            .sort((a, b) => b.value - a.value) // 개수 순으로 정렬
    }, [poems])

    // 감정 신뢰도 분포 (구간별)
    const confidenceData = useMemo(() => {
        if (!poems.length) return []

        const bins = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }

        poems.forEach(poem => {
            const confidence = poem.emotion_confidence || 0
            if (confidence < 0.2) bins['0.0-0.2']++
            else if (confidence < 0.4) bins['0.2-0.4']++
            else if (confidence < 0.6) bins['0.4-0.6']++
            else if (confidence < 0.8) bins['0.6-0.8']++
            else bins['0.8-1.0']++
        })

        return Object.entries(bins).map(([name, value]) => ({ name, value }))
    }, [poems])

    // 최근 7일 감정 추이 (상위 5개 감정만 표시)
    const recentWeekData = useMemo(() => {
        if (!poems.length || emotionCount.length === 0) {
            // 데이터가 없어도 7일 구조는 유지
            const last7Days = []
            const today = new Date()
            for (let i = 6; i >= 0; i--) {
                const date = new Date(today)
                date.setDate(date.getDate() - i)
                last7Days.push({
                    date: date.toISOString().split('T')[0],
                    dateFormatted: date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })
                })
            }
            return last7Days.map(item => ({
                date: item.dateFormatted
            }))
        }

        // 가장 많이 나타난 상위 5개 감정 선택
        const topEmotions = emotionCount.slice(0, 5).map(item => item.name)
        
        const last7Days = []
        const today = new Date()
        
        for (let i = 6; i >= 0; i--) {
            const date = new Date(today)
            date.setDate(date.getDate() - i)
            const dateStr = date.toISOString().split('T')[0]
            
            const dayData = dailyEmotionData.find(d => d.date === dateStr)
            const dayResult = { date: dateStr }
            
            // 상위 5개 감정만 포함
            topEmotions.forEach(emotion => {
                dayResult[emotion] = dayData ? (dayData[emotion] || 0) : 0
            })
            
            last7Days.push(dayResult)
        }

        return last7Days.map(item => ({
            ...item,
            date: new Date(item.date).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })
        }))
    }, [dailyEmotionData, emotionCount])

    if (poems.length === 0) {
        return (
            <div className="px-6 sm:px-8 md:px-10 pt-4 sm:pt-6 md:pt-8 pb-4 sm:pb-6 md:pb-8 max-w-4xl mx-auto">
                <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-8">감정 추이</h2>
                <div className="p-6 bg-transparent border border-gray-600 rounded-lg text-center">
                    <p className="text-gray-600">아직 생성된 시가 없습니다.</p>
                    <p className="text-sm text-gray-500 mt-2">시를 생성하면 감정 추이를 확인할 수 있습니다.</p>
                </div>
            </div>
        )
    }

    return (
        <div className="px-6 sm:px-8 md:px-10 pt-4 sm:pt-6 md:pt-8 pb-4 sm:pb-6 md:pb-8 max-w-6xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-8">감정 추이</h2>
            
            <div className="space-y-8">
                {/* 귀여운 감정 분석 - 한 곳에서 표시 */}
                <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                    {loadingAnalysis ? (
                        /* 로딩 중 */
                        <div className="text-center">
                            <p className="text-gray-600">감정 분석 중...</p>
                        </div>
                    ) : analysisError ? (
                        /* 에러 발생 */
                        <div>
                            <p className="text-gray-800 text-sm mb-2">
                                감정 분석 오류: {analysisError}
                            </p>
                            <button
                                onClick={loadCuteAnalysis}
                                className="text-sm text-gray-600 hover:text-gray-800 underline"
                            >
                                다시 시도
                            </button>
                        </div>
                    ) : cuteAnalysis ? (
                        /* 분석 결과 표시 */
                        <div className="flex items-start gap-4">
                            {/* 이모지 섹션 */}
                            <div className="flex-shrink-0">
                                <div className="text-4xl">{cuteAnalysis.emoji}</div>
                            </div>
                            
                            {/* 내용 섹션 */}
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="text-lg font-semibold text-gray-800">
                                        {cuteAnalysis.success ? '감정 이야기' : '감정 분석 안내'}
                                    </h3>
                                    <button
                                        onClick={loadCuteAnalysis}
                                        className="text-xs text-gray-600 hover:text-gray-800 underline cursor-pointer"
                                    >
                                        새로고침
                                    </button>
                                </div>
                                
                                <div className="space-y-3">
                                    <p className="text-gray-800 leading-relaxed">
                                        {cuteAnalysis.story}
                                    </p>
                                    
                                    {cuteAnalysis.success && (
                                        <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-gray-600">
                                            <span className="text-sm text-gray-600">{cuteAnalysis.summary}</span>
                                            <span className="text-sm text-gray-800 font-medium">{cuteAnalysis.message}</span>
                                        </div>
                                    )}
                                    
                                    {!cuteAnalysis.success && (
                                        <div className="pt-3 border-t border-gray-600">
                                            <p className="text-sm text-gray-800 mb-2">{cuteAnalysis.message}</p>
                                            <p className="text-xs text-gray-600 leading-relaxed">
                                                💡 프로젝트 루트의 .env 파일에 GEMINI_API_KEY를 추가하고 백엔드 서버를 재시작하세요.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ) : poems.length > 0 ? (
                        /* 분석 결과 없음 (재시도 버튼) */
                        <div>
                            <p className="text-gray-800 text-sm mb-2">
                                감정 분석을 불러오지 못했습니다.
                            </p>
                            <button
                                onClick={loadCuteAnalysis}
                                className="text-sm text-gray-600 hover:text-gray-800 underline"
                            >
                                감정 분석 다시 시도
                            </button>
                        </div>
                    ) : null}
                </div>

                {/* 통계 요약 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-transparent border border-gray-600 rounded-lg">
                        <p className="text-sm text-gray-600 mb-1">총 시 개수</p>
                        <p className="text-2xl font-bold text-gray-800">{poems.length}개</p>
                    </div>
                    <div className="p-4 bg-transparent border border-gray-600 rounded-lg">
                        <p className="text-sm text-gray-600 mb-1">가장 많은 감정</p>
                        <p className="text-2xl font-bold text-gray-800">
                            {emotionCount.length > 0 
                                ? emotionCount.reduce((max, item) => item.value > max.value ? item : max).name
                                : '-'
                            }
                        </p>
                    </div>
                    <div className="p-4 bg-transparent border border-gray-600 rounded-lg">
                        <p className="text-sm text-gray-600 mb-1">평균 신뢰도</p>
                        <p className="text-2xl font-bold text-gray-800">
                            {poems.length > 0
                                ? (poems.reduce((sum, p) => sum + (p.emotion_confidence || 0), 0) / poems.length).toFixed(2)
                                : '0.00'
                            }
                        </p>
                    </div>
                </div>

                {/* 최근 7일 감정 추이 */}
                <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">최근 7일 감정 추이</h3>
                    <div style={{ width: '100%', height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={recentWeekData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                            <XAxis 
                                dataKey="date" 
                                stroke="#666"
                                style={{ fontSize: '12px' }}
                            />
                            <YAxis 
                                stroke="#666"
                                style={{ fontSize: '12px' }}
                            />
                            <Tooltip 
                                contentStyle={{ 
                                    backgroundColor: 'white', 
                                    border: '1px solid #ccc',
                                    borderRadius: '4px'
                                }}
                            />
                            <Legend />
                            {recentWeekData.length > 0 && Object.keys(recentWeekData[0])
                                .filter(key => key !== 'date')
                                .map(emotion => (
                                    <Line 
                                        key={emotion}
                                        type="monotone" 
                                        dataKey={emotion} 
                                        stroke={EMOTION_COLORS[emotion] || '#9E9E9E'} 
                                        strokeWidth={2}
                                        dot={{ r: 4 }}
                                    />
                                ))
                            }
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* 감정 신뢰도 분포 */}
                {confidenceData.some(item => item.value > 0) && (
                    <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">감정 신뢰도 분포</h3>
                        <div style={{ width: '100%', height: '300px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={confidenceData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                                <XAxis 
                                    dataKey="name" 
                                    stroke="#666"
                                    style={{ fontSize: '12px' }}
                                />
                                <YAxis 
                                    stroke="#666"
                                    style={{ fontSize: '12px' }}
                                />
                                <Tooltip 
                                    contentStyle={{ 
                                        backgroundColor: 'white', 
                                        border: '1px solid #ccc',
                                        borderRadius: '4px'
                                    }}
                                />
                                <Bar dataKey="value" fill="#79A9E6" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}

                {/* 전체 기간 감정 추이 (주 단위) */}
                {weeklyEmotionData.length > 0 && (
                    <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">전체 기간 감정 추이 (주 단위)</h3>
                        <div style={{ width: '100%', height: '400px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={weeklyEmotionData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                                <XAxis 
                                    dataKey="weekLabel" 
                                    stroke="#666"
                                    style={{ fontSize: '11px' }}
                                    angle={-45}
                                    textAnchor="end"
                                    height={80}
                                />
                                <YAxis 
                                    stroke="#666"
                                    style={{ fontSize: '12px' }}
                                />
                                <Tooltip 
                                    contentStyle={{ 
                                        backgroundColor: 'white', 
                                        border: '1px solid #ccc',
                                        borderRadius: '4px'
                                    }}
                                />
                                <Legend />
                                {weeklyEmotionData.length > 0 && Object.keys(weeklyEmotionData[0])
                                    .filter(key => key !== 'date' && key !== 'weekLabel')
                                    .slice(0, 8) // 최대 8개 감정만 표시 (너무 많으면 복잡함)
                                    .map(emotion => (
                                        <Line 
                                            key={emotion}
                                            type="monotone" 
                                            dataKey={emotion} 
                                            stroke={EMOTION_COLORS[emotion] || '#9E9E9E'} 
                                            strokeWidth={2}
                                            dot={{ r: 3 }}
                                        />
                                    ))
                                }
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default EmotionTrend

