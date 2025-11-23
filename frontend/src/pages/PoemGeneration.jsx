import React, { useState, useEffect, useRef } from 'react'

// API URL 설정
// - SOLAR 모델: Colab URL 사용 (VITE_COLAB_API_URL)
// - koGPT2 모델: 로컬 URL 사용 (VITE_API_URL 또는 localhost)
const COLAB_API_URL = import.meta.env.VITE_COLAB_API_URL || ''  // Colab ngrok URL
const LOCAL_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/poem/generate'
const STORAGE_KEY = 'saved_poems'
const SETTINGS_KEY = 'app_settings'

// 디버깅: 환경 변수 확인
console.log('🔍 환경 변수 확인:', {
    VITE_COLAB_API_URL: import.meta.env.VITE_COLAB_API_URL || '(없음)',
    VITE_API_URL: import.meta.env.VITE_API_URL || '(없음)',
    COLAB_API_URL: COLAB_API_URL || '(없음)',
    LOCAL_API_URL
})

// 커스텀 드롭다운 컴포넌트
function CustomDropdown({ value, onChange, options, placeholder, disabled }) {
    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef(null)

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false)
            }
        }

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside)
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [isOpen])

    const selectedOption = options.find(opt => opt.value === value) || { label: placeholder }

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                type="button"
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 focus:border-gray-600 text-sm text-left flex items-center justify-between cursor-pointer hover:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <span className={value ? 'text-gray-800' : 'text-gray-600'}>
                    {selectedOption.label}
                </span>
                <svg 
                    className={`w-4 h-4 text-gray-600 transition-transform duration-200 ${isOpen ? 'transform rotate-180' : ''}`}
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>
            
            {isOpen && (
                <div className="absolute z-10 w-full mt-1 border border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto bg-white/30 backdrop-blur-xl">
                    {options.map((option) => (
                        <button
                            key={option.value}
                            type="button"
                            onClick={() => {
                                onChange(option.value)
                                setIsOpen(false)
                            }}
                            className={`w-full px-3 py-2 text-left text-sm transition-colors ${
                                value === option.value
                                    ? 'bg-[#79A9E6]/80 text-white'
                                    : 'text-gray-800 hover:bg-white/20'
                            }`}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}

function PoemGeneration() {
    const [text, setText] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [saved, setSaved] = useState(false)
    
    // 프롬프트 옵션 상태
    const [lines, setLines] = useState(4)
    const [mood, setMood] = useState('')
    const [requiredKeywords, setRequiredKeywords] = useState('')
    const [bannedWords, setBannedWords] = useState('')
    const [useRhyme, setUseRhyme] = useState(false)
    const [showOptions, setShowOptions] = useState(false)
    const [modelType, setModelType] = useState('')  // 'solar' 또는 'kogpt2'
    const [useTrainedModel, setUseTrainedModel] = useState(false)  // 학습된 모델 사용 여부
    
    // 설정 로드 (컴포넌트 마운트 시)
    useEffect(() => {
        try {
            const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
            // 기본 모델 타입 설정
            if (settings.defaultModelType) {
                setModelType(settings.defaultModelType)
                // koGPT2 선택 시 학습된 모델 자동 사용
                if (settings.defaultModelType === 'kogpt2') {
                    setUseTrainedModel(true)
                } else if (settings.defaultModelType === 'solar') {
                    setUseTrainedModel(false)
                }
            }
        } catch (err) {
            console.error('설정 로드 실패:', err)
        }
    }, [])

    const handleSubmit = async (e) => {
        e.preventDefault()
        
        if (!text.trim()) {
            setError('텍스트를 입력해주세요.')
            return
        }

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            // 타임아웃 설정 (백엔드 타임아웃과 맞춤: 300초 = 5분, 첫 요청 시 모델 로딩으로 더 오래 걸릴 수 있음)
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 330000) // 5.5분 (백엔드 300초 + 여유)
            
            // 옵션 파라미터 구성
            // showOptions가 false이면 프롬프트 옵션을 전송하지 않음
            const requestBody = {
                text: text.trim(),
                // 프롬프트 옵션이 열려있을 때만 전송
                ...(showOptions && lines && lines !== 4 ? { lines } : {}),
                ...(showOptions && mood.trim() ? { mood: mood.trim() } : {}),
                ...(showOptions && requiredKeywords.trim() 
                    ? { required_keywords: requiredKeywords.split(',').map(k => k.trim()).filter(k => k) } 
                    : {}),
                ...(showOptions && bannedWords.trim() 
                    ? { banned_words: bannedWords.split(',').map(k => k.trim()).filter(k => k) } 
                    : {}),
                ...(showOptions && useRhyme ? { use_rhyme: true } : {}),
                ...(modelType ? { model_type: modelType } : {}),
                ...(useTrainedModel ? { use_trained_model: true } : {}),
            }
            
            // 모델 타입에 따라 API URL 선택
            // SOLAR 모델: Colab URL 사용 (설정된 경우)
            // koGPT2 모델: 로컬 URL 사용
            let apiUrl = LOCAL_API_URL
            console.log('🔍 디버깅 정보:', {
                modelType,
                COLAB_API_URL: COLAB_API_URL || '(없음)',
                LOCAL_API_URL
            })
            
            // SOLAR 모델 선택 시 코랩 URL 필수
            if (modelType === 'solar') {
                if (COLAB_API_URL) {
                    apiUrl = `${COLAB_API_URL}/api/poem/generate`
                    console.log('🌐 Colab API 사용:', apiUrl)
                } else {
                    console.error('❌ SOLAR 모델을 사용하려면 코랩 URL이 필요합니다!')
                    console.error('💡 .env 파일에 VITE_COLAB_API_URL을 설정하고 프론트엔드를 재시작하세요.')
                    setError('SOLAR 모델을 사용하려면 코랩 URL이 필요합니다. .env 파일에 VITE_COLAB_API_URL을 설정하고 프론트엔드를 재시작하세요.')
                    setLoading(false)
                    return
                }
            } else {
                // koGPT2 또는 모델 미선택 시 로컬 서버 사용
                console.log('💻 로컬 API 사용:', apiUrl)
                if (!modelType) {
                    console.warn('⚠️ 모델이 선택되지 않았습니다. koGPT2 모델을 사용합니다.')
                }
            }
            
            // ngrok 무료 버전 경고 페이지 우회를 위한 헤더
            const headers = {
                'Content-Type': 'application/json',
            }
            
            // ngrok-free.dev 도메인인 경우 추가 헤더
            if (apiUrl.includes('ngrok-free.dev')) {
                headers['ngrok-skip-browser-warning'] = 'true'
            }
            
            console.log('📤 요청 전송:', {
                url: apiUrl,
                method: 'POST',
                headers: headers,
                modelType: modelType
            })
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(requestBody),
                signal: controller.signal,
                mode: 'cors'  // CORS 명시적 설정
            })
            
            console.log('📥 응답 받음:', {
                status: response.status,
                statusText: response.statusText,
                url: response.url,
                ok: response.ok
            })
            
            clearTimeout(timeoutId)

            let data
            try {
                data = await response.json()
            } catch (jsonError) {
                // JSON 파싱 실패 시 텍스트 응답 사용
                const text = await response.text()
                setError(`서버 오류: ${response.status} ${response.statusText}${text ? ` - ${text.substring(0, 200)}` : ''}`)
                return
            }

            if (!response.ok) {
                // 백엔드에서 반환하는 상세 에러 메시지 표시
                const errorMessage = data.detail || data.message || `서버 오류: ${response.status} ${response.statusText}`
                setError(errorMessage)
                return
            }

            if (data.success) {
                setResult(data)
                setSaved(false)
                
                // 자동 저장 기능 (설정에서 활성화된 경우)
                try {
                    const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
                    if (settings.autoSave !== false) {  // 기본값은 true
                        handleSavePoem(data)
                    }
                } catch (err) {
                    console.error('자동 저장 설정 확인 실패:', err)
                }
            } else {
                setError(data.message || '시 생성에 실패했습니다.')
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                setError('시 생성 시간이 너무 오래 걸려 중단되었습니다. 첫 요청은 모델 로딩으로 5분 이상 걸릴 수 있습니다. 잠시 후 다시 시도해주세요.')
            } else if (err.name === 'TypeError' && err.message.includes('fetch')) {
                setError('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.')
            } else {
                setError(`오류가 발생했습니다: ${err.message || '알 수 없는 오류'}`)
            }
            console.error('Error:', err)
        } finally {
            setLoading(false)
        }
    }

    const handleReset = () => {
        setText('')
        setResult(null)
        setError(null)
        setSaved(false)
        setLines(4)
        setMood('')
        setRequiredKeywords('')
        setBannedWords('')
        setUseRhyme(false)
        setModelType('')
        setUseTrainedModel(false)
    }

    const handleSavePoem = (poemResult = null) => {
        // poemResult가 없으면 현재 result 사용 (수동 저장)
        const dataToSave = poemResult || result
        if (!dataToSave || !dataToSave.poem) return

        const poemData = {
            id: Date.now().toString(),
            poem: dataToSave.poem,
            keywords: dataToSave.keywords || [],
            emotion: dataToSave.emotion || '',
            emotion_confidence: dataToSave.emotion_confidence || 0,
            originalText: text.trim(),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        }

        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            savedPoems.unshift(poemData) // 최신 시가 맨 위에 오도록
            localStorage.setItem(STORAGE_KEY, JSON.stringify(savedPoems))
            setSaved(true)
        } catch (err) {
            console.error('시 저장 실패:', err)
            setError('시 저장에 실패했습니다.')
        }
    }

    return (
        <div className="px-6 sm:px-8 md:px-10 pt-4 sm:pt-6 md:pt-8 pb-4 sm:pb-6 md:pb-8 max-w-4xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-3">
                시 생성
            </h2>

            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <label 
                        htmlFor="text-input" 
                        className="block text-sm font-medium text-gray-800 mb-4"
                    >
                        일상글을 입력해주세요
                    </label>
                    <textarea
                        id="text-input"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="오늘 하루는 어떤 하루였나요? 당신의 일상을 들려주세요..."
                        className="w-full px-4 py-3 border border-gray-600 rounded-lg focus:outline-none focus:border-gray-600 resize-none text-gray-800"
                        rows="12"
                        disabled={loading}
                    />
                </div>

                {/* 모델 선택 */}
                <div className="rounded-lg p-4 border border-gray-600 bg-transparent">
                    <label className="block text-sm font-medium text-gray-800 mb-3">
                        모델 선택
                    </label>
                    <div className="flex gap-3">
                        <button
                            type="button"
                            onClick={() => {
                                setModelType('solar')
                                setUseTrainedModel(false)  // SOLAR 선택 시 기본 모델 사용
                            }}
                            disabled={loading}
                            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                                modelType === 'solar'
                                    ? 'bg-[#79A9E6] text-white'
                                    : 'bg-transparent border border-gray-600 text-gray-800 hover:bg-gray-50'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                            SOLAR (GPU)
                            <div className="text-xs mt-1 opacity-80">
                                {COLAB_API_URL ? 'Colab 연동' : '로컬 서버'}
                            </div>
                        </button>
                        <button
                            type="button"
                            onClick={() => {
                                setModelType('kogpt2')
                                setUseTrainedModel(true)  // koGPT2 선택 시 학습된 모델 자동 사용
                            }}
                            disabled={loading}
                            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                                modelType === 'kogpt2'
                                    ? 'bg-[#79A9E6] text-white'
                                    : 'bg-transparent border border-gray-600 text-gray-800 hover:bg-gray-50'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                            koGPT2 (CPU)
                            <div className="text-xs mt-1 opacity-80">학습된 모델 사용</div>
                        </button>
                    </div>
                    {!modelType && (
                        <p className="text-xs text-gray-600 mt-2">
                            모델을 선택하지 않으면 자동으로 GPU/CPU를 감지하여 선택됩니다.
                        </p>
                    )}
                    
                    {/* 학습된 모델 사용 상태 표시 */}
                    {modelType === 'kogpt2' && (
                        <div className="mt-4 pt-4 border-t border-gray-300">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-gray-800">
                                    ✅ 학습된 모델 사용 중
                                </span>
                            </div>
                            <p className="text-xs text-gray-600 mt-1">
                                Colab에서 학습한 모델로 시를 생성합니다. 산문의 의미를 이해하고 시로 변환합니다.
                            </p>
                        </div>
                    )}
                </div>

                {/* 프롬프트 옵션 */}
                <div className="rounded-lg p-4 border border-gray-600">
                    <button
                        type="button"
                        onClick={() => setShowOptions(!showOptions)}
                        className="w-full flex items-center justify-between text-sm font-medium text-gray-800"
                    >
                        <span>프롬프트 옵션 {showOptions ? '▼' : '▶'}</span>
                        <span className="text-xs text-gray-600">선택사항</span>
                    </button>
                    
                    {showOptions && (
                        <div className="mt-4 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                {/* 줄 수 */}
                                <div>
                                    <label className="block text-xs font-medium text-gray-800 mb-1">
                                        줄 수
                                    </label>
                                    <input
                                        type="number"
                                        min="2"
                                        max="20"
                                        value={lines}
                                        onChange={(e) => setLines(parseInt(e.target.value) || 4)}
                                        className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 text-sm"
                                        disabled={loading}
                                    />
                                </div>

                                {/* 분위기 */}
                                <div>
                                    <label className="block text-xs font-medium text-gray-800 mb-1">
                                        분위기
                                    </label>
                                    <CustomDropdown
                                        value={mood}
                                        onChange={setMood}
                                        options={[
                                            { value: '잔잔한', label: '잔잔한' },
                                            { value: '담담한', label: '담담한' },
                                            { value: '쓸쓸한', label: '쓸쓸한' },
                                            { value: '따뜻한', label: '따뜻한' },
                                            { value: '설레는', label: '설레는' },
                                            { value: '지친', label: '지친' },
                                        ]}
                                        placeholder="자동 분석"
                                        disabled={loading}
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                {/* 필수 키워드 */}
                                <div>
                                    <label className="block text-xs font-medium text-gray-800 mb-1">
                                        필수 키워드
                                    </label>
                                    <input
                                        type="text"
                                        value={requiredKeywords}
                                        onChange={(e) => setRequiredKeywords(e.target.value)}
                                        placeholder="쉼표로 구분"
                                        className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 text-sm"
                                        disabled={loading}
                                    />
                                </div>

                                {/* 금칙어 */}
                                <div>
                                    <label className="block text-xs font-medium text-gray-800 mb-1">
                                        금칙어
                                    </label>
                                    <input
                                        type="text"
                                        value={bannedWords}
                                        onChange={(e) => setBannedWords(e.target.value)}
                                        placeholder="쉼표로 구분"
                                        className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 text-sm"
                                        disabled={loading}
                                    />
                                </div>
                            </div>

                            {/* 운율 토글 */}
                            <div className="flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    id="use-rhyme"
                                    checked={useRhyme}
                                    onChange={(e) => setUseRhyme(e.target.checked)}
                                    className="w-4 h-4 text-[#79A9E6] border-gray-600 rounded focus:ring-gray-400"
                                    disabled={loading}
                                />
                                <label htmlFor="use-rhyme" className="text-xs font-medium text-gray-800">
                                    두운/두행두운(간단 운율) 사용
                                </label>
                            </div>
                        </div>
                    )}
                </div>

                <div className="flex gap-3">
                    <button
                        type="submit"
                        disabled={loading || !text.trim()}
                        className="px-6 py-3 bg-transparent border border-gray-800 text-gray-800 rounded-lg font-medium hover:bg-gray-50 hover:border-gray-600 disabled:cursor-not-allowed transition-colors"
                    >
                        {loading ? '시 생성 중...' : '시 생성하기'}
                    </button>
                    
                    {result && (
                        <button
                            type="button"
                            onClick={handleReset}
                            className="px-6 py-3 bg-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-400 transition-colors"
                        >
                            다시 작성
                        </button>
                    )}
                </div>
            </form>

            {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                    {error}
                </div>
            )}

            {result && (
                <div className="mt-8 space-y-6">
                    {/* 키워드 */}
                    {result.keywords && result.keywords.length > 0 && (
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">
                                추출된 키워드
                            </h3>
                            <div className="flex flex-wrap gap-2">
                                {result.keywords.map((keyword, index) => (
                                    <span
                                        key={index}
                                        className="px-3 py-1 bg-white border border-gray-600 text-gray-800 rounded-full text-sm"
                                    >
                                        {keyword}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* 감정 */}
                    {result.emotion && (
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">
                                감정 분석
                            </h3>
                            <div className="flex items-center gap-3">
                                <span className="px-4 py-2 bg-white border border-gray-600 text-gray-800 rounded-lg font-medium">
                                    {result.emotion}
                                </span>
                                <span className="text-sm text-gray-600">
                                    (신뢰도: {(result.emotion_confidence * 100).toFixed(1)}%)
                                </span>
                            </div>
                        </div>
                    )}

                    {/* 생성된 시 */}
                    {result.poem && (
                        <div>
                            <div className="flex items-center justify-between mb-3">
                                <h3 className="text-lg font-semibold text-gray-800">
                                생성된 시
                            </h3>
                                <button
                                    type="button"
                                    onClick={handleSavePoem}
                                    disabled={saved}
                                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                                        saved
                                            ? 'bg-green-100 text-green-700 cursor-not-allowed'
                                            : 'bg-[#79A9E6] text-white hover:bg-[#5A8FD6]'
                                    }`}
                                >
                                    {saved ? '✓ 보관함에 저장됨' : '보관함에 저장'}
                                </button>
                            </div>
                            <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                                <div className="whitespace-pre-line text-gray-800 leading-relaxed">
                                    {result.poem}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default PoemGeneration
