import React, { useState, useEffect, useRef } from 'react'

const STORAGE_KEY = 'saved_poems'
const SETTINGS_KEY = 'app_settings'

// 커스텀 드롭다운 컴포넌트 (PoemGeneration과 동일한 스타일)
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
                className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 focus:border-gray-600 text-sm text-left flex items-center justify-between cursor-pointer hover:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-transparent"
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

function Settings() {
    const [poemCount, setPoemCount] = useState(0)
    const [showResetConfirm, setShowResetConfirm] = useState(false)
    
    // 기본 설정 상태
    const [defaultModelType, setDefaultModelType] = useState('')
    const [autoSave, setAutoSave] = useState(true)

    useEffect(() => {
        loadPoemCount()
        loadSettings()
    }, [])

    const loadPoemCount = () => {
        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            setPoemCount(savedPoems.length)
        } catch (err) {
            console.error('시 개수 불러오기 실패:', err)
        }
    }

    const loadSettings = () => {
        try {
            const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
            setDefaultModelType(settings.defaultModelType || '')
            setAutoSave(settings.autoSave !== undefined ? settings.autoSave : true)
        } catch (err) {
            console.error('설정 불러오기 실패:', err)
        }
    }

    const saveSettings = () => {
        try {
            const settings = {
                defaultModelType,
                autoSave
            }
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
            alert('설정이 저장되었습니다.')
        } catch (err) {
            console.error('설정 저장 실패:', err)
            alert('설정 저장에 실패했습니다.')
        }
    }

    const resetSettings = () => {
        if (window.confirm('기본 설정을 초기화하시겠습니까?')) {
            setDefaultModelType('')
            setAutoSave(true)
            localStorage.removeItem(SETTINGS_KEY)
            alert('설정이 초기화되었습니다.')
        }
    }

    const handleResetArchive = () => {
        if (!showResetConfirm) {
            setShowResetConfirm(true)
            return
        }

        try {
            localStorage.setItem(STORAGE_KEY, '[]')
            setPoemCount(0)
            setShowResetConfirm(false)
            alert('보관함이 초기화되었습니다.')
        } catch (err) {
            console.error('보관함 초기화 실패:', err)
            alert('보관함 초기화에 실패했습니다.')
        }
    }

    const handleExportData = () => {
        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            const dataStr = JSON.stringify(savedPoems, null, 2)
            const dataBlob = new Blob([dataStr], { type: 'application/json' })
            const url = URL.createObjectURL(dataBlob)
            const link = document.createElement('a')
            link.href = url
            link.download = `시옷_보관함_${new Date().toISOString().split('T')[0]}.json`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            URL.revokeObjectURL(url)
            alert('데이터가 내보내기되었습니다.')
        } catch (err) {
            console.error('데이터 내보내기 실패:', err)
            alert('데이터 내보내기에 실패했습니다.')
        }
    }

    const handleImportData = () => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = 'application/json'
        input.onchange = (e) => {
            const file = e.target.files[0]
            if (!file) return

            const reader = new FileReader()
            reader.onload = (event) => {
                try {
                    const importedData = JSON.parse(event.target.result)
                    if (!Array.isArray(importedData)) {
                        alert('올바른 형식의 파일이 아닙니다.')
                        return
                    }

                    const existingPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
                    const mergedPoems = [...existingPoems, ...importedData]
                    localStorage.setItem(STORAGE_KEY, JSON.stringify(mergedPoems))
                    loadPoemCount()
                    alert(`${importedData.length}개의 시가 가져와졌습니다.`)
                } catch (err) {
                    console.error('데이터 가져오기 실패:', err)
                    alert('데이터 가져오기에 실패했습니다. 파일 형식을 확인해주세요.')
                }
            }
            reader.readAsText(file)
        }
        input.click()
    }

    return (
        <div className="px-6 sm:px-8 md:px-10 pt-4 sm:pt-6 md:pt-8 pb-4 sm:pb-6 md:pb-8 max-w-4xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-8">설정</h2>
            
            {/* 보관함 관리 */}
            <div className="space-y-6">
                <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">보관함 관리</h3>
                    <div className="space-y-4">
                        <div className="p-4 bg-transparent border border-gray-600 rounded-lg">
                            <p className="text-sm text-gray-600 mb-4">
                                현재 보관된 시: <span className="font-semibold text-gray-800">{poemCount}개</span>
                            </p>
                            
                            <div className="space-y-3">
                                {/* 보관함 초기화 */}
                                <div>
                                    {!showResetConfirm ? (
                                        <button
                                            onClick={handleResetArchive}
                                            className="w-full px-4 py-2 bg-[#79A9E6] text-white rounded-lg font-medium hover:bg-[#5A8FD6] transition-colors"
                                        >
                                            보관함 초기화
                                        </button>
                                    ) : (
                                        <div className="space-y-2">
                                            <p className="text-sm text-gray-600">정말 보관함을 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다.</p>
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={handleResetArchive}
                                                    className="flex-1 px-4 py-2 bg-[#79A9E6] text-white rounded-lg font-medium hover:bg-[#5A8FD6] transition-colors"
                                                >
                                                    확인
                                                </button>
                                                <button
                                                    onClick={() => setShowResetConfirm(false)}
                                                    className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-400 transition-colors"
                                                >
                                                    취소
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                                
                                {/* 데이터 내보내기 */}
                                <div>
                                    <button
                                        onClick={handleExportData}
                                        className="w-full px-4 py-2 bg-transparent border border-gray-600 text-gray-800 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                                    >
                                        데이터 내보내기
                                    </button>
                                </div>
                                
                                {/* 데이터 가져오기 */}
                                <div>
                                    <button
                                        onClick={handleImportData}
                                        className="w-full px-4 py-2 bg-transparent border border-gray-600 text-gray-800 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                                    >
                                        데이터 가져오기
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* 기본 생성 설정 */}
                <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">기본 생성 설정</h3>
                    <div className="p-4 bg-transparent border border-gray-600 rounded-lg space-y-4">
                        {/* 기본 모델 타입 */}
                        <div>
                            <label className="block text-sm font-medium text-gray-800 mb-2">
                                기본 모델 타입
                            </label>
                            <CustomDropdown
                                value={defaultModelType}
                                onChange={setDefaultModelType}
                                options={[
                                    { value: '', label: '자동 선택 (GPU/CPU 감지)' },
                                    { value: 'solar', label: 'SOLAR (GPU, 고품질)' },
                                    { value: 'kogpt2', label: 'koGPT2 (CPU, 빠른 생성)' }
                                ]}
                                placeholder="자동 선택 (GPU/CPU 감지)"
                            />
                            <p className="text-xs text-gray-500 mt-1">시 생성 시 기본으로 사용할 모델입니다</p>
                            
                            {/* SOLAR 모델 사용 방법 안내 */}
                            {defaultModelType === 'solar' && (
                                <div className="mt-4 p-4 bg-transparent border border-gray-600 rounded-lg">
                                    <h4 className="text-sm font-semibold text-gray-800 mb-3">SOLAR 모델 사용 방법</h4>
                                    <div className="space-y-3 text-sm text-gray-800">
                                        <div>
                                            <p className="font-medium mb-2">1. Google Colab에서 서버 실행</p>
                                            <ul className="list-disc list-inside space-y-1 ml-2 text-gray-700">
                                                <li>Colab 노트북 생성 및 GPU 활성화</li>
                                                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded font-mono text-xs">colab_server.py</code> 실행</li>
                                                <li>ngrok 토큰 설정 (<a href="https://ngrok.com" target="_blank" rel="noopener noreferrer" className="text-[#79A9E6] hover:underline">https://ngrok.com</a>에서 발급)</li>
                                                <li>생성된 ngrok URL 복사</li>
                                            </ul>
                                        </div>
                                        <div>
                                            <p className="font-medium mb-2">2. 프론트엔드 환경 변수 설정</p>
                                            <p className="text-gray-700 mb-1">프로젝트 루트에 <code className="bg-gray-100 px-1.5 py-0.5 rounded font-mono text-xs">.env</code> 파일 생성:</p>
                                            <div className="bg-gray-900 text-green-400 p-2 rounded font-mono text-xs overflow-x-auto border border-gray-700">
                                                <span className="text-gray-400">VITE_COLAB_API_URL=</span>
                                                <span className="text-green-300">https://xxxx-xxxx.ngrok.io</span>
                                            </div>
                                        </div>
                                        <div>
                                            <p className="font-medium mb-1">3. 프론트엔드 재시작</p>
                                            <p className="text-gray-700">프론트엔드를 재시작하면 SOLAR 모델이 자동으로 사용됩니다.</p>
                                        </div>
                                        <div className="pt-2 border-t border-gray-300">
                                            <p className="text-xs text-gray-600">
                                                💡 자세한 내용은 <code className="bg-gray-100 px-1 py-0.5 rounded font-mono">COLAB_FRONTEND_GUIDE.md</code> 파일을 참고하세요.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            {/* koGPT2 모델 사용 방법 안내 */}
                            {defaultModelType === 'kogpt2' && (
                                <div className="mt-4 p-4 bg-transparent border border-gray-600 rounded-lg">
                                    <h4 className="text-sm font-semibold text-gray-800 mb-3">koGPT2 모델 사용 방법</h4>
                                    <div className="space-y-3 text-sm text-gray-800">
                                        <div>
                                            <p className="font-medium mb-2">로컬 백엔드 서버 실행</p>
                                            <ul className="list-disc list-inside space-y-1 ml-2 text-gray-700">
                                                <li>터미널에서 <code className="bg-gray-100 px-1.5 py-0.5 rounded font-mono text-xs">cd backend</code> 실행</li>
                                                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded font-mono text-xs">./start.sh</code> 실행</li>
                                                <li>학습된 모델이 <code className="bg-gray-100 px-1.5 py-0.5 rounded font-mono text-xs">trained_models/</code> 폴더에 있어야 합니다</li>
                                            </ul>
                                        </div>
                                        <div className="pt-2 border-t border-gray-300">
                                            <p className="text-xs text-gray-600">
                                                💡 학습된 모델이 없으면 기본 koGPT2 모델이 사용됩니다.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 자동 저장 */}
                        <div>
                            <label className="flex items-center gap-3 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={autoSave}
                                    onChange={(e) => setAutoSave(e.target.checked)}
                                    className="w-5 h-5 text-[#79A9E6] border-gray-600 rounded focus:ring-2 focus:ring-[#79A9E6]"
                                />
                                <div>
                                    <span className="text-sm font-medium text-gray-800">시 생성 후 자동 저장</span>
                                    <p className="text-xs text-gray-500">시가 생성되면 자동으로 보관함에 저장됩니다</p>
                                </div>
                            </label>
                        </div>

                        {/* 설정 저장 버튼 */}
                        <div className="flex gap-2 pt-2">
                            <button
                                onClick={saveSettings}
                                className="flex-1 px-4 py-2 bg-[#79A9E6] text-white rounded-lg font-medium hover:bg-[#5A8FD6] transition-colors"
                            >
                                설정 저장
                            </button>
                            <button
                                onClick={resetSettings}
                                className="px-4 py-2 bg-transparent border border-gray-600 text-gray-800 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                            >
                                초기화
                            </button>
                        </div>
                    </div>
                </div>

                {/* 앱 정보 */}
                <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">앱 정보</h3>
                    <div className="p-4 bg-transparent border border-gray-600 rounded-lg">
                        <div className="space-y-2 text-sm text-gray-600">
                            <p><span className="font-medium text-gray-800">시옷 (SIOT)</span></p>
                            <p>일상글을 시로 변환하는 웹 애플리케이션</p>
                            <p className="text-xs text-gray-500 mt-4">
                                AI 기반 한국어 시 생성 서비스
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Settings
