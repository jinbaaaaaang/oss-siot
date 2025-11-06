import React, { useState, useEffect } from 'react'

const STORAGE_KEY = 'saved_poems'

function Settings() {
    const [poemCount, setPoemCount] = useState(0)
    const [showResetConfirm, setShowResetConfirm] = useState(false)

    useEffect(() => {
        loadPoemCount()
    }, [])

    const loadPoemCount = () => {
        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            setPoemCount(savedPoems.length)
        } catch (err) {
            console.error('시 개수 불러오기 실패:', err)
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
            </div>
        </div>
    )
}

export default Settings
