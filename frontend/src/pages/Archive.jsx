import React, { useState, useEffect } from 'react'

const STORAGE_KEY = 'saved_poems'

function Archive() {
    const [poems, setPoems] = useState([])
    const [selectedPoem, setSelectedPoem] = useState(null)
    const [isEditing, setIsEditing] = useState(false)
    const [editedPoem, setEditedPoem] = useState('')

    useEffect(() => {
        loadPoems()
    }, [])

    const loadPoems = () => {
        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            setPoems(savedPoems)
        } catch (err) {
            console.error('시 목록 불러오기 실패:', err)
        }
    }

    const handlePoemClick = (poem) => {
        setSelectedPoem(poem)
        setEditedPoem(poem.poem)
        setIsEditing(false)
    }

    const handleEdit = () => {
        setIsEditing(true)
    }

    const handleSaveEdit = () => {
        if (!selectedPoem || !editedPoem.trim()) return

        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            const updatedPoems = savedPoems.map(p => 
                p.id === selectedPoem.id
                    ? { ...p, poem: editedPoem.trim(), updatedAt: new Date().toISOString() }
                    : p
            )
            localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedPoems))
            
            const updatedPoem = updatedPoems.find(p => p.id === selectedPoem.id)
            setSelectedPoem(updatedPoem)
            setPoems(updatedPoems)
            setIsEditing(false)
        } catch (err) {
            console.error('시 수정 실패:', err)
            alert('시 수정에 실패했습니다.')
        }
    }

    const handleCancelEdit = () => {
        setEditedPoem(selectedPoem.poem)
        setIsEditing(false)
    }

    const handleDelete = (poemId) => {
        if (!confirm('정말 이 시를 삭제하시겠습니까?')) return

        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            const filteredPoems = savedPoems.filter(p => p.id !== poemId)
            localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredPoems))
            
            setPoems(filteredPoems)
            if (selectedPoem && selectedPoem.id === poemId) {
                setSelectedPoem(null)
                setIsEditing(false)
            }
        } catch (err) {
            console.error('시 삭제 실패:', err)
            alert('시 삭제에 실패했습니다.')
        }
    }

    const handleCloseDetail = () => {
        setSelectedPoem(null)
        setIsEditing(false)
    }

    const formatDate = (dateString) => {
        const date = new Date(dateString)
        return date.toLocaleDateString('ko-KR', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    return (
        <div className="px-6 sm:px-8 md:px-10 pt-4 sm:pt-6 md:pt-8 pb-4 sm:pb-6 md:pb-8 max-w-7xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-3">시 보관함</h2>
            
            {poems.length === 0 ? (
                <div className="text-center py-16">
                    <div className="mb-4">
                        <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                    </div>
                    <p className="text-gray-600 text-lg font-medium">보관된 시가 없습니다</p>
                    <p className="text-gray-500 mt-2 text-sm">시를 생성하고 보관함에 저장해보세요</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {poems.map((poem) => (
                        <div
                            key={poem.id}
                            onClick={() => handlePoemClick(poem)}
                            className="p-6 bg-transparent border border-gray-600 rounded-lg cursor-pointer hover:border-gray-700 transition-all duration-200 flex flex-col"
                        >
                            <div className="flex-1 mb-4">
                                <div className="text-gray-800 font-medium line-clamp-4 whitespace-pre-line leading-relaxed">
                                    {poem.poem}
                                </div>
                            </div>
                            <div className="flex items-center justify-between text-sm text-gray-600 pt-4 border-t border-gray-600">
                                <span className="text-xs">{formatDate(poem.updatedAt || poem.createdAt)}</span>
                                {poem.emotion && (
                                    <span className="px-2 py-1 bg-transparent border border-gray-600 text-gray-800 rounded-full text-xs">
                                        {poem.emotion}
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* 상세보기/수정 모달 */}
            {selectedPoem && (
                <div className="fixed inset-0 bg-white/20 backdrop-blur-md flex items-center justify-center z-50 p-4">
                    <div className="bg-white/30 backdrop-blur-xl border border-gray-600 rounded-lg max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6 custom-scrollbar">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-semibold text-gray-800">시 상세보기</h3>
                            <button
                                onClick={handleCloseDetail}
                                className="text-gray-600 hover:text-gray-800 text-2xl"
                            >
                                ×
                            </button>
                        </div>

                        {/* 원본 일상글 */}
                        {selectedPoem.originalText && (
                            <div className="mb-4">
                                <h4 className="text-sm font-medium text-gray-800 mb-2">원본 일상글</h4>
                                <div className="p-3 bg-transparent border border-gray-600 rounded text-gray-800 text-sm">
                                    {selectedPoem.originalText}
                                </div>
                            </div>
                        )}

                        {/* 키워드 */}
                        {selectedPoem.keywords && selectedPoem.keywords.length > 0 && (
                            <div className="mb-4">
                                <h4 className="text-sm font-medium text-gray-800 mb-2">키워드</h4>
                                <div className="flex flex-wrap gap-2">
                                    {selectedPoem.keywords.map((keyword, index) => (
                                        <span
                                            key={index}
                                            className="px-3 py-1 bg-transparent border border-gray-600 text-gray-800 rounded-full text-sm"
                                        >
                                            {keyword}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* 감정 */}
                        {selectedPoem.emotion && (
                            <div className="mb-4">
                                <h4 className="text-sm font-medium text-gray-800 mb-2">감정 분석</h4>
                                <div className="flex items-center gap-3">
                                    <span className="px-4 py-2 bg-transparent border border-gray-600 text-gray-800 rounded-lg font-medium">
                                        {selectedPoem.emotion}
                                    </span>
                                    {selectedPoem.emotion_confidence && (
                                        <span className="text-sm text-gray-600">
                                            (신뢰도: {(selectedPoem.emotion_confidence * 100).toFixed(1)}%)
                                        </span>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* 시 내용 */}
                        <div className="mb-4">
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-sm font-medium text-gray-800">시</h4>
                                {!isEditing && (
                                    <button
                                        onClick={handleEdit}
                                        className="px-4 py-2 bg-[#79A9E6] text-white rounded-lg text-sm hover:bg-[#5A8FD6] transition-colors"
                                    >
                                        수정하기
                                    </button>
                                )}
                            </div>
                            {isEditing ? (
                                <div>
                                    <textarea
                                        value={editedPoem}
                                        onChange={(e) => setEditedPoem(e.target.value)}
                                        className="w-full px-4 py-3 border border-gray-600 rounded-lg focus:outline-none focus:border-gray-600 resize-none text-gray-800"
                                        rows="10"
                                    />
                                    <div className="flex gap-2 mt-3">
                                        <button
                                            onClick={handleSaveEdit}
                                            className="px-4 py-2 bg-[#79A9E6] text-white rounded-lg hover:bg-[#5A8FD6] transition-colors"
                                        >
                                            저장하기
                                        </button>
                                        <button
                                            onClick={handleCancelEdit}
                                            className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
                                        >
                                            취소
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                                    <div className="whitespace-pre-line text-gray-800 leading-relaxed">
                                        {selectedPoem.poem}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 날짜 정보 */}
                        <div className="text-sm text-gray-600 mb-4">
                            <p>생성일: {formatDate(selectedPoem.createdAt)}</p>
                            {selectedPoem.updatedAt && selectedPoem.updatedAt !== selectedPoem.createdAt && (
                                <p>수정일: {formatDate(selectedPoem.updatedAt)}</p>
                            )}
                        </div>

                        {/* 삭제 버튼 */}
                        <div className="flex justify-end">
                            <button
                                onClick={() => handleDelete(selectedPoem.id)}
                                className="px-4 py-2 bg-red-400 text-white rounded-lg text-sm hover:bg-red-500 transition-colors"
                            >
                                삭제하기
                            </button>
                        </div>
                    </div>
            </div>
            )}
        </div>
    )
}

export default Archive
