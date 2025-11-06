import React from 'react'

function About() {
    return (
        <div className="p-6 sm:p-8 md:p-10 max-w-5xl mx-auto">
            {/* 헤더 섹션 */}
            <div className="text-center mb-12">
                <h1 className="text-4xl sm:text-5xl font-semibold text-gray-800 mb-4">
                    시옷 (SIOT)
                </h1>
                <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                    당신의 일상을 아름다운 시로 변환하는 AI 시 생성 서비스
                </p>
            </div>

            {/* 소개 섹션 */}
            <div className="mb-12">
                <div className="bg-transparent border border-gray-600 rounded-lg p-8">
                    <h2 className="text-2xl font-semibold text-gray-800 mb-4">✨ 시옷이란?</h2>
                    <p className="text-gray-800 leading-relaxed mb-4">
                        시옷은 당신의 일상 이야기를 받아 AI가 아름다운 한국어 시로 변환해주는 웹 애플리케이션입니다. 
                        하루하루의 소소한 경험과 감정을 시적인 언어로 재해석하여, 
                        여러분만의 특별한 시를 만들어드립니다.
                    </p>
                    <p className="text-gray-800 leading-relaxed mb-4">
                        일기나 메모를 입력하기만 하면, AI가 키워드를 추출하고 감정을 분석하여 
                        자연스럽고 감성적인 시를 생성합니다. 생성된 시는 보관함에 저장하고 
                        언제든지 수정하거나 다시 읽어볼 수 있습니다.
                    </p>
                    <p className="text-gray-800 leading-relaxed">
                        시 한 줄을 책갈피에 끼워두고, 잠시 숨을 고르듯이 당신의 일상을 시로 담아보세요.
                    </p>
                </div>
            </div>

            {/* 주요 기능 섹션 */}
            <div className="mb-12">
                <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-6 text-center">
                    주요 기능
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* 기능 1 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <div className="text-3xl mb-4">🔍</div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">키워드 추출</h3>
                        <p className="text-gray-800 leading-relaxed">
                            TF-IDF 알고리즘을 사용하여 일상글에서 핵심 키워드를 자동으로 추출합니다. 
                            추출된 키워드는 시 생성의 핵심 재료가 됩니다.
                        </p>
                    </div>

                    {/* 기능 2 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <div className="text-3xl mb-4">💭</div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">감정 분석</h3>
                        <p className="text-gray-800 leading-relaxed">
                            XNLI 모델을 활용한 제로샷 감정 분류로 텍스트의 감정을 분석합니다. 
                            긍정, 중립, 부정의 감정을 분위기로 매핑하여 시의 톤을 결정합니다.
                        </p>
                    </div>

                    {/* 기능 3 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <div className="text-3xl mb-4">✍️</div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">AI 시 생성</h3>
                        <p className="text-gray-800 leading-relaxed">
                            SOLAR-10.7B-Instruct 모델을 사용하여 키워드와 감정을 바탕으로 
                            자연스럽고 아름다운 한국어 시를 생성합니다.
                        </p>
                    </div>

                    {/* 기능 4 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <div className="text-3xl mb-4">📚</div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">시 보관함</h3>
                        <p className="text-gray-800 leading-relaxed">
                            생성된 시들을 보관함에 저장하고 관리할 수 있습니다. 
                            언제든지 시를 수정하거나 삭제할 수 있어, 완벽한 시를 만들어가세요.
                        </p>
                    </div>
                </div>
            </div>

            {/* 사용 방법 섹션 */}
            <div className="mb-12">
                <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-6 text-center">
                    사용 방법
                </h2>
                <div className="space-y-4">
                    <div className="flex items-start gap-4 bg-transparent border border-gray-600 rounded-lg p-6">
                        <div className="flex-shrink-0 w-10 h-10 bg-[#79A9E6] text-white rounded-full flex items-center justify-center font-bold">
                            1
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">일상글 입력</h3>
                            <p className="text-gray-800">
                                "시 생성" 페이지에서 오늘 하루 있었던 일이나 느낀 점을 자유롭게 작성해주세요. 
                                짧은 문장부터 긴 이야기까지 어떤 형식이어도 괜찮습니다.
                            </p>
                        </div>
                    </div>

                    <div className="flex items-start gap-4 bg-transparent border border-gray-600 rounded-lg p-6">
                        <div className="flex-shrink-0 w-10 h-10 bg-[#79A9E6] text-white rounded-full flex items-center justify-center font-bold">
                            2
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">시 생성하기</h3>
                            <p className="text-gray-800">
                                "시 생성하기" 버튼을 클릭하면 AI가 자동으로 키워드를 추출하고 감정을 분석한 후, 
                                몇 초 안에 아름다운 시를 생성해드립니다.
                            </p>
                        </div>
                    </div>

                    <div className="flex items-start gap-4 bg-transparent border border-gray-600 rounded-lg p-6">
                        <div className="flex-shrink-0 w-10 h-10 bg-[#79A9E6] text-white rounded-full flex items-center justify-center font-bold">
                            3
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">시 저장 및 관리</h3>
                            <p className="text-gray-800">
                                생성된 시가 마음에 드시면 "보관함에 저장" 버튼을 눌러 저장하세요. 
                                "시 보관함"에서 저장된 모든 시를 확인하고, 원하면 수정하거나 삭제할 수 있습니다.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 기술 스택 섹션 */}
            <div className="mb-12">
                <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-6 text-center">
                    기술 스택
                </h2>
                <div className="bg-transparent border border-gray-600 rounded-lg p-8">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">프론트엔드</h3>
                            <ul className="space-y-2 text-gray-800">
                                <li>• React + Vite</li>
                                <li>• Tailwind CSS</li>
                                <li>• React Router</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">백엔드</h3>
                            <ul className="space-y-2 text-gray-800">
                                <li>• FastAPI</li>
                                <li>• Python 3.8+</li>
                                <li>• PyTorch</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">AI 모델</h3>
                            <ul className="space-y-2 text-gray-800">
                                <li>• SOLAR-10.7B-Instruct (시 생성)</li>
                                <li>• XNLI (감정 분석)</li>
                                <li>• TF-IDF (키워드 추출)</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">번역 서비스</h3>
                            <ul className="space-y-2 text-gray-800">
                                <li>• Google Cloud Translation API v3</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {/* 정보 섹션 */}
            <div className="bg-transparent border border-gray-600 rounded-lg p-8 text-center">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">프로젝트 정보</h2>
                <p className="text-gray-800 mb-4">
                    시옷(SIOT)은 오픈소스 프로젝트입니다. 
                    GitHub에서 소스코드를 확인하고 기여할 수 있습니다.
                </p>
                <div className="flex flex-wrap justify-center gap-4 text-sm text-gray-600">
                    <span>✨ 오픈소스</span>
                    <span>•</span>
                    <span>🚀 지속적인 업데이트</span>
                    <span>•</span>
                    <span>💡 커뮤니티 기여 환영</span>
                </div>
            </div>
        </div>
    )
}

export default About
