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
                    <h2 className="text-2xl font-semibold text-gray-800 mb-6">시옷이란?</h2>
                    
                    <div className="space-y-6">
                        <div>
                            <p className="text-gray-800 leading-relaxed text-lg mb-3">
                                시옷(SIOT)은 당신의 일상 이야기를 아름다운 한국어 시로 변환해주는 AI 웹 애플리케이션입니다.
                            </p>
                            <div>
                            <p className="text-gray-700 leading-relaxed">
                                아침에 마신 커피 한 잔, 지하철 창밖으로 스쳐 지나간 풍경,
                                저녁 노을을 바라보며 느꼈던 그 감정까지.<br />
                                하루하루의 소소한 순간들이 시적인 언어로 재탄생합니다.
                            </p>
                            </div>
                        </div>

                        <div className="py-4">
                            <p className="text-gray-800 leading-relaxed text-lg">
                                "일상의 말들이 시가 되는 순간, <br />
                                당신의 하루는 문학이 됩니다."
                            </p>
                        </div>

                        <div>
                            <div>
                            <p className="text-gray-700 leading-relaxed mb-3">
                                일기나 메모를 입력하기만 하면, AI가 자동으로 키워드를 추출하고 감정을 분석합니다. <br />
                                그렇게 생성된 시는 보관함에 저장되어 언제든지 다시 읽고 수정할 수 있으며, 
                                시간에 따른 감정의 변화도 시각화하여 확인할 수 있습니다.
                            </p>
                            </div>
                            <p className="text-gray-700 leading-relaxed">
                                시옷은 다양한 AI 모델과 API를 목적에 맞게 활용합니다. 
                                GPU 환경에서는 고품질 SOLAR 모델을, CPU 환경에서는 파인튜닝된 koGPT2 모델을 자동으로 선택하여 
                                최적의 성능을 제공합니다. 또한 Gemini API를 활용하여 감정 분석 결과를 사용자 친화적으로 표현하고, 
                                프롬프트 옵션을 적용한 경우 시 품질을 개선합니다.
                            </p>
                        </div>

                        <div className="pt-4 border-t border-gray-300">
                            <p className="text-gray-600 leading-relaxed">
                                시 한 줄을 책갈피에 끼워두고,<br />
                                잠시 숨을 고르듯이 당신의 일상을 시로 담아보세요.
                            </p>
                        </div>
                    </div>
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
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">키워드 추출</h3>
                        <p className="text-gray-800 leading-relaxed">
                            TF-IDF 알고리즘을 사용하여 일상글에서 핵심 키워드를 자동으로 추출합니다. 
                            추출된 키워드는 시 생성의 핵심 재료가 됩니다.
                        </p>
                    </div>

                    {/* 기능 2 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">감정 분석</h3>
                        <p className="text-gray-800 leading-relaxed">
                            XNLI 모델을 활용한 제로샷 감정 분류로 텍스트의 감정을 분석합니다. 
                            긍정, 중립, 부정의 감정을 분위기로 매핑하여 시의 톤을 결정합니다.
                        </p>
                    </div>

                    {/* 기능 3 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">AI 시 생성</h3>
                        <p className="text-gray-800 leading-relaxed mb-2">
                            두 가지 AI 모델을 지원합니다:
                        </p>
                        <ul className="text-gray-700 text-sm space-y-1 mb-2">
                            <li>• <strong>SOLAR-10.7B-Instruct</strong>: GPU 환경에서 고품질 시 생성 (약 10.7B 파라미터)</li>
                            <li>• <strong>koGPT2</strong>: CPU 환경에서 빠른 시 생성 (약 124M 파라미터)</li>
                        </ul>
                        <p className="text-gray-800 leading-relaxed text-sm">
                            시스템이 자동으로 환경을 감지하여 최적의 모델을 선택합니다.
                        </p>
                    </div>

                    {/* 기능 4 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">모델 파인튜닝</h3>
                        <p className="text-gray-800 leading-relaxed">
                            koGPT2 모델을 KPoEM 데이터셋으로 파인튜닝하여 더욱 시다운 시를 생성할 수 있습니다. 
                            k-fold 교차 검증을 통해 모델 성능을 평가하고 최적의 모델을 선택합니다.
                            학습된 모델은 로컬에서 추론하여 빠르고 효율적인 시 생성이 가능합니다.
                        </p>
                    </div>
                    
                    {/* 기능 7 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">Gemini API 활용</h3>
                        <p className="text-gray-800 leading-relaxed">
                            Gemini API를 활용하여 감정 분석 결과를 자연스럽고 따뜻한 스토리로 변환합니다. 
                            또한 프롬프트 옵션(줄 수, 분위기, 필수 키워드 등)을 적용한 경우, 
                            생성된 시를 Gemini로 개선하여 더 나은 품질을 제공합니다.
                        </p>
                    </div>

                    {/* 기능 5 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">시 보관함</h3>
                        <p className="text-gray-800 leading-relaxed">
                            생성된 시들을 보관함에 저장하고 관리할 수 있습니다. 
                            언제든지 시를 수정하거나 삭제할 수 있으며, 
                            데이터를 내보내거나 가져올 수 있습니다.
                        </p>
                    </div>

                    {/* 기능 6 */}
                    <div className="bg-transparent border border-gray-600 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <h3 className="text-xl font-semibold text-gray-800 mb-3">감정 추이 시각화</h3>
                        <p className="text-gray-800 leading-relaxed">
                            생성된 시들의 감정 변화를 차트로 확인할 수 있습니다. 
                            최근 7일 감정 추이, 감정 분포, 신뢰도 분포 등 다양한 시각화를 제공하여 
                            시간에 따른 감정 변화를 한눈에 파악할 수 있습니다.
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
                            <ul className="space-y-2 text-gray-800 text-sm">
                                <li>• <strong>React 19.1.1</strong> + <strong>Vite 7.1.7</strong></li>
                                <li>• <strong>Tailwind CSS 4.1.16</strong> (스타일링)</li>
                                <li>• <strong>React Router DOM 7.9.5</strong> (라우팅)</li>
                                <li>• <strong>Recharts 3.3.0</strong> (데이터 시각화)</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">백엔드</h3>
                            <ul className="space-y-2 text-gray-800 text-sm">
                                <li>• <strong>FastAPI 0.120.3</strong> (웹 프레임워크)</li>
                                <li>• <strong>Uvicorn</strong> (ASGI 서버)</li>
                                <li>• <strong>Python 3.8+</strong></li>
                                <li>• <strong>PyTorch 2.0+</strong> (딥러닝 프레임워크)</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">AI 모델</h3>
                            <ul className="space-y-2 text-gray-800 text-sm">
                                <li>• <strong>SOLAR-10.7B-Instruct</strong> (시 생성, GPU 권장)</li>
                                <li>• <strong>koGPT2-base-v2</strong> (시 생성, CPU 친화적)</li>
                                <li>• <strong>XNLI (xlm-roberta-large-xnli)</strong> (감정 분석)</li>
                                <li>• <strong>TF-IDF (scikit-learn)</strong> (키워드 추출)</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">기타 서비스</h3>
                            <ul className="space-y-2 text-gray-800 text-sm">
                                <li>• <strong>Google Cloud Translation API v3</strong> (중국어 번역, 선택)</li>
                                <li>• <strong>Gemini API</strong> (감정 분석 후처리, 시 개선, 선택)</li>
                                <li>• <strong>Hugging Face Transformers</strong> (모델 로딩)</li>
                                <li>• <strong>Google Colab</strong> (GPU 환경 제공, 선택)</li>
                                <li>• <strong>ngrok</strong> (Colab 서버 터널링)</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {/* 모델 선택 전략 섹션 */}
            <div className="mb-12">
                <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-6 text-center">
                    모델 선택 전략
                </h2>
                <div className="bg-transparent border border-gray-600 rounded-lg p-8">
                    <div className="space-y-6">
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">자동 모델 선택</h3>
                            <p className="text-gray-800 leading-relaxed mb-3">
                                시옷은 시스템 환경을 자동으로 감지하여 최적의 모델을 선택합니다:
                            </p>
                            <ul className="list-disc list-inside space-y-2 text-gray-700 ml-4">
                                <li><strong>GPU 감지 시</strong>: SOLAR-10.7B-Instruct 모델 자동 선택 (고품질, 약 10.7B 파라미터)</li>
                                <li><strong>CPU만 사용 가능 시</strong>: koGPT2 모델 자동 선택 (빠른 생성, 약 124M 파라미터)</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">모델 비교</h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-gray-800 border-collapse">
                                    <thead>
                                        <tr className="border-b border-gray-300">
                                            <th className="text-left p-2">항목</th>
                                            <th className="text-left p-2">SOLAR-10.7B</th>
                                            <th className="text-left p-2">koGPT2</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr className="border-b border-gray-200">
                                            <td className="p-2 font-medium">파라미터 수</td>
                                            <td className="p-2">약 10.7B</td>
                                            <td className="p-2">약 124M</td>
                                        </tr>
                                        <tr className="border-b border-gray-200">
                                            <td className="p-2 font-medium">모델 크기</td>
                                            <td className="p-2">약 21GB</td>
                                            <td className="p-2">약 500MB</td>
                                        </tr>
                                        <tr className="border-b border-gray-200">
                                            <td className="p-2 font-medium">권장 환경</td>
                                            <td className="p-2">GPU (6-8GB VRAM)</td>
                                            <td className="p-2">CPU / GPU 모두 가능</td>
                                        </tr>
                                        <tr className="border-b border-gray-200">
                                            <td className="p-2 font-medium">생성 품질</td>
                                            <td className="p-2">매우 높음</td>
                                            <td className="p-2">보통</td>
                                        </tr>
                                        <tr>
                                            <td className="p-2 font-medium">생성 속도</td>
                                            <td className="p-2">빠름 (GPU 기준)</td>
                                            <td className="p-2">중간 (CPU 기준)</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-3">AI 모델 활용 전략</h3>
                            <p className="text-gray-800 leading-relaxed mb-2">
                                시옷은 다양한 AI 모델과 API를 목적에 맞게 활용하는 하이브리드 접근 방식을 채택했습니다:
                            </p>
                            <ul className="list-disc list-inside space-y-2 text-gray-700 ml-4 text-sm">
                                <li><strong>시 생성: koGPT2 모델 파인튜닝</strong>
                                    <ul className="list-circle list-inside ml-4 mt-1 space-y-1">
                                        <li>koGPT2 모델을 KPoEM 데이터셋으로 파인튜닝하여 시 생성 능력 향상</li>
                                        <li>k-fold 교차 검증을 통한 모델 평가</li>
                                        <li>학습된 모델을 로컬에서 추론</li>
                                    </ul>
                                </li>
                                <li><strong>감정 분석 후처리: Gemini API 활용</strong>
                                    <ul className="list-circle list-inside ml-4 mt-1 space-y-1">
                                        <li>감정 데이터를 사용자 친화적인 스토리로 변환</li>
                                        <li>사전 학습된 모델의 자연어 생성 능력 활용</li>
                                    </ul>
                                </li>
                                <li><strong>시 개선: Gemini API 활용</strong>
                                    <ul className="list-circle list-inside ml-4 mt-1 space-y-1">
                                        <li>프롬프트 옵션 적용 시 생성된 시를 Gemini로 개선</li>
                                        <li>불필요한 텍스트 제거 및 시적 표현 개선</li>
                                    </ul>
                                </li>
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
                    <span>오픈소스</span>
                    <span>•</span>
                    <span>지속적인 업데이트</span>
                    <span>•</span>
                    <span>커뮤니티 기여 환영</span>
                </div>
            </div>
        </div>
    )
}

export default About
