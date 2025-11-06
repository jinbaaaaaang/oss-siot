# -*- coding: utf-8 -*-
from typing import List, Optional
import time
import asyncio
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# .env 파일 로드 (프로젝트 루트 또는 backend 디렉토리에 있을 수 있음)
env_path = Path(__file__).parent.parent.parent / ".env"  # 프로젝트 루트
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"  # backend 디렉토리
if env_path.exists():
    load_dotenv(env_path)
    print(f"[Config] .env 파일 로드됨: {env_path}")

from app.services.keyword_extractor import extract_keywords
from app.services.emotion_classifier import classify_emotion
from app.services.poem_generator import generate_poem_from_keywords
from app.services.poem_model_loader import _load_poem_model

app = FastAPI(title="Poem API (SOLAR Instruct, Colab GPU)")

# 터널/프론트 개발 환경 다양성을 위해 CORS는 와일드카드 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 필요 시 특정 도메인으로 좁히세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 모델을 미리 로드합니다.
    첫 요청 시 지연 시간을 줄이기 위해 사전 로딩합니다.
    """
    print("\n" + "="*80)
    print("🚀 서버 시작 중: 모델 사전 로딩 시작...")
    print("="*80)
    
    try:
        # 모델 로딩 (백그라운드 스레드에서 실행)
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, _load_poem_model)
        print("="*80)
        print("✅ 모델 사전 로딩 완료! 첫 요청부터 빠르게 응답할 수 있습니다.")
        print("="*80 + "\n")
    except Exception as e:
        print(f"⚠️ 모델 사전 로딩 실패: {e}")
        print("   (첫 요청 시 자동으로 로드됩니다.)\n")
        import traceback
        traceback.print_exc()

class PoemRequest(BaseModel):
    text: str
    lines: Optional[int] = None  # 줄 수 (행)
    mood: Optional[str] = None  # 분위기 (잔잔/담담/쓸쓸)
    required_keywords: Optional[List[str]] = None  # 필수 키워드
    banned_words: Optional[List[str]] = None  # 금칙어
    use_rhyme: Optional[bool] = False  # 두운/두행두운 운율 사용 여부
    acrostic: Optional[str] = None  # 아크로스틱 (예: "사랑해")
    model_type: Optional[str] = None  # 모델 타입: "solar" (GPU) 또는 "kogpt2" (CPU)

class PoemResponse(BaseModel):
    keywords: List[str]
    emotion: str
    emotion_confidence: float
    poem: str
    success: bool
    message: Optional[str] = None

@app.get("/health")
def health():
    from app.services.poem_config import MODEL_TYPE, GEN_MODEL_ID
    from app.services.poem_model_loader import _is_gpu, _device_info
    
    device_info = _device_info()
    is_gpu = _is_gpu()
    
    model_display = f"{MODEL_TYPE.upper()}" + (f" (GPU: {device_info})" if is_gpu else " (CPU)")
    
    return {
        "ok": True,
        "service": "poem",
        "model_type": MODEL_TYPE,
        "model_id": GEN_MODEL_ID,
        "device": device_info,
        "has_gpu": is_gpu,
        "model": model_display
    }

@app.post("/api/poem/generate", response_model=PoemResponse)
async def generate_poem_from_text(request: PoemRequest):
    """
    사용자의 일상글을 받아 키워드, 감정을 추출하고 시를 생성합니다.
    - 키워드: TF-IDF
    - 감정: XNLI 제로샷 (긍정/중립/부정 → 분위기 매핑)
    - 시: SOLAR-10.7B-Instruct (4bit, chat 템플릿)
    """
    t0 = time.time()
    print("\n" + "="*80)
    print("[API] /api/poem/generate 진입")

    # 요청 검증
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
    text = request.text.strip()
    print(f"[API] 입력 길이: {len(text)}자")

    # 1) 키워드 추출 (시 생성과 독립적으로 진행)
    print("[API] 1단계: 키워드 추출 시작...")
    keywords = extract_keywords(text, max_keywords=10)  # 더 많은 키워드 추출
    print(f"[API] ✓ 키워드 추출 완료: {keywords}")
    print("=" * 60)
    print("📝 추출된 키워드:", keywords)
    print("=" * 60)

    # 2) 감정 분류 (시 생성과 독립적으로 진행, 사용자가 분위기를 지정하지 않은 경우에만)
    print("[API] 2단계: 감정 분류 시작...")
    emo = classify_emotion(text)
    emotion = emo.get("emotion", "중립")
    default_mood = emo.get("mood", "담담한")
    confidence = float(emo.get("confidence", 0.0))
    
    # 사용자가 지정한 분위기가 있으면 사용, 없으면 자동 분석 결과 사용
    mood = request.mood if request.mood else default_mood
    lines = request.lines if request.lines else 4
    
    print(f"[API] ✓ 감정 분류 완료: 감정={emotion}, 분위기={mood}, 신뢰도={confidence:.3f}")
    print("=" * 60)
    print(f"💭 감정 분석 결과:")
    print(f"   - 감정: {emotion}")
    print(f"   - 분위기: {mood} (사용자 지정: {request.mood is not None})")
    print(f"   - 신뢰도: {confidence:.3f}")
    print(f"   - 줄 수: {lines}")
    if request.required_keywords:
        print(f"   - 필수 키워드: {request.required_keywords}")
    if request.banned_words:
        print(f"   - 금칙어: {request.banned_words}")
    if request.use_rhyme:
        print(f"   - 운율 사용: 예")
    if request.acrostic:
        print(f"   - 아크로스틱: {request.acrostic}")
    print("=" * 60)

    # 필수 키워드가 있으면 키워드 리스트에 추가
    final_keywords = keywords.copy()
    if request.required_keywords:
        for kw in request.required_keywords:
            if kw not in final_keywords:
                final_keywords.insert(0, kw)  # 필수 키워드를 앞에 추가

    # 3) 시 생성 (스레드 실행 + 타임아웃)
    print("[API] 3단계: 시 생성 시작...", flush=True)
    loop = asyncio.get_event_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 속도 최적화: max_new_tokens를 줄여서 생성 시간 단축 (80토큰으로 설정)
            # 원본 텍스트도 전달하여 맥락 반영
            print("[API] 시 생성 함수 호출 중... (속도 최적화: 80토큰)", flush=True)
            poem = await asyncio.wait_for(
                loop.run_in_executor(
                    executor, 
                    generate_poem_from_keywords, 
                    final_keywords, 
                    mood, 
                    lines, 
                    80, 
                    text,
                    request.banned_words,
                    request.use_rhyme,
                    request.acrostic,
                    request.model_type  # 모델 타입 전달
                ),
                timeout=300.0  # 5분 타임아웃 (첫 요청 시 모델 로딩 + 생성 + 번역 시간 포함)
            )
        print(f"[API] ✓ 시 생성 완료 (길이 {len(poem)}자)", flush=True)
    except asyncio.TimeoutError:
        print("[API] ❌ 타임아웃(>300s)", flush=True)
        raise HTTPException(status_code=504, detail="시 생성 시간이 초과되었습니다 (5분). 첫 요청은 모델 로딩으로 더 오래 걸릴 수 있습니다. 잠시 후 다시 시도해 주세요.")
    except Exception as e:
        error_type = type(e).__name__
        msg = str(e) or "시 생성 중 오류가 발생했습니다."
        print(f"[API] ❌ 생성 예외: {error_type}: {msg}")
        import traceback
        print("[API] 전체 트레이스백:")
        traceback.print_exc()
        
        # 더 구체적인 에러 메시지 제공
        if "메모리" in msg or "memory" in msg.lower() or "cuda" in msg.lower():
            detail_msg = f"GPU 메모리 부족 또는 CUDA 오류입니다. {msg[:200]}"
        elif "생성하지 않았습니다" in msg or "비어있습니다" in msg:
            detail_msg = f"모델이 텍스트를 생성하지 못했습니다. {msg[:200]}"
        else:
            detail_msg = f"시 생성 중 오류가 발생했습니다: {msg[:200]}"
        
        raise HTTPException(status_code=500, detail=detail_msg)

    # 4) 검증(아주 관대)
    poem_clean = (poem or "").strip()
    if not poem_clean:
        print("[API] ❌ 최종 결과 빈 문자열")
        raise HTTPException(status_code=500, detail="시 생성에 실패했습니다. 생성된 내용이 없습니다.")

    # 한글 문자가 3자 이상이면 통과
    korean_chars = sum(1 for c in poem_clean if ord('가') <= ord(c) <= ord('힣'))
    print(f"[API] 최종 검증: 길이={len(poem_clean)}자, 한글문자={korean_chars}자")
    if korean_chars < 3 and len(poem_clean) < 3:
        raise HTTPException(status_code=500, detail="시 생성에 실패했습니다. 생성된 내용이 너무 짧습니다.")

    print(f"[API] 전체 처리 시간: {time.time() - t0:.2f}s")
    print("="*80)

    return PoemResponse(
        keywords=keywords,
        emotion=emotion,
        emotion_confidence=confidence,
        poem=poem_clean,
        success=True,
        message="시가 성공적으로 생성되었습니다.",
    )