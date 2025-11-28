# -*- coding: utf-8 -*-
from typing import List, Optional, Dict
import time
import asyncio
import concurrent.futures
from pathlib import Path
import os
import sys

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

# 서비스 모듈 import
from app.services.keyword_extractor import extract_keywords
from app.services.emotion_classifier import classify_emotion
from app.services.poem_generator import generate_poem_from_keywords
from app.services.poem_model_loader import _load_poem_model

# 학습된 모델 사용을 위한 import (koGPT2 + LoRA 등)
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))
try:
    from use_trained_model import load_trained_model, generate_poem_from_prose
    HAS_TRAINED_MODEL = True
except ImportError:
    HAS_TRAINED_MODEL = False
    print("⚠️ 학습된 모델 모듈을 로드할 수 없습니다.")

# FastAPI 앱 생성
app = FastAPI(
    title="Poem Generation API",
    description="SOLAR 모델을 사용한 시 생성 API (Colab GPU 지원)",
    version="1.0.0",
)

# CORS 설정
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",  # ngrok 사용 시 모든 origin 허용 (개발 환경)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],  # "*"가 있으면 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],          # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],          # 모든 헤더 허용
    expose_headers=["*"],
    max_age=3600,                 # preflight 캐시 시간
)


# OPTIONS 요청 명시적 처리 (CORS preflight)
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """CORS preflight 요청 처리"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        },
    )


@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 모델을 미리 로드합니다.
    - 코랩에서는 첫 요청 시 로딩(다운로드 시간이 길기 때문)
    - 로컬/서버 환경에서는 사전 로딩으로 첫 요청 지연 최소화
    """
    is_colab = os.path.exists("/content")

    if is_colab:
        print("\n" + "=" * 80)
        print("🌐 코랩 환경 감지: 모델 사전 로딩 건너뜀")
        print("=" * 80)
        print("💡 첫 요청 시 자동으로 모델이 로드됩니다.")
        print("   (모델 다운로드에 5-10분이 걸릴 수 있습니다)")
        print("=" * 80 + "\n")
        return

    print("\n" + "=" * 80)
    print("🚀 서버 시작 중: 모델 사전 로딩 시작...")
    print("=" * 80)

    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, _load_poem_model)
        print("=" * 80)
        print("✅ 모델 사전 로딩 완료! 첫 요청부터 빠르게 응답할 수 있습니다.")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"⚠️ 모델 사전 로딩 실패: {e}")
        print("   (첫 요청 시 자동으로 로드됩니다.)\n")
        import traceback
        traceback.print_exc()

# ============================
# Pydantic 모델 정의
# ============================

class PoemRequest(BaseModel):
    text: str
    lines: Optional[int] = None                 # 줄 수 (행)
    mood: Optional[str] = None                  # 분위기 (잔잔/담담/쓸쓸)
    required_keywords: Optional[List[str]] = None  # 필수 키워드
    banned_words: Optional[List[str]] = None       # 금칙어
    use_rhyme: Optional[bool] = False           # 운율 사용 여부
    acrostic: Optional[str] = None              # 아크로스틱
    model_type: Optional[str] = None            # "solar" 또는 "kogpt2"
    use_trained_model: Optional[bool] = False   # 학습된 모델 사용 여부
    trained_model_path: Optional[str] = None    # 학습된 모델 경로
    use_gemini_improvement: Optional[bool] = True  # Gemini로 시 개선 여부


class PoemResponse(BaseModel):
    keywords: List[str]
    emotion: str
    emotion_confidence: float
    poem: str
    success: bool
    message: Optional[str] = None


class EmotionAnalysisRequest(BaseModel):
    poems: List[Dict]  # 시 목록 (emotion, createdAt 등 포함)


class EmotionAnalysisResponse(BaseModel):
    story: str
    summary: str
    emoji: str
    message: str
    success: bool


# ============================
# 기본 엔드포인트들
# ============================

@app.get("/")
def root():
    """루트 경로 - API 정보 반환"""
    return {
        "message": "Poem Generation API",
        "service": "siot-OSS",
        "endpoints": {
            "health": "/health",
            "generate_poem": "/api/poem/generate",
            "analyze_emotion": "/api/emotion/analyze-cute",
        },
        "docs": "/docs",
    }


@app.get("/favicon.ico")
def favicon():
    """favicon 요청 처리 (404 방지)"""
    from fastapi.responses import Response
    return Response(status_code=204)  # No Content


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
        "model": model_display,
    }


# ============================
# 시 생성 엔드포인트
# ============================

@app.post("/api/poem/generate", response_model=PoemResponse)
async def generate_poem_from_text(request: PoemRequest):
    """
    사용자의 일상글을 받아:
      1) 키워드 추출 (TF-IDF)
      2) 감정 분류 (XNLI 제로샷 기반)
      3) 시 생성 (SOLAR / koGPT2 / 학습 모델)
      4) (필요 시) 번역·개선
    을 수행한 뒤 시를 반환합니다.
    """
    t0 = time.time()
    print("\n" + "=" * 80)
    print("[API] /api/poem/generate 진입")

    # 0) 요청 검증
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
    text = request.text.strip()
    print(f"[API] 입력 길이: {len(text)}자")

    # 1) 키워드 추출
    print("[API] 1단계: 키워드 추출 시작...")
    keywords = extract_keywords(text, max_keywords=10)
    print(f"[API] ✓ 키워드 추출 완료: {keywords}")
    print("=" * 60)
    print("📝 추출된 키워드:", keywords)
    print("=" * 60)

    # 2) 감정 분류
    print("[API] 2단계: 감정 분류 시작...")
    emo = classify_emotion(text)
    emotion = emo.get("emotion", "중립")
    default_mood = emo.get("mood", "담담한")
    confidence = float(emo.get("confidence", 0.0))

    mood = request.mood if request.mood else default_mood
    lines = request.lines if request.lines else 4

    print(f"[API] ✓ 감정 분류 완료: 감정={emotion}, 분위기={mood}, 신뢰도={confidence:.3f}")
    print("=" * 60)
    print("💭 감정 분석 결과:")
    print(f"   - 감정: {emotion}")
    print(f"   - 분위기: {mood} (사용자 지정: {request.mood is not None})")
    print(f"   - 신뢰도: {confidence:.3f}")
    print(f"   - 줄 수: {lines}")
    if request.required_keywords:
        print(f"   - 필수 키워드: {request.required_keywords}")
    if request.banned_words:
        print(f"   - 금칙어: {request.banned_words}")
    if request.use_rhyme:
        print("   - 운율 사용: 예")
    if request.acrostic:
        print(f"   - 아크로스틱: {request.acrostic}")
    print("=" * 60)

    # 필수 키워드가 있으면 키워드 리스트에 추가
    final_keywords = keywords.copy()
    if request.required_keywords:
        for kw in request.required_keywords:
            if kw not in final_keywords:
                final_keywords.insert(0, kw)

    # 3) 시 생성 (기본 모델 or 학습된 모델)
    print("[API] 3단계: 시 생성 시작...", flush=True)

    use_trained = request.use_trained_model and HAS_TRAINED_MODEL
    trained_model_path = request.trained_model_path

    # 학습된 모델 경로 자동 탐색
    if use_trained and not trained_model_path:
        trained_models_dir = backend_path / "trained_models"
        if trained_models_dir.exists():
            model_folders = [
                f
                for f in trained_models_dir.iterdir()
                if f.is_dir()
                and "20251109_08" in f.name
                and "kogpt2" in f.name.lower()
            ]
            if model_folders:
                trained_model_path = str(
                    sorted(model_folders, key=lambda x: x.name, reverse=True)[0]
                )
                print(f"[API] 자동으로 학습된 모델 찾음: {trained_model_path}", flush=True)
            else:
                print("[API] ⚠️ 학습된 모델을 찾을 수 없습니다. 기본 모델 사용", flush=True)
                use_trained = False
        else:
            print("[API] ⚠️ trained_models 폴더가 없습니다. 기본 모델 사용", flush=True)
            use_trained = False

    loop = asyncio.get_event_loop()

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if use_trained and trained_model_path:
                # =============================
                # 3-A) 학습된 모델 사용 (koGPT2 + LoRA 등)
                # =============================
                print("[API] 학습된 모델 사용 모드", flush=True)

                def generate_with_trained_model():
                    # 캐시된 모델 사용 (없으면 로드)
                    if (
                        not hasattr(generate_with_trained_model, "_tokenizer")
                        or not hasattr(generate_with_trained_model, "_model")
                        or not hasattr(generate_with_trained_model, "_device")
                        or getattr(
                            generate_with_trained_model, "_model_path", None
                        )
                        != trained_model_path
                    ):
                        print("[API] 학습된 모델 로딩 중...", flush=True)
                        tokenizer, model, device = load_trained_model(trained_model_path)
                        generate_with_trained_model._tokenizer = tokenizer
                        generate_with_trained_model._model = model
                        generate_with_trained_model._device = device
                        generate_with_trained_model._model_path = trained_model_path
                        print("[API] 학습된 모델 로딩 완료", flush=True)

                    # 산문 직접 입력 → 시 생성
                    raw_poem = generate_poem_from_prose(
                        text,
                        generate_with_trained_model._tokenizer,
                        generate_with_trained_model._model,
                        generate_with_trained_model._device,
                        max_new_tokens=150,
                    )

                    # Gemini 개선 옵션 체크
                    has_prompt_options = (
                        (request.lines is not None and request.lines != 4)
                        or (request.mood is not None and request.mood.strip() != "")
                        or (
                            request.required_keywords is not None
                            and len(request.required_keywords) > 0
                        )
                        or (
                            request.banned_words is not None
                            and len(request.banned_words) > 0
                        )
                        or (request.use_rhyme is True)
                        or (
                            request.acrostic is not None
                            and request.acrostic.strip() != ""
                        )
                    )

                    print(
                        "[API] 프롬프트 옵션 체크:",
                        f"lines={request.lines}, mood={request.mood}, "
                        f"required_keywords={request.required_keywords}, banned_words={request.banned_words}, "
                        f"use_rhyme={request.use_rhyme}, acrostic={request.acrostic}",
                        flush=True,
                    )
                    print(
                        f"[API] has_prompt_options={has_prompt_options}, "
                        f"use_gemini_improvement={request.use_gemini_improvement}",
                        flush=True,
                    )

                    from app.main import improve_poem_with_gemini  # 순환 import 방지용 지연 import

                    if has_prompt_options and request.use_gemini_improvement is not False:
                        try:
                            print("[API] Gemini로 시 개선 시작", flush=True)
                            improved_poem = improve_poem_with_gemini(
                                raw_poem,
                                text,
                                lines=request.lines,
                                mood=request.mood,
                                required_keywords=request.required_keywords,
                                banned_words=request.banned_words,
                                use_rhyme=request.use_rhyme,
                                acrostic=request.acrostic,
                            )
                            if improved_poem and improved_poem != raw_poem:
                                print(
                                    f"[API] ✓ Gemini 개선 완료: 원본 {len(raw_poem)}자 → 개선 {len(improved_poem)}자",
                                    flush=True,
                                )
                                return improved_poem
                            else:
                                print(
                                    "[API] ⚠️ Gemini 개선 결과가 원본과 동일하거나 비어있음. 원본 반환",
                                    flush=True,
                                )
                                return raw_poem
                        except Exception as e:
                            print(f"[API] ❌ Gemini 시 개선 실패, 원본 사용: {e}", flush=True)
                            import traceback

                            traceback.print_exc()
                            return raw_poem
                    else:
                        if has_prompt_options:
                            print(
                                "[API] 프롬프트 옵션은 있으나 Gemini 개선 비활성화",
                                flush=True,
                            )
                        else:
                            print(
                                "[API] 프롬프트 옵션 없음 → Gemini 개선 생략 (원본 시 반환)",
                                flush=True,
                            )
                        return raw_poem

                poem = await asyncio.wait_for(
                    loop.run_in_executor(executor, generate_with_trained_model),
                    timeout=300.0,
                )
            else:
                # =============================
                # 3-B) 기본 모델 사용 (SOLAR / koGPT2)
                #  → generate_poem_from_keywords 안에서
                #     프롬프트 구성 + SOLAR 호출 + 후처리 + (필요 시 번역)까지 수행
                # =============================
                print(
                    "[API] 기본 모델로 시 생성 중... (max_new_tokens=80)",
                    flush=True,
                )
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
                        request.model_type,  # "solar" (Colab) 또는 "kogpt2"
                    ),
                    timeout=300.0,
                )

        print(f"[API] ✓ 시 생성 완료 (길이 {len(poem)}자)", flush=True)
    except asyncio.TimeoutError:
        print("[API] ❌ 타임아웃(>300s)", flush=True)
        raise HTTPException(
            status_code=504,
            detail=(
                "시 생성 시간이 초과되었습니다 (5분). "
                "첫 요청은 모델 로딩으로 더 오래 걸릴 수 있습니다. 잠시 후 다시 시도해 주세요."
            ),
        )
    except Exception as e:
        error_type = type(e).__name__
        msg = str(e) or "시 생성 중 오류가 발생했습니다."
        print(f"[API] ❌ 생성 예외: {error_type}: {msg}")
        import traceback

        print("[API] 전체 트레이스백:")
        traceback.print_exc()

        if "메모리" in msg or "memory" in msg.lower() or "cuda" in msg.lower():
            detail_msg = f"GPU 메모리 부족 또는 CUDA 오류입니다. {msg[:200]}"
        elif "생성하지 않았습니다" in msg or "비어있습니다" in msg:
            detail_msg = f"모델이 텍스트를 생성하지 못했습니다. {msg[:200]}"
        else:
            detail_msg = f"시 생성 중 오류가 발생했습니다: {msg[:200]}"

        raise HTTPException(status_code=500, detail=detail_msg)

    # 4) 최종 검증 (아주 관대)
    poem_clean = (poem or "").strip()
    if not poem_clean:
        print("[API] ❌ 최종 결과 빈 문자열")
        raise HTTPException(
            status_code=500,
            detail="시 생성에 실패했습니다. 생성된 내용이 없습니다.",
        )

    korean_chars = sum(1 for c in poem_clean if ord("가") <= ord(c) <= ord("힣"))
    print(f"[API] 최종 검증: 길이={len(poem_clean)}자, 한글문자={korean_chars}자")
    if korean_chars < 3 and len(poem_clean) < 3:
        raise HTTPException(
            status_code=500,
            detail="시 생성에 실패했습니다. 생성된 내용이 너무 짧습니다.",
        )

    print(f"[API] 전체 처리 시간: {time.time() - t0:.2f}s")
    print("=" * 80)

    return PoemResponse(
        keywords=keywords,
        emotion=emotion,
        emotion_confidence=confidence,
        poem=poem_clean,
        success=True,
        message="시가 성공적으로 생성되었습니다.",
    )


# ============================
# Gemini 기반 시 개선 / 감정 스토리
# ============================

def improve_poem_with_gemini(
    raw_poem: str,
    original_prose: str = "",
    lines: Optional[int] = None,
    mood: Optional[str] = None,
    required_keywords: Optional[List[str]] = None,
    banned_words: Optional[List[str]] = None,
    use_rhyme: Optional[bool] = False,
    acrostic: Optional[str] = None,
) -> str:
    """
    Gemini API를 사용하여 koGPT2로 생성한 시를 개선합니다.
    (SOLAR + 번역 파이프라인과는 독립적인 옵션 기능)
    """
    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Gemini] ⚠️ GEMINI_API_KEY가 설정되지 않았습니다. 원본 시 반환")
            return raw_poem

        genai.configure(api_key=api_key)

        try:
            print("[Gemini] 사용 가능한 모델 목록 확인 중...", flush=True)
            available_models = []
            for model_info in genai.list_models():
                if "generateContent" in model_info.supported_generation_methods:
                    available_models.append(model_info.name)

            print(f"[Gemini] 사용 가능한 모델 {len(available_models)}개 발견", flush=True)

            preferred_models = [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-lite-preview-06-17",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "models/gemini-pro",
            ]

            model = None
            selected_model_name = None
            for model_name in preferred_models:
                if model_name in available_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        selected_model_name = model_name
                        print(
                            f"[Gemini] 모델 로드 성공: {model_name}",
                            flush=True,
                        )
                        break
                    except Exception as e:
                        print(
                            f"[Gemini] 모델 {model_name} 로드 실패: {e}",
                            flush=True,
                        )
                        continue

            if model is None and available_models:
                selected_model_name = available_models[0]
                model = genai.GenerativeModel(selected_model_name)
                print(
                    f"[Gemini] 대체 모델 사용: {selected_model_name}",
                    flush=True,
                )
            elif model is None:
                print("[Gemini] 사용 가능한 모델을 찾을 수 없습니다. 원본 시 반환", flush=True)
                return raw_poem
        except Exception as e:
            print(f"[Gemini] 모델 로드 실패: {e}, 원본 시 반환", flush=True)
            import traceback

            traceback.print_exc()
            return raw_poem

        # 옵션 텍스트 구성
        option_parts = []
        if lines is not None and lines != 4:
            option_parts.append(f"- 정확히 {lines}줄로 작성해주세요.")
        if mood and mood.strip():
            option_parts.append(
                f"- 분위기: {mood.strip()} (이 분위기를 시에 반영해주세요)"
            )
        if required_keywords:
            kw_str = ", ".join(required_keywords)
            option_parts.append(
                f"- 필수 키워드: {kw_str} (반드시 이 키워드들을 시에 포함해주세요)"
            )
        if banned_words:
            banned_str = ", ".join(banned_words)
            option_parts.append(
                f"- 금지 단어: {banned_str} (절대 사용하지 마세요)"
            )
        if use_rhyme:
            option_parts.append(
                "- 운율을 사용해주세요 (비슷한 발음이나 반복되는 소리로 리듬감을 주세요)"
            )
        if acrostic and acrostic.strip():
            acrostic_chars = " ".join(list(acrostic.strip()))
            option_parts.append(
                f"- 두문자 시: 각 줄의 첫 글자가 '{acrostic_chars}' 순서대로 오도록 해주세요 "
                f"(총 {len(acrostic.strip())}줄)"
            )

        options_text = "\n".join(option_parts) if option_parts else ""

        if options_text:
            print(
                f"[Gemini] 프롬프트 옵션 적용: {len(option_parts)}개 옵션",
                flush=True,
            )
            for i, opt in enumerate(option_parts, 1):
                print(f"[Gemini]   {i}. {opt}", flush=True)
        else:
            print("[Gemini] 프롬프트 옵션 없음 (기본 개선만 수행)", flush=True)

        prompt = f"""다음은 AI가 생성한 한국어 시입니다. 이 시를 개선해주세요.

원본 산문 (참고용):
{original_prose[:200] if original_prose else "없음"}

생성된 시:
{raw_poem}

다음 작업을 수행해주세요:
1. 불필요한 텍스트 제거 (뉴스 스타일 문장, 설명문 등)
2. "시:", "산문:" 같은 프롬프트 패턴 제거
3. 적절한 줄바꿈 유지 (문장 끝과 쉼표 뒤)
4. 시적 표현 개선 (자연스럽고 아름다운 표현으로 다듬기)
{f"5. 다음 요구사항을 반드시 지켜주세요:\n{options_text}" if options_text else ""}

중요: 시의 주제와 핵심 의미는 유지하되, 표현을 더 시답게 개선해주세요.

개선된 시만 출력해주세요. 설명이나 추가 텍스트 없이 시만 출력하세요.
"""

        print(
            f"[Gemini] 프롬프트 전송 중... (원본 시 길이: {len(raw_poem)}자)",
            flush=True,
        )
        response = model.generate_content(prompt)

        improved_poem = ""
        if hasattr(response, "text"):
            improved_poem = response.text.strip()
        elif getattr(response, "candidates", None):
            c0 = response.candidates[0]
            if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
                improved_poem = "".join(
                    [p.text for p in c0.content.parts if hasattr(p, "text")]
                ).strip()

        print(f"[Gemini] 응답 받음: {len(improved_poem)}자", flush=True)
        if improved_poem:
            print(
                f"[Gemini] 응답 미리보기 (처음 100자): {improved_poem[:100]}",
                flush=True,
            )

        if improved_poem and len(improved_poem) > 10:
            print(
                f"[Gemini] 시 개선 완료: {len(raw_poem)}자 → {len(improved_poem)}자",
                flush=True,
            )
            return improved_poem
        else:
            print(
                f"[Gemini] 개선 결과가 비어있거나 너무 짧음 ({len(improved_poem)}자). 원본 시 반환",
                flush=True,
            )
            return raw_poem

    except Exception as e:
        print(f"[Gemini] 시 개선 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return raw_poem


def analyze_emotions_cutely(poems: List[Dict]) -> Dict[str, str]:
    """
    Gemini API를 사용하여 감정 데이터를 귀여운 스토리로 변환합니다.
    """
    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Gemini] ⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
            return {
                "story": "감정 분석을 위해 Gemini API 키를 설정해주세요.",
                "summary": "API 키가 필요합니다.",
                "emoji": "🔑",
                "message": "설정에서 API 키를 추가해주세요.",
                "success": False,
            }

        genai.configure(api_key=api_key)

        try:
            available_models = []
            for model_info in genai.list_models():
                if "generateContent" in model_info.supported_generation_methods:
                    available_models.append(model_info.name)

            print(f"[Gemini] 사용 가능한 모델: {available_models[:5]}")

            preferred_models = [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-lite-preview-06-17",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "models/gemini-pro",
            ]

            model = None
            for model_name in preferred_models:
                if model_name in available_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        print(f"[Gemini] 모델 로드 성공: {model_name}")
                        break
                    except Exception as e:
                        print(f"[Gemini] 모델 {model_name} 시도 실패: {e}")
                        continue

            if model is None:
                if available_models:
                    first_model = available_models[0]
                    print(f"[Gemini] 사용 가능한 첫 번째 모델 사용: {first_model}")
                    model = genai.GenerativeModel(first_model)
                else:
                    raise Exception("사용 가능한 모델을 찾을 수 없습니다.")

        except Exception as e:
            print(f"[Gemini] 모델 목록 확인 실패, 기본 모델 시도: {e}")
            try:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
            except Exception:
                try:
                    model = genai.GenerativeModel("models/gemini-1.5-flash")
                except Exception:
                    raise Exception(f"모델을 로드할 수 없습니다. 오류: {e}")

        # 감정 데이터 집계
        emotion_data: Dict[str, Dict[str, int]] = {}
        for poem in poems:
            if not poem.get("emotion") or not poem.get("createdAt"):
                continue
            emotion = poem["emotion"]
            date = poem["createdAt"][:10]
            if date not in emotion_data:
                emotion_data[date] = {}
            emotion_data[date][emotion] = emotion_data[date].get(emotion, 0) + 1

        emotion_counts: Dict[str, int] = {}
        for poem in poems:
            if poem.get("emotion"):
                emotion = poem["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        prompt = f"""당신은 감정 분석 전문가입니다. 다음 감정 데이터를 자연스럽고 따뜻한 톤으로 분석해주세요.

감정 데이터:
{emotion_data}

감정별 개수:
{emotion_counts}

다음 형식으로 응답해주세요:

1. 감정 추이 스토리 (100-150자): 날짜별 감정 변화를 자연스럽게 설명해주세요.
2. 감정 요약 (50-80자): 전체적인 감정 패턴을 간결하게 요약해주세요.
3. 대표 이모지: 가장 많이 나타난 감정에 맞는 이모지 하나
4. 따뜻한 메시지 (30-50자): 사용자에게 전하는 자연스러운 메시지

주의사항:
- 어린아이에게 하는 말투나 과도하게 귀여운 표현은 피해주세요
- 자연스럽고 성숙한 톤을 유지해주세요
- 따뜻하지만 전문적인 느낌을 주세요

형식:
스토리: [여기에 스토리]
요약: [여기에 요약]
이모지: [여기에 이모지]
메시지: [여기에 메시지]
"""

        response = model.generate_content(prompt)
        result_text = response.text

        story = ""
        summary = ""
        emoji = "💭"
        message = ""

        for line in result_text.split("\n"):
            line = line.strip()
            if line.startswith("스토리:"):
                story = line.replace("스토리:", "").strip()
            elif line.startswith("요약:"):
                summary = line.replace("요약:", "").strip()
            elif line.startswith("이모지:"):
                emoji = line.replace("이모지:", "").strip()
            elif line.startswith("메시지:"):
                message = line.replace("메시지:", "").strip()

        if not story:
            story = result_text[:150] if result_text else "감정 데이터를 분석했어요!"
        if not summary:
            summary = "감정 변화를 확인했어요."
        if not message:
            message = "오늘도 수고하셨어요!"

        return {
            "story": story,
            "summary": summary,
            "emoji": emoji,
            "message": message,
            "success": True,
        }

    except Exception as e:
        print(f"[Gemini] ❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return {
            "story": "감정 분석 중 오류가 발생했어요. 다시 시도해주세요.",
            "summary": "오류 발생",
            "emoji": "😢",
            "message": "잠시 후 다시 시도해주세요.",
            "success": False,
        }


@app.post("/api/emotion/analyze-cute", response_model=EmotionAnalysisResponse)
async def analyze_emotions_cutely_endpoint(request: EmotionAnalysisRequest):
    """
    감정 데이터를 받아서 Gemini API로 귀여운 스토리로 변환합니다.
    """
    print("\n" + "=" * 80)
    print("[API] /api/emotion/analyze-cute 진입")
    print(f"[API] 시 개수: {len(request.poems)}개")

    if not request.poems:
        raise HTTPException(status_code=400, detail="시 데이터가 없습니다.")

    result = analyze_emotions_cutely(request.poems)

    print("[API] ✓ 감정 분석 완료")
    print("=" * 80)

    return EmotionAnalysisResponse(**result)
