# -*- coding: utf-8 -*-
"""
시 생성 메인 함수 (SOLAR / koGPT2 공용)

- Colab에서는 보통 MODEL_TYPE = "solar" 로 사용
- 프롬프트 구성: app.services.poem_prompt_builder
- 모델/토크나이저 로딩: app.services.poem_model_loader
- 후처리(줄 정리 등): app.services.poem_text_processor
"""

import time
import traceback
from typing import List, Optional, Dict, Any

import torch

from app.services.poem_config import (
    USE_ML_MODEL,
    MODEL_TYPE,
    DEFAULT_LINES,
    DEFAULT_MAX_NEW_TOKENS_GPU,
    DEFAULT_MAX_NEW_TOKENS_CPU,
)
from app.services.poem_model_loader import (
    _load_poem_model,
    _is_gpu,
    _device_info,
)
from app.services.poem_prompt_builder import (
    _build_messages,
    _build_messages_kogpt2,
)
from app.services.poem_text_processor import (
    _postprocess_poem,
)

# =====================================================================
# Colab BrokenPipeError 방지용 안전한 print
# =====================================================================
import builtins
import errno

_original_print = builtins.print


def safe_print(*args, **kwargs):
    """
    Colab에서 stdout 파이프가 끊어진 뒤에도 BrokenPipeError가 나지 않도록 하는 print.

    - EPIPE(Broken pipe) 에러만 조용히 무시
    - 나머지 에러는 그대로 발생시켜서 디버깅에 쓰기
    """
    try:
        _original_print(*args, **kwargs)
    except OSError as e:
        if e.errno == errno.EPIPE:
            # 출력 파이프가 끊어졌을 때는 그냥 무시
            return
        raise


# 이 모듈 안에서는 print -> safe_print 사용
print = safe_print  # noqa: E305


# =====================================================================
# 내부 디버깅용 함수
# =====================================================================

def _log_header(title: str):
    """블록 단위 로그 헤더"""
    try:
        print("\n" + "=" * 80, flush=True)
        print(f"[poem_generator] {title}", flush=True)
        print("=" * 80, flush=True)
    except Exception:
        # 여기서까지 에러 나면 그냥 조용히 무시 (BrokenPipe 등)
        pass


def _debug_tensor(name: str, tensor: torch.Tensor):
    """텐서 모양/타입 출력 (필요할 때만 참조용)"""
    try:
        print(
            f"[debug] {name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}",
            flush=True,
        )
    except Exception:
        print(f"[debug] {name}: <unavailable>", flush=True)


# =====================================================================
# 핵심: 키워드/분위기 기반 시 생성 함수 (엔진)
# =====================================================================

@torch.no_grad()
def generate_poem_from_keywords(
    keywords: List[str],
    mood: str = "잔잔한",
    lines: int = DEFAULT_LINES,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_GPU,
    original_text: str = "",
    banned_words: Optional[List[str]] = None,
    use_rhyme: bool = False,
    acrostic: Optional[str] = None,
    model_type: Optional[str] = None,  # "solar" 또는 "kogpt2"
) -> str:
    """
    실제로 SOLAR/koGPT2에 프롬프트를 넣어서 시를 생성하는 엔진 함수.
    - keywords, mood, lines 등을 이미 알고 있다는 가정.
    """
    func_start = time.time()
    _log_header("시 생성 함수 진입 (단순 버전)")

    print(f"[args] keywords={keywords}", flush=True)
    print(f"[args] mood={mood}, lines={lines}, max_new_tokens={max_new_tokens}", flush=True)
    print(f"[env] MODEL_TYPE(default)={MODEL_TYPE}, device={_device_info()}, USE_ML_MODEL={USE_ML_MODEL}", flush=True)

    if not USE_ML_MODEL:
        raise RuntimeError("USE_ML_MODEL=False 상태입니다. (ML 모델 생성 비활성화)")

    from app.services.poem_config import MODEL_TYPE as DEFAULT_MODEL_TYPE
    actual_model_type = (model_type or DEFAULT_MODEL_TYPE).lower()
    if actual_model_type not in ["solar", "kogpt2"]:
        print(f"[warn] 잘못된 model_type '{actual_model_type}' → 기본값 '{DEFAULT_MODEL_TYPE}' 사용", flush=True)
        actual_model_type = DEFAULT_MODEL_TYPE

    # 1) 모델 로딩
    print(f"[step] 1. 모델 로딩 시작 (model_type={actual_model_type})", flush=True)
    tok, model = _load_poem_model(actual_model_type)
    print(f"[step] ✓ 모델 로딩 완료 (device={_device_info()})", flush=True)

    # 2) 프롬프트 & 토크나이즈
    print(f"[step] 2. 프롬프트 구성 및 토크나이즈", flush=True)
    t_enc = time.time()

    if not lines or lines <= 0:
        lines = DEFAULT_LINES

    if actual_model_type == "kogpt2":
        prompt_text = _build_messages_kogpt2(
            keywords=keywords,
            mood=mood,
            lines=lines,
            original_text=original_text,
            banned_words=banned_words,
            use_rhyme=use_rhyme,
            acrostic=acrostic,
        )
        print(f"[step] koGPT2 프롬프트 (앞 200자): {repr(prompt_text[:200])}", flush=True)
        enc_ids = tok.encode(prompt_text, return_tensors="pt")
    else:
        messages = _build_messages(
            keywords=keywords,
            mood=mood,
            lines=lines,
            original_text=original_text,
            banned_words=banned_words,
            use_rhyme=use_rhyme,
            acrostic=acrostic,
        )
        enc_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    # Tensor 형태 정리
    if isinstance(enc_ids, torch.Tensor):
        if enc_ids.dim() == 1:
            enc_ids = enc_ids.unsqueeze(0)
    else:
        enc_ids = torch.tensor(enc_ids, dtype=torch.long).unsqueeze(0)

    enc_ids = enc_ids.to(dtype=torch.long)
    print(f"[step] ✓ 토크나이즈 완료 ({time.time() - t_enc:.2f}s)", flush=True)
    _debug_tensor("input_ids(raw)", enc_ids)

    # 2-2) device 이동
    model_device = next(model.parameters()).device
    input_ids = enc_ids.to(model_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

    # 3) 생성 파라미터 (단순)
    print("[step] 3. 생성 파라미터 설정 (단순)", flush=True)
    is_gpu = _is_gpu()
    safe_max_new = max_new_tokens
    min_required = max(30, lines * 8)
    if safe_max_new < min_required:
        safe_max_new = min_required

    if is_gpu:
        safe_max_new = min(safe_max_new, 80)
    else:
        safe_max_new = min(safe_max_new, 40)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": safe_max_new,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    print(
        "[gen_kwargs(simple)]",
        {k: (v if not isinstance(v, torch.Tensor) else f"Tensor(shape={tuple(v.shape)})") for k, v in gen_kwargs.items()},
        flush=True,
    )

    # 4) model.generate()
    print("[step] 4. model.generate() 호출 (단순)", flush=True)
    t_gen = time.time()
    try:
        out = model.generate(**gen_kwargs)
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"model.generate() 단계에서 오류 발생: {type(e).__name__}: {str(e)[:200]}")

    print(f"[step] ✓ 생성 완료 ({time.time() - t_gen:.2f}s)", flush=True)

    # 5) 디코딩 (입력 이후 토큰만)
    print("[step] 5. 디코딩 및 (옵션) 후처리", flush=True)
    if out.dim() != 2 or out.shape[0] != 1:
        raise Exception(f"출력 텐서 형태가 이상합니다: {tuple(out.shape)}")

    input_len = input_ids.shape[1]
    output_len = out.shape[1]
    new_tokens = output_len - input_len
    if new_tokens <= 0:
        raise Exception(f"모델이 새 토큰을 생성하지 않았습니다. (input_len={input_len}, output_len={output_len})")

    generated_ids = out[0, input_len:]
    decoded = tok.decode(generated_ids, skip_special_tokens=True).strip()
    if not decoded:
        decoded = tok.decode(out[0], skip_special_tokens=True).strip()
    if not decoded:
        raise Exception("디코딩 결과가 비어 있습니다.")

    # 6) 후처리 오류를 무시하고 최소한 텍스트는 반환
    try:
        poem = _postprocess_poem(decoded, min_lines=lines, max_lines=lines * 3).strip()
        if not poem:
            raise ValueError("후처리 결과가 비었습니다.")
        result_text = poem
        print(f"[done] 최종 시 길이: {len(result_text)}자 (후처리 적용)", flush=True)
    except Exception as e:
        print(f"[warn] _postprocess_poem 예외 → 후처리 없이 원문 사용: {type(e).__name__}: {str(e)[:200]}", flush=True)
        result_text = decoded
        print(f"[done] 최종 시 길이: {len(result_text)}자 (후처리 없이)", flush=True)

    print(f"[done] 총 소요 시간: {time.time() - func_start:.2f}s", flush=True)
    return result_text


# =====================================================================
# 기존 API 호환용 (감정 → 분위기 매핑, 직접 키워드 주는 경우)
# =====================================================================

def generate_poem(keywords: List[str], emotion: str, max_length: int = 120) -> str:
    """
    감정 레이블을 받아 분위기로 매핑해서 generate_poem_from_keywords를 호출하는
    기존 호환용 래퍼 함수.
    """
    emotion_to_mood = {
        "기쁨": "잔잔한",
        "슬픔": "쓸쓸한",
        "중립": "담담한",
        "사랑": "잔잔한",
        "그리움": "쓸쓸한",
    }
    mood = emotion_to_mood.get(emotion, "담담한")

    max_new = min(
        max_length,
        DEFAULT_MAX_NEW_TOKENS_GPU if _is_gpu() else DEFAULT_MAX_NEW_TOKENS_CPU,
    )

    return generate_poem_from_keywords(
        keywords=keywords,
        mood=mood,
        lines=DEFAULT_LINES,
        max_new_tokens=max_new,
    )


# =====================================================================
# 새 진입 함수: 순수 텍스트만 받아서 전체 파이프라인 실행
# =====================================================================

def generate_poem_from_text(
    text: str,
    model_type: Optional[str] = None,
    lines: Optional[int] = None,
) -> Dict[str, Any]:
    """
    /api/poem/generate 에서 직접 호출할 래퍼.

    1) 입력 text에서 키워드 추출 (keyword_extractor)
    2) 감정/분위기 분석 (emotion_classifier)
    3) SOLAR/koGPT2로 시 생성 (generate_poem_from_keywords)
    4) translator로 비한국어가 있으면 한국어로 번역
    5) JSON 응답 형태로 결과 반환

    FastAPI 엔드포인트에서는 이 함수만 쓰면 됨.
    """
    from app.services.keyword_extractor import extract_keywords
    from app.services.emotion_classifier import classify_emotion
    from app.services.translator import translate_poem_with_retry

    text = (text or "").strip()
    if not text:
        return {
            "keywords": [],
            "emotion": "중립",
            "emotion_confidence": 0.0,
            "poem": "",
            "success": False,
            "message": "입력 텍스트가 비어 있습니다.",
        }

    # 1) 키워드 추출
    try:
        keywords = extract_keywords(text, max_keywords=8)
    except Exception as e:
        print(f"[generate_poem_from_text] 키워드 추출 실패: {e}", flush=True)
        keywords = []

    # 2) 감정 + 분위기 분석 (emotion_classifier는 dict 반환)
    try:
        emo_info = classify_emotion(text)
        emo_label = emo_info.get("emotion", "중립")
        mood = emo_info.get("mood", "담담한")
        emo_conf = float(emo_info.get("confidence", 0.0))
    except Exception as e:
        print(f"[generate_poem_from_text] 감정 분석 실패: {e}", flush=True)
        emo_label, mood, emo_conf = "중립", "담담한", 0.0

    # 3) max_new_tokens 결정
    max_new_tokens = (
        DEFAULT_MAX_NEW_TOKENS_GPU if _is_gpu() else DEFAULT_MAX_NEW_TOKENS_CPU
    )

    safe_lines = lines or DEFAULT_LINES

    # 4) 실제 시 생성 (키워드 + mood 기반 엔진 호출)
    try:
        poem_text = generate_poem_from_keywords(
            keywords=keywords,
            mood=mood,
            lines=safe_lines,
            max_new_tokens=max_new_tokens,
            original_text=text,
            model_type=model_type,  # None이면 poem_config.MODEL_TYPE 사용
        )
    except Exception as e:
        print(f"[generate_poem_from_text] 시 생성 실패: {e}", flush=True)
        return {
            "keywords": keywords,
            "emotion": emo_label,
            "emotion_confidence": emo_conf,
            "poem": "",
            "success": False,
            "message": f"시 생성 중 오류가 발생했습니다: {e}",
        }

    # 5) 번역기: 비한국어가 있으면 한국어로 강제 번역 시도
    try:
        poem_text_ko = translate_poem_with_retry(poem_text)
    except Exception as e:
        print(f"[generate_poem_from_text] 번역 실패, 원문 그대로 사용: {e}", flush=True)
        poem_text_ko = poem_text

    # 6) API 응답 포맷
    return {
        "keywords": keywords,
        "emotion": emo_label,
        "emotion_confidence": emo_conf,
        "poem": poem_text_ko,
        "success": True,
        "message": "시가 성공적으로 생성되었습니다.",
    }