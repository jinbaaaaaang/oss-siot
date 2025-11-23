# -*- coding: utf-8 -*-
"""
시 생성 관련 설정 및 상수 정의
"""

import os
from typing import Optional
import torch

# ======== 전역 설정 ========
USE_ML_MODEL: bool = True

# GPU 자동 감지 함수
def _is_gpu_available() -> bool:
    """GPU 사용 가능 여부 확인 (CUDA 또는 MPS)"""
    try:
        # CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            return True
        # MPS (Apple Silicon GPU)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True
        return False
    except Exception:
        return False

# 모델 타입 자동 결정
# 1. 환경 변수가 설정되어 있으면 우선 사용
# 2. 환경 변수가 없으면 GPU 여부에 따라 자동 선택
_env_model_type = os.getenv("POEM_MODEL_TYPE", "").lower()

if _env_model_type in ["solar", "kogpt2"]:
    # 환경 변수로 명시적으로 지정된 경우
    MODEL_TYPE: str = _env_model_type
    print(f"[poem_config] 환경 변수로 모델 타입 지정: {MODEL_TYPE}")
elif _env_model_type:
    # 잘못된 환경 변수 값
    print(f"[poem_config] ⚠️ 알 수 없는 모델 타입: {_env_model_type}, GPU 여부에 따라 자동 선택")
    MODEL_TYPE = "solar" if _is_gpu_available() else "kogpt2"
else:
    # 환경 변수가 없으면 GPU 여부에 따라 자동 선택
    if _is_gpu_available():
        MODEL_TYPE = "solar"
        print(f"[poem_config] ✅ GPU 감지됨 → SOLAR 모델 자동 선택 (고품질)")
    else:
        MODEL_TYPE = "kogpt2"
        print(f"[poem_config] ℹ️ GPU 없음 → koGPT2 모델 자동 선택 (CPU 친화적)")

# 모델 ID 결정
if MODEL_TYPE == "kogpt2":
    GEN_MODEL_ID: str = "skt/kogpt2-base-v2"  # koGPT2 (CPU 친화적, 약 124M 파라미터)
else:
    GEN_MODEL_ID: str = "upstage/SOLAR-10.7B-Instruct-v1.0"  # SOLAR (GPU 권장, 약 10.7B 파라미터)

print(f"[poem_config] 최종 선택된 모델: {MODEL_TYPE} ({GEN_MODEL_ID})")

# 생성 기본 하이퍼파라미터 (속도 최적화)
DEFAULT_LINES: int = 4
DEFAULT_MAX_NEW_TOKENS_GPU: int = 80    # 속도 향상을 위해 96 → 80으로 감소
DEFAULT_MAX_NEW_TOKENS_CPU: int = 24    # CPU는 아주 짧게만
DEFAULT_TEMPERATURE: float = 0.7        # 더 빠른 생성 (0.8 → 0.7)
DEFAULT_TOP_P: float = 0.85             # 더 빠른 생성 (0.9 → 0.85)
DEFAULT_TOP_K: int = 30                 # 더 빠른 생성 (40 → 30)
DEFAULT_NO_REPEAT_NGRAM: int = 2
DEFAULT_REPETITION_PENALTY: float = 1.05 # 반복 방지 완화로 속도 향상 (1.1 → 1.05)

