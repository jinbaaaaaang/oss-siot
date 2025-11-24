# -*- coding: utf-8 -*-
"""
시 생성 모델 로딩 관련 함수
"""

import time
import traceback

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from typing import Tuple, Optional

from app.services.poem_config import MODEL_TYPE, GEN_MODEL_ID

# ======== 글로벌 캐시 ========
_gen_tok: Optional[AutoTokenizer] = None
_gen_model: Optional[AutoModelForCausalLM] = None
_gen_tok_kogpt2: Optional[AutoTokenizer] = None
_gen_model_kogpt2: Optional[AutoModelForCausalLM] = None
_gen_tok_solar: Optional[AutoTokenizer] = None
_gen_model_solar: Optional[AutoModelForCausalLM] = None


def _is_gpu() -> bool:
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


def _device_info() -> str:
    try:
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return f"CUDA GPU(name={name}, VRAM≈{mem:.1f}GB)"
            except Exception as e:
                return f"CUDA GPU(unknown, error={str(e)[:50]})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "MPS (Apple Silicon GPU)"
        return "CPU"
    except Exception as e:
        return f"Error: {str(e)[:50]}"


def _log_header(title: str):
    print("[_log_header] 함수 시작", flush=True)
    try:
        print("[_log_header] 첫 번째 print 전", flush=True)
        print("\n" + "=" * 80, flush=True)
        print("[_log_header] 두 번째 print 전", flush=True)
        print(f"[poem_generator] {title}", flush=True)
        print("[_log_header] 세 번째 print 전", flush=True)
        print("=" * 80, flush=True)
        print("[_log_header] 함수 완료", flush=True)
    except Exception as e:
        print(f"[_log_header] 오류 발생: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def _load_poem_model(model_type: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    모델 로딩 (캐시 지원)
    model_type: "solar" 또는 "kogpt2", None이면 기본값(MODEL_TYPE) 사용
    """
    global _gen_tok, _gen_model, _gen_tok_kogpt2, _gen_model_kogpt2, _gen_tok_solar, _gen_model_solar
    
    # 사용할 모델 타입 결정
    from app.services.poem_config import MODEL_TYPE as DEFAULT_MODEL_TYPE
    target_model_type = (model_type or DEFAULT_MODEL_TYPE).lower()
    
    if target_model_type not in ["solar", "kogpt2"]:
        print(f"[_load_poem_model] ⚠️ 잘못된 모델 타입: {target_model_type}, 기본값 사용")
        target_model_type = DEFAULT_MODEL_TYPE
    
    # 캐시 확인
    if target_model_type == "kogpt2":
        if _gen_tok_kogpt2 is not None and _gen_model_kogpt2 is not None:
            print("[_load_poem_model] 캐시된 koGPT2 모델 재사용", flush=True)
            return _gen_tok_kogpt2, _gen_model_kogpt2
    else:  # solar
        if _gen_tok_solar is not None and _gen_model_solar is not None:
            print("[_load_poem_model] 캐시된 SOLAR 모델 재사용", flush=True)
            return _gen_tok_solar, _gen_model_solar
    
    # 기존 캐시 확인 (하위 호환성)
    if _gen_tok is not None and _gen_model is not None and target_model_type == DEFAULT_MODEL_TYPE:
        print("[_load_poem_model] 캐시된 토크나이저/모델 재사용", flush=True)
        return _gen_tok, _gen_model

    print("[_load_poem_model] _log_header 호출 전", flush=True)
    try:
        _log_header("모델 로딩 시작")
        print("[_load_poem_model] _log_header 정상 완료", flush=True)
    except Exception as e:
        print(f"[_load_poem_model] _log_header 오류: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    print("[_load_poem_model] _log_header 호출 후", flush=True)
    print("[_load_poem_model] 헤더 출력 완료", flush=True)
    
    # 모델 타입에 따라 모델 ID 결정
    if target_model_type == "kogpt2":
        model_id = "skt/kogpt2-base-v2"
    else:
        model_id = "upstage/SOLAR-10.7B-Instruct-v1.0"
    
    print(f"[_load_poem_model] 모델 타입: {target_model_type}, 모델 ID: {model_id}", flush=True)
    print("[_load_poem_model] 디바이스 정보 확인 중...", flush=True)
    try:
        device_info = _device_info()
        print(f"[_load_poem_model] 실행 디바이스: {device_info}", flush=True)
    except Exception as e:
        print(f"[_load_poem_model] ⚠️ 디바이스 정보 확인 실패: {e}", flush=True)
        device_info = "알 수 없음"
    
    start_all = time.time()
    print(f"[_load_poem_model] 시작 시간 기록 완료", flush=True)

    # 1) 토크나이저
    print("[_load_poem_model] ===== 1단계: 토크나이저 로딩 시작 =====")
    t0 = time.time()
    print("[_load_poem_model] 토크나이저 로딩 중...")
    print(f"[_load_poem_model] 모델 ID: {GEN_MODEL_ID}")
    print("[_load_poem_model] AutoTokenizer.from_pretrained() 호출 전...")
    
    try:
        print("[_load_poem_model] ⏳ 토크나이저 다운로드/로딩 중 (이 과정은 시간이 걸릴 수 있습니다)...")
        import sys
        sys.stdout.flush()  # 버퍼 강제 출력
        
        tok = AutoTokenizer.from_pretrained(model_id)
        print("[_load_poem_model] ✓ AutoTokenizer.from_pretrained() 호출 완료")
        sys.stdout.flush()
        
        print("[_load_poem_model] ✓ 토크나이저 객체 생성 완료")
        
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            print("[_load_poem_model] pad_token → eos_token으로 설정")
        
        print(f"[_load_poem_model] ✓ 토크나이저 로딩 완료 ({time.time() - t0:.2f}s)")
        print(f"[_load_poem_model] 토크나이저 vocab 크기: {len(tok)}")
    except Exception as e:
        print(f"[_load_poem_model] ❌ 토크나이저 로딩 실패: {e}")
        traceback.print_exc()
        raise Exception(f"토크나이저 로딩 실패: {str(e)[:200]}")

    # 2) 모델
    if target_model_type == "kogpt2":
        # koGPT2는 작은 모델이므로 CPU에서도 실행 가능 (양자화 불필요)
        print(f"[_load_poem_model] koGPT2 모델 로딩 (CPU/GPU 모두 가능)")
        t1 = time.time()
        try:
            # 디바이스 선택 (CUDA > MPS > CPU)
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32  # MPS는 float16을 완전히 지원하지 않을 수 있음
            else:
                device = "cpu"
                dtype = torch.float32
            
            print(f"[_load_poem_model] 디바이스: {device}")
            print("[_load_poem_model] 모델 다운로드 및 로딩 시작...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            model = model.to(device).eval()
            print("[_load_poem_model] ✓ 모델 객체 생성 및 eval 모드 설정 완료")
            print(f"[_load_poem_model] ✓ 모델 로딩 완료 ({time.time() - t1:.2f}s, 디바이스: {device})")
            
            # 캐시에 저장
            _gen_tok_kogpt2 = tok
            _gen_model_kogpt2 = model
            _gen_tok = tok  # 하위 호환성
            _gen_model = model
        except Exception as e:
            print(f"[_load_poem_model] ❌ koGPT2 모델 로딩 실패: {e}")
            traceback.print_exc()
            raise Exception(f"koGPT2 모델 로딩 실패: {str(e)[:200]}")
    elif _is_gpu():
        print("[_load_poem_model] GPU 감지됨 → 4bit NF4 양자화 + device_map=auto")
        try:
            # GPU 메모리 상태 확인 (로딩 전)
            print("[_load_poem_model] GPU 정보 확인 중...")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"[_load_poem_model] ✓ GPU 정보: {gpu_name}")
            print(f"[_load_poem_model] ✓ GPU 메모리: 총 {gpu_mem_total:.1f}GB, 사용 중 {gpu_mem_allocated:.2f}GB")
        except Exception as e:
            print(f"[_load_poem_model] ⚠️ GPU 정보 확인 실패: {e}")
        
        print("[_load_poem_model] BitsAndBytesConfig 설정 중...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("[_load_poem_model] ✓ BitsAndBytesConfig 설정 완료")
        
        t1 = time.time()
        try:
            print("[_load_poem_model] 모델 다운로드 및 로딩 시작...")
            print("[_load_poem_model] ⏳ 이 과정은 몇 분이 걸릴 수 있습니다 (모델 크기: ~21GB)")
            print("[_load_poem_model] ⏳ 진행 상황을 기다려주세요...")
            
            # 모델 로딩 시도
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("[_load_poem_model] ✓ 모델 객체 생성 완료")
            
            print("[_load_poem_model] 모델을 eval 모드로 설정 중...")
            model.eval()
            print("[_load_poem_model] ✓ eval 모드 설정 완료")
            
            # 캐시에 저장
            _gen_tok_solar = tok
            _gen_model_solar = model
            _gen_tok = tok  # 하위 호환성
            _gen_model = model
            
            load_time = time.time() - t1
            print(f"[_load_poem_model] ✓ 모델 로딩/배치 완료 ({load_time:.2f}s)")
            
            # GPU 메모리 상태 확인 (로딩 후)
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"[_load_poem_model] GPU 메모리 (로딩 후): 할당={gpu_mem_allocated:.2f}GB, 캐시={gpu_mem_reserved:.2f}GB")
            except:
                pass
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[_load_poem_model] ❌ 4bit 로딩 실패 (RuntimeError): {error_msg}")
            try:
                import traceback as tb
                tb.print_exc()
            except:
                pass
            if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                raise Exception(f"GPU 메모리 부족 또는 CUDA 오류: 모델을 로드할 수 없습니다. 런타임을 재시작하거나 더 큰 GPU를 사용하세요. ({error_msg[:200]})")
            raise Exception(f"모델 로딩 실패: {error_msg[:200]}")
        except Exception as e:
            error_msg = str(e)
            print(f"[_load_poem_model] ❌ 4bit 로딩 실패: {error_msg}")
            try:
                import traceback as tb
                tb.print_exc()
            except:
                pass
            raise Exception(f"모델 로딩 중 오류 발생: {error_msg[:200]}")
    else:
        print("[_load_poem_model] ⚠️ GPU 없음 → CPU float32 로드(매우 느림, 권장 X)")
        print("[_load_poem_model] ⚠️ CPU 모드는 매우 느리며 메모리 부족이 발생할 수 있습니다")
        print("[_load_poem_model] 💡 CPU를 사용하시려면 POEM_MODEL_TYPE=kogpt2 환경 변수를 설정하세요")
        t1 = time.time()
        try:
            print("[_load_poem_model] CPU 모드 모델 다운로드 및 로딩 시작...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            print("[_load_poem_model] ✓ 모델 객체 생성 완료")
            model = model.to("cpu").eval()
            print("[_load_poem_model] ✓ CPU로 이동 및 eval 모드 설정 완료")
            print(f"[_load_poem_model] ✓ 모델 로딩 완료 ({time.time() - t1:.2f}s)")
            
            # 캐시에 저장
            if target_model_type == "kogpt2":
                _gen_tok_kogpt2 = tok
                _gen_model_kogpt2 = model
            else:
                _gen_tok_solar = tok
                _gen_model_solar = model
            _gen_tok = tok  # 하위 호환성
            _gen_model = model
        except Exception as e:
            print(f"[_load_poem_model] ❌ CPU 모드 모델 로딩 실패: {e}")
            traceback.print_exc()
            raise Exception(f"CPU 모드 모델 로딩 실패: {str(e)[:200]}")

    print(f"[_load_poem_model] 총 로딩 시간: {time.time() - start_all:.2f}s")
    return tok, model


# 외부에서 사용할 수 있도록 함수 export
__all__ = ['_load_poem_model', '_is_gpu', '_device_info']

