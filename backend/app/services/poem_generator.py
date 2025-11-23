# -*- coding: utf-8 -*-
"""
시 생성 메인 함수
"""

import time
import traceback
from typing import List, Optional

import torch

# 번역 모듈 import
from app.services.translator import (
    translate_poem_with_retry,
    detect_language,
)

# 분리된 모듈들 import
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


def _log_header(title: str):
    """로깅 헤더 출력"""
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


def _debug_tensor(name: str, tensor: torch.Tensor):
    """텐서 디버깅 정보 출력"""
    try:
        print(f"[debug] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    except Exception:
        print(f"[debug] {name}: <unavailable>")


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
    model_type: Optional[str] = None,  # "solar" 또는 "kogpt2", None이면 기본값 사용
) -> str:
    """
    키워드/분위기 기반 시 생성 (채팅 템플릿 + 4bit 양자화).
    템플릿 폴백 없음 → 실패 시 예외 발생.
    """
    _log_header("시 생성 함수 진입")
    func_start = time.time()
    print(f"[args] keywords={keywords}")
    print(f"[args] mood={mood}, lines={lines}, max_new_tokens={max_new_tokens}")
    print(f"[env] device={_device_info()}, USE_ML_MODEL={USE_ML_MODEL}")

    if not USE_ML_MODEL:
        raise RuntimeError("USE_ML_MODEL=False 상태입니다. 이 구현은 ML 생성만 지원합니다.")

    # 사용할 모델 타입 결정
    from app.services.poem_config import MODEL_TYPE as DEFAULT_MODEL_TYPE
    actual_model_type = (model_type or DEFAULT_MODEL_TYPE).lower()
    if actual_model_type not in ["solar", "kogpt2"]:
        print(f"[generate_poem_from_keywords] ⚠️ 잘못된 모델 타입: {actual_model_type}, 기본값 사용")
        actual_model_type = DEFAULT_MODEL_TYPE

    # 1) 모델 로딩
    print(f"[step] 1. 모델 로딩 (모델 타입: {actual_model_type})")
    tok, model = _load_poem_model(actual_model_type)

    # 2) 프롬프트 구성 및 토크나이즈 (모델 타입에 따라 분기)
    print(f"[step] 2. 프롬프트 구성 및 토크나이즈 (모델 타입: {actual_model_type})")
    t_enc = time.time()
    
    if actual_model_type == "kogpt2":
        # koGPT2는 chat template 없이 일반 토크나이저 사용
        prompt_text = _build_messages_kogpt2(
            keywords, mood, lines, original_text, banned_words, use_rhyme, acrostic
        )
        print(f"[step] koGPT2 프롬프트 텍스트:")
        print(f"  {repr(prompt_text[:400])}")
        
        try:
            enc_ids = tok.encode(prompt_text, return_tensors="pt")
            print(f"[step] ✓ 토크나이즈 완료 (길이: {enc_ids.shape[1]} 토큰)")
        except Exception as e:
            print(f"[error] koGPT2 토크나이즈 오류: {e}")
            traceback.print_exc()
            raise Exception(f"프롬프트 토크나이즈 실패: {str(e)[:200]}")
    else:
        # SOLAR는 chat template 사용
        messages = _build_messages(
            keywords, mood, lines, original_text, banned_words, use_rhyme, acrostic
        )
        
        # 프롬프트에 키워드가 제대로 반영되었는지 확인
        print(f"[step] 입력 키워드: {keywords}")
        if len(messages) > 1 and isinstance(messages[1], dict) and "content" in messages[1]:
            user_msg_content = messages[1]["content"]
        else:
            user_msg_content = ""
            print(f"[warning] 메시지 형식이 예상과 다릅니다: {len(messages)}개 메시지")
        print(f"[step] 프롬프트에 포함된 키워드 확인:")
        max_kw_check = min(5, len(keywords)) if keywords else 0
        for kw in keywords[:max_kw_check] if keywords else []:
            if kw in user_msg_content:
                print(f"  ✓ '{kw}' 포함됨")
            else:
                print(f"  ⚠️ '{kw}' 누락됨!")
        
        # 프롬프트 텍스트 미리보기 (디버깅용)
        try:
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(f"[step] 최종 프롬프트 텍스트 (앞 400자):")
            print(f"  {repr(prompt_text[:400])}")
            
            prompt_lower = prompt_text.lower()
            max_kw_check = min(5, len(keywords)) if keywords else 0
            keywords_found = [
                kw for kw in (keywords[:max_kw_check] if keywords else [])
                if kw.lower() in prompt_lower
            ]
            if keywords_found:
                print(
                    f"[step] ✓ 최종 프롬프트에 {len(keywords_found)}개 키워드 포함됨: "
                    f"{keywords_found[:min(3, len(keywords_found))]}"
                )
            else:
                print(f"[step] ⚠️ 최종 프롬프트에 키워드가 포함되지 않았습니다!")
        except Exception as e:
            print(f"[step] 프롬프트 텍스트 미리보기 실패: {e}")
        
        try:
            enc_result = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # apply_chat_template는 return_tensors="pt"일 때 텐서를 반환
            if isinstance(enc_result, torch.Tensor):
                if enc_result.dim() == 1:
                    enc_ids = enc_result.unsqueeze(0)
                elif enc_result.dim() == 2 and enc_result.shape[0] == 1:
                    enc_ids = enc_result
                else:
                    print(f"[warning] 예상치 못한 텐서 형태: {enc_result.shape}, 첫 번째 배치만 사용")
                    enc_ids = enc_result[0:1]
            elif isinstance(enc_result, list):
                enc_ids = torch.tensor([enc_result], dtype=torch.long)
            else:
                enc_ids = torch.tensor([[enc_result]], dtype=torch.long)
            
            enc_ids = enc_ids.to(dtype=torch.long)
            
            if enc_ids.shape[0] != 1:
                raise Exception(f"입력 텐서의 배치 크기가 1이 아닙니다: {enc_ids.shape}")
            if enc_ids.shape[1] < 10:
                raise Exception(f"입력 텐서가 너무 짧습니다: {enc_ids.shape[1]} 토큰")
            if enc_ids.shape[1] > 2048:
                print(f"[warning] ⚠️ 입력 텐서가 매우 깁니다: {enc_ids.shape[1]} 토큰")
        except Exception as e:
            print(f"[error] chat 템플릿 적용 오류: {e}")
            traceback.print_exc()
            raise Exception(f"프롬프트 토크나이즈 실패: {str(e)[:200]}")
    
    print(f"[step] ✓ 토크나이즈 완료 ({time.time() - t_enc:.2f}s)")
    _debug_tensor("input_ids(raw)", enc_ids)

    # 모델의 최대 위치 임베딩 길이 확인 및 입력 길이 제한
    try:
        max_pos_embeddings = getattr(model.config, 'max_position_embeddings', None)
        if max_pos_embeddings is None:
            # GPT2 계열 모델의 기본값
            max_pos_embeddings = 1024 if actual_model_type == "kogpt2" else 2048
            print(f"[step] 모델 config에서 max_position_embeddings를 찾을 수 없어 기본값 사용: {max_pos_embeddings}")
        else:
            print(f"[step] 모델 max_position_embeddings: {max_pos_embeddings}")
        
        # position embedding은 0부터 시작 → 인덱스 0 ~ (max_pos_embeddings-1)
        safe_max_input = max_pos_embeddings - 100  # 생성 토큰 공간 확보
        if safe_max_input < 100:
            safe_max_input = max_pos_embeddings - 50
        if safe_max_input < 50:
            safe_max_input = max_pos_embeddings - 1
        
        current_input_len = enc_ids.shape[1]
        if current_input_len >= safe_max_input:
            print(
                f"[warning] ⚠️ 입력 토큰 길이({current_input_len})가 안전한 최대 길이({safe_max_input})를 초과합니다. 자릅니다."
            )
            print(
                f"[warning] 모델 max_position_embeddings: {max_pos_embeddings}, "
                f"생성 공간 확보를 위해 {safe_max_input}로 제한"
            )
            enc_ids = enc_ids[:, :safe_max_input]
            print(f"[step] ✓ 입력 토큰 길이를 {enc_ids.shape[1]}로 제한했습니다.")
        else:
            print(
                f"[step] ✓ 입력 토큰 길이({current_input_len})가 "
                f"안전한 최대 길이({safe_max_input}) 이내입니다."
            )
    except Exception as e:
        print(f"[warning] 모델 config 확인 중 오류: {e}, 기본 제한(924) 적용")
        if enc_ids.shape[1] >= 924:
            print(f"[warning] 입력 토큰 길이를 924로 제한합니다.")
            enc_ids = enc_ids[:, :924]

    # device_map=auto 일 때 모델 파라미터 device로 이동
    try:
        model_device = next(model.parameters()).device
        print(f"[step] 모델 device 확인: {model_device}")
    except StopIteration:
        print("[error] ❌ 모델 파라미터를 찾을 수 없습니다!")
        raise Exception("모델이 제대로 로드되지 않았습니다. 모델 로딩 상태를 확인하세요.")
    
    # input_ids를 모델 device로 이동
    try:
        input_ids = enc_ids.to(model_device)
        _debug_tensor("input_ids(final)", input_ids)
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[error] ❌ input_ids를 device로 이동 실패: {error_msg}")
        if "out of memory" in error_msg.lower():
            raise Exception("GPU 메모리 부족으로 input_ids를 이동할 수 없습니다. 런타임을 재시작하세요.")
        raise Exception(f"Device 이동 실패: {error_msg[:200]}")
    
    # attention_mask 생성 (패딩 없음 → 전부 1)
    try:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
        print(
            f"[step] attention_mask 생성: shape={tuple(attention_mask.shape)}, "
            f"dtype={attention_mask.dtype}, device={attention_mask.device}"
        )
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[error] ❌ attention_mask 생성 실패: {error_msg}")
        if "out of memory" in error_msg.lower():
            raise Exception("GPU 메모리 부족으로 attention_mask를 생성할 수 없습니다.")
        raise Exception(f"attention_mask 생성 실패: {error_msg[:200]}")

    # 3) 생성 파라미터 확정
    print("[step] 3. 생성 파라미터 설정")
    is_gpu = _is_gpu()
    safe_max_new = max_new_tokens
    
    # 최소 생성 길이
    min_required = max(30, lines * 8)
    if safe_max_new < min_required:
        print(
            f"[warning] max_new_tokens({safe_max_new})가 너무 작습니다. "
            f"최소 {min_required}로 조정합니다.",
            flush=True,
        )
        safe_max_new = min_required
    
    if is_gpu:
        # 둘 다 100으로 맞춤
        safe_max_new = min(safe_max_new, 100)
    else:
        if actual_model_type == "kogpt2":
            safe_max_new = min(safe_max_new, 60)
        else:
            safe_max_new = min(safe_max_new, DEFAULT_MAX_NEW_TOKENS_CPU)
    
    print(f"[step] 최종 max_new_tokens: {safe_max_new} (요청: {max_new_tokens}, 최소: {min_required})")

    # 기본 gen_kwargs
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": safe_max_new,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    
    # GPU 메모리 상태 확인 (디버깅용)
    if _is_gpu():
        try:
            mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(
                f"[step] GPU 메모리 (생성 전): 할당={mem_allocated:.2f}GB, "
                f"캐시={mem_reserved:.2f}GB"
            )
        except:
            pass
    
    # ★★ 여기서 koGPT2용 하이퍼파라미터를 네가 원한 값으로 통일 ★★
    if actual_model_type == "kogpt2":
        # koGPT2: 네가 예시로 준 설정을 그대로 반영
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,  # 반복 줄이기용 옵션 유지
        })
    else:
        # SOLAR 모델
        if is_gpu:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.8,
                "top_k": 25,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
            })
        else:
            gen_kwargs.update({
                "do_sample": False,
            })

    print(
        "[gen_kwargs]",
        {k: (v if not isinstance(v, torch.Tensor) else f"Tensor(shape={tuple(v.shape)})")
         for k, v in gen_kwargs.items()}
    )

    # 4) 생성
    print("[step] 4. model.generate() 호출")
    print(f"[step] 생성 파라미터 요약:")
    print(f"   - max_new_tokens: {safe_max_new}")
    print(f"   - do_sample: {gen_kwargs.get('do_sample', False)}")
    print(f"   - device: {model_device}")
    
    t_gen = time.time()
    try:
        out = model.generate(**gen_kwargs)
    except IndexError as e:
        error_msg = str(e)
        print(f"[error] generate() IndexError 발생 (position embedding 오류 가능)")
        print(f"[error] 오류 메시지: {error_msg}")
        print(f"[error] 입력 토큰 길이: {input_ids.shape[1]}")
        print(
            f"[error] 모델 max_position_embeddings: "
            f"{getattr(model.config, 'max_position_embeddings', '알 수 없음')}"
        )
        traceback.print_exc()
        raise Exception(
            "입력 토큰 길이가 모델의 최대 길이를 초과했습니다. "
            f"프롬프트가 너무 깁니다. (입력: {input_ids.shape[1]} 토큰)"
        )
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[error] generate() RuntimeError 발생")
        print(f"[error] 오류 메시지: {error_msg}")
        traceback.print_exc()
        
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            raise Exception(f"GPU 메모리 부족 또는 CUDA 오류입니다: {error_msg[:200]}")
        else:
            raise Exception(f"모델 생성 중 런타임 오류: {error_msg[:200]}")
    except ValueError as e:
        error_msg = str(e)
        print(f"[error] generate() ValueError 발생")
        print(f"[error] 오류 메시지: {error_msg}")
        traceback.print_exc()
        raise Exception(f"생성 파라미터 오류입니다: {error_msg[:200]}")
    except Exception as e:
        error_msg = str(e)
        print(f"[error] generate() 예외 발생: {type(e).__name__}")
        print(f"[error] 오류 메시지: {error_msg}")
        traceback.print_exc()
        raise Exception(
            f"시 생성 중 오류가 발생했습니다: {type(e).__name__}: {error_msg[:200]}"
        )
    gen_sec = time.time() - t_gen
    print(f"[step] ✓ 생성 완료 ({gen_sec:.2f}s)")
    try:
        print(f"[debug] output shape={tuple(out.shape)}, device={out.device}")
        if _is_gpu():
            mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(
                f"[step] GPU 메모리 (생성 후): "
                f"할당={mem_allocated:.2f}GB, 캐시={mem_reserved:.2f}GB"
            )
    except Exception:
        pass

    # 5) 디코딩 (프롬프트 길이 이후만)
    print("[step] 5. 디코딩 및 후처리")
    
    if len(input_ids.shape) < 2 or input_ids.shape[1] == 0:
        print(f"[error] ❌ input_ids 형태 오류: {input_ids.shape}")
        raise Exception(f"입력 텐서 형태 오류: {input_ids.shape}")
    input_len = input_ids.shape[1]
    
    if len(out.shape) < 2:
        print(f"[error] ❌ 출력 텐서 차원이 부족합니다: {out.shape}")
        raise Exception(f"출력 텐서 형태 오류: {out.shape}")
    
    if out.shape[0] != 1:
        print(f"[error] ❌ 출력 배치 크기가 1이 아닙니다: {out.shape}")
        raise Exception(f"출력 텐서 형태 오류: {out.shape}")
    
    if out.shape[1] == 0:
        print(f"[error] ❌ 출력 텐서 길이가 0입니다: {out.shape}")
        raise Exception(f"출력 텐서가 비어있습니다: {out.shape}")
    
    output_len = out.shape[1]
    new_tokens = output_len - input_len
    
    print(f"[debug] 입력 길이: {input_len} 토큰, 출력 길이: {output_len} 토큰")
    print(f"[debug] 생성된 새 토큰 수: {new_tokens}")
    
    if new_tokens <= 0:
        print("[error] ❌ 모델이 새로운 토큰을 생성하지 않았습니다!")
        print(f"[error] 입력 토큰 수: {input_len}, 출력 토큰 수: {output_len}")
        print(f"[error] 생성 파라미터를 확인하세요 (max_new_tokens={safe_max_new})")
        
        try:
            if out.shape[0] > 0:
                full_decoded = tok.decode(out[0], skip_special_tokens=False)
                print(f"[error] 전체 출력 (처음 500자): {repr(full_decoded[:500])}")
                if len(full_decoded) > 500:
                    print(f"[error] 전체 출력 (마지막 500자): {repr(full_decoded[-500:])}")
        except Exception as e:
            print(f"[error] 디코딩 실패: {e}")
        
        raise Exception(
            "모델이 새로운 토큰을 생성하지 않았습니다. "
            f"(입력: {input_len}, 출력: {output_len} 토큰)"
        )
    
    if new_tokens < 5:
        print(f"[warning] ⚠️ 생성된 토큰이 매우 적습니다 ({new_tokens} 토큰)")
    
    # 생성된 부분만 디코딩
    try:
        if input_len >= output_len:
            print(f"[error] ⚠️ input_len({input_len}) >= output_len({output_len}), 빈 텐서 사용")
            generated_ids = torch.tensor([], dtype=out.dtype, device=out.device)
        else:
            generated_ids = out[0][input_len:]
        
        decoded = tok.decode(
            generated_ids, skip_special_tokens=True, errors='replace'
        ).strip()
        
        if len(generated_ids) > 0:
            sample_size = min(10, len(generated_ids))
            print(
                f"[debug] 생성된 토큰 ID 샘플 (처음 {sample_size}개): "
                f"{generated_ids[:sample_size].tolist()}"
            )
        else:
            print("[debug] 생성된 토큰 ID가 비어있습니다.")
        print(f"[debug] decoded(앞 300자): {repr(decoded[:300])}")
        print(f"[debug] decoded 전체 길이: {len(decoded)}자")
        
        if len(decoded.strip()) < 10:
            print(f"[warning] ⚠️ 디코딩 결과가 매우 짧습니다: {len(decoded)}자")
            try:
                if out.shape[0] > 0:
                    full_decoded = tok.decode(out[0], skip_special_tokens=False)
                    if len(full_decoded) > 200:
                        print(
                            f"[debug] 전체 출력 확인 (마지막 200자): "
                            f"{repr(full_decoded[-200:])}"
                        )
                    else:
                        print(f"[debug] 전체 출력 확인: {repr(full_decoded)}")
                    
                    if input_len < output_len and input_ids.shape[0] > 0:
                        full_decoded_clean = tok.decode(
                            out[0], skip_special_tokens=True, errors='replace'
                        )
                        input_decoded = tok.decode(
                            input_ids[0], skip_special_tokens=True, errors='replace'
                        )
                        if full_decoded_clean.startswith(input_decoded):
                            decoded_alt = full_decoded_clean[len(input_decoded):].strip()
                            print(
                                f"[debug] 대체 디코딩 방법 결과 (앞 300자): "
                                f"{repr(decoded_alt[:300])}"
                            )
                            if len(decoded_alt) > len(decoded):
                                print("[debug] 대체 방법이 더 긴 결과를 반환했습니다. 이를 사용합니다.")
                                decoded = decoded_alt
            except Exception as e2:
                print(f"[warning] 대체 디코딩 시도 실패: {e2}")
                
    except Exception as e:
        print(f"[error] 디코딩 오류: {e}")
        traceback.print_exc()
        
        try:
            print("[debug] 대체 디코딩 방법 시도 중...")
            if out.shape[0] > 0:
                decoded = tok.decode(
                    out[0], skip_special_tokens=True, errors='replace'
                ).strip()
                if len(decoded) > input_len * 2:
                    decoded = decoded[input_len * 2:].strip()
                print(f"[debug] 대체 디코딩 결과: {repr(decoded[:300])}")
            else:
                raise Exception("출력 텐서가 비어있습니다.")
        except Exception as e2:
            print(f"[error] 대체 디코딩도 실패: {e2}")
            raise Exception(f"디코딩 중 오류가 발생했습니다: {str(e)[:200]}")
    
    if not decoded or len(decoded.strip()) == 0:
        print("[error] ❌ 디코딩 결과가 비어있습니다!")
        print(f"[error] 생성된 토큰 수: {new_tokens}")
        if new_tokens > 0:
            try:
                if out.shape[0] > 0:
                    raw_decoded = tok.decode(out[0], skip_special_tokens=False)
                    print(
                        "[error] 전체 디코딩 (skip_special_tokens=False, 처음 500자): "
                        f"{repr(raw_decoded[:500])}"
                    )
                    if len(raw_decoded) > 500:
                        print(
                            "[error] 전체 디코딩 (skip_special_tokens=False, 마지막 500자): "
                            f"{repr(raw_decoded[-500:])}"
                        )
            except Exception as decode_err:
                print(f"[error] 디코딩 실패: {decode_err}")
        raise Exception("디코딩 결과가 비어있습니다. 생성 파라미터를 확인하세요.")

    poem = _postprocess_poem(decoded, min_lines=lines, max_lines=lines * 3)
    poem = poem.strip()
    print("[debug] postprocessed(앞 200자):", repr(poem[:200]))
    print("[debug] 생성된 텍스트 전체:", repr(poem))
    korean_debug = sum(1 for c in poem if ord('가') <= ord(c) <= ord('힣'))
    chinese_debug = sum(1 for c in poem if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
    print(f"[debug] 언어 분석: 한국어={korean_debug}자, 중국어={chinese_debug}자")

    # 6) 검증: 내용/언어 간단 체크
    if not poem or len(poem.strip()) == 0:
        print("[check] ❌ 생성된 시가 비어있음")
        print(f"[check] 원본 디코딩 결과: {repr(decoded[:500])}")
        raise Exception("시 생성에 실패했습니다. 생성된 내용이 없습니다.")
    
    korean_chars = sum(1 for c in poem if ord('가') <= ord(c) <= ord('힣'))
    english_chars = sum(
        1 for c in poem if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z')
    )
    total_chars_check = len([c for c in poem if c.strip()])
    
    print(
        f"[check] 최종 시 길이={len(poem)}자, "
        f"한글문자수={korean_chars}자, 영어문자수={english_chars}자"
    )
    
    if len(poem.strip()) < 1:
        raise Exception("시 생성에 실패했습니다. 생성된 내용이 너무 짧습니다.")
    
    if total_chars_check > 0:
        english_ratio = english_chars / total_chars_check if total_chars_check > 0 else 0
        if english_ratio > 0.7 and korean_chars < 5:
            print(
                f"[check] ❌ 이상한 출력 감지: 영어 비율 {english_ratio:.1%}, "
                f"한글 {korean_chars}자"
            )
            print(f"[check] 생성된 텍스트: {repr(poem[:200])}")
            raise Exception("시 생성에 실패했습니다. 의미 없는 텍스트가 생성되었습니다. 다시 시도해주세요.")
    
    # 7) 한국어 비율 및 언어 감지
    print("[check] 한국어 검증:")
    korean_chars = sum(1 for c in poem if ord('가') <= ord(c) <= ord('힣'))
    total_chars = len([c for c in poem if c.strip()])
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
    
    print(
        f"[check] 한국어 문자 수: {korean_chars}자 / 전체 문자 수: "
        f"{total_chars}자 (비율: {korean_ratio:.2%})"
    )
    
    chinese_chars = sum(1 for c in poem if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
    japanese_hiragana = sum(1 for c in poem if ord('\u3040') <= ord(c) <= ord('\u309f'))
    japanese_katakana = sum(1 for c in poem if ord('\u30a0') <= ord(c) <= ord('\u30ff'))
    japanese_chars = japanese_hiragana + japanese_katakana
    english_chars = sum(
        1 for c in poem if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z')
    )
    
    detected_lang_name, detected_lang_code = detect_language(poem)
    
    if chinese_chars > 0:
        print(f"[check] ⚠️ 중국어 문자 감지: {chinese_chars}자")
    if japanese_chars > 0:
        print(
            f"[check] ⚠️ 일본어 문자 감지: 히라가나 {japanese_hiragana}자, "
            f"가타카나 {japanese_katakana}자"
        )
    if english_chars > total_chars * 0.5 and korean_chars < total_chars * 0.3:
        print(f"[check] ⚠️ 영어 텍스트 감지: {english_chars}자")
    print(f"[check] 감지된 언어: {detected_lang_name} (코드: {detected_lang_code})")
    
    needs_translation = (
        detected_lang_code != "ko"
        or (korean_chars == 0 and total_chars > 5)
        or (korean_ratio < 0.8 and total_chars > 10)
    )
    
    needs_translation_or_other_language = needs_translation or (
        chinese_chars > 0 
        or japanese_chars > 0 
        or (english_chars > total_chars * 0.2 and korean_chars < total_chars * 0.6)
        or (korean_ratio < 0.8 and total_chars > 10)
    )
    
    if needs_translation_or_other_language:
        poem = translate_poem_with_retry(poem, max_retries=5)
        
        # 번역 후 다시 분석
        korean_chars = sum(1 for c in poem if ord('가') <= ord(c) <= ord('힣'))
        chinese_chars = sum(1 for c in poem if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
        japanese_hiragana = sum(1 for c in poem if ord('\u3040') <= ord(c) <= ord('\u309f'))
        japanese_katakana = sum(1 for c in poem if ord('\u30a0') <= ord(c) <= ord('\u30ff'))
        japanese_chars = japanese_hiragana + japanese_katakana
        english_chars = sum(
            1 for c in poem if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z')
        )
        total_chars = len([c for c in poem if c.strip()])
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
    
    if korean_ratio < 0.5 and total_chars > 10:
        print(f"[check] ⚠️ 한국어 비율이 낮습니다 ({korean_ratio:.2%})")
        if chinese_chars > 0:
            print(f"[check] ⚠️ 중국어가 포함되어 있습니다. 재생성을 권장합니다.")
    
    # 키워드 반영 여부 (참고용)
    print("[check] 키워드 반영 여부 확인:")
    poem_lower = poem.lower()
    keywords_in_poem = []
    max_keywords_to_check = min(5, len(keywords)) if keywords else 0
    for kw in keywords[:max_keywords_to_check] if keywords else []:
        kw_lower = kw.lower()
        if kw_lower in poem_lower:
            keywords_in_poem.append(kw)
            print(f"  ✓ 키워드 '{kw}' 반영됨")
        else:
            if len(kw) >= 2:
                found_partial = False
                for i in range(len(kw) - 1):
                    if kw[i:i+2].lower() in poem_lower:
                        found_partial = True
                        print(f"  ~ 키워드 '{kw}' 일부 반영됨 (참고)")
                        break
                if not found_partial:
                    print(f"  ⚠️ 키워드 '{kw}' 직접 반영 안 됨 (의미적으로는 포함될 수 있음)")
            else:
                print(f"  ⚠️ 키워드 '{kw}' 직접 반영 안 됨 (의미적으로는 포함될 수 있음)")
    
    if keywords_in_poem:
        print(f"[check] ✓ {len(keywords_in_poem)}개 키워드가 시에 반영됨")
    else:
        print("[check] ⚠️ 키워드가 직접적으로 보이지 않지만, 의미적으로는 반영되었을 수 있습니다.")

    print(f"[done] 총 소요 시간: {time.time() - func_start:.2f}s")
    return poem


def generate_poem(keywords: List[str], emotion: str, max_length: int = 120) -> str:
    """
    기존 API 호환: 감정 → 분위기 매핑 후 시 생성.
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