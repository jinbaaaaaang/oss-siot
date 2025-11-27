# -*- coding: utf-8 -*-
"""
Colab에서 학습한 모델을 로컬에서 사용하는 스크립트

사용 방법:
1. Colab에서 학습한 모델 폴더를 다운로드
2. 아래 코드에서 model_path를 모델 폴더 경로로 설정
3. 실행
"""

import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# 키워드 추출을 위한 import (선택적)
try:
    import sys
    backend_path = Path(__file__).parent
    sys.path.insert(0, str(backend_path))
    from app.services.keyword_extractor import extract_keywords
    HAS_KEYWORD_EXTRACTOR = True
except ImportError:
    HAS_KEYWORD_EXTRACTOR = False

# ===== 설정 =====
# Colab에서 다운로드한 모델 폴더 경로 설정
MODEL_PATH = ""  # 실제 모델 경로로 변경하세요

# CPU 강제 사용 여부 (True로 설정하면 GPU가 있어도 CPU 사용)
FORCE_CPU = False  # CPU만 사용하려면 True로 변경

# GPU 사용 여부 (자동 감지, FORCE_CPU가 True면 무시됨)
USE_GPU = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())


def load_trained_model(model_path: str):
    """학습된 모델 로드"""
    print(f"\n{'='*80}")
    print(f"모델 로딩: {model_path}")
    print(f"{'='*80}\n")
    
    # 디바이스 선택
    if FORCE_CPU:
        device = "cpu"
        dtype = torch.float32
        print(f"🔧 CPU 강제 사용 모드")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float32
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        print(f"✅ Apple Silicon GPU 사용")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"⚠️ CPU 모드 (GPU 없음, 느림)")
    
    # 토크나이저 로드
    print(f"[1/2] 토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ 토크나이저 로딩 완료")
    
    # 모델 로드
    print(f"[2/2] 모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype
    )
    model = model.to(device).eval()
    print(f"✅ 모델 로딩 완료 (디바이스: {device})\n")
    
    return tokenizer, model, device


def generate_poem_from_prose(
    prose_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_new_tokens: int = 150  # 시 길이 조절: 100 → 150 (적당한 길이의 시 생성)
) -> str:
    """
    산문을 입력받아 시를 생성합니다.
    
    학습 형식: "산문: [내용]\n시: [내용]"
    따라서 입력은 "산문: [내용]\n시: " 형식으로 제공
    """
    # 학습 시 사용한 형식으로 입력 구성
    # 학습 형식: "산문: [내용]\n시: [시 내용]"
    # 따라서 입력은 "산문: [내용]\n시: " 형식으로 제공
    # 원문의 주제를 더 잘 반영하도록 프롬프트 강화
    
    # 원문에서 핵심 키워드 추출 (선택적, 키워드 추출기가 있으면 사용)
    enhanced_prose = prose_text.strip()
    keyword_text = ""
    if HAS_KEYWORD_EXTRACTOR and len(prose_text) > 10:
        try:
            # 원문에서 중요한 키워드 추출 (최대 5개)
            keywords = extract_keywords(prose_text, max_keywords=5)
            if keywords:
                # 키워드를 프롬프트에 포함하여 주제 일관성 강화
                keyword_text = ", ".join(keywords[:3])  # 상위 3개만 사용
        except Exception:
            # 키워드 추출 실패 시 원문 그대로 사용
            pass
    
    # 프롬프트 구성: 원문의 의미를 더 강조
    # 원문의 내용을 명확히 반영하도록 프롬프트 강화
    # 키워드가 있으면 간결하게 주제 강조
    if keyword_text:
        # 키워드를 프롬프트에 명확히 포함하여 주제 일관성 강화
        enhanced_prose = f"{prose_text.strip()}\n주제: {keyword_text}"
    else:
        # 키워드가 없어도 원문 내용을 그대로 사용
        enhanced_prose = prose_text.strip()
    
    input_text = f"산문: {enhanced_prose}\n시: "
    
    # 토크나이즈
    enc_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    prompt_length = enc_ids.shape[1]
    
    # attention_mask 생성 (경고 방지)
    attention_mask = torch.ones_like(enc_ids)
    
    # 입력 토큰 길이 제한
    max_pos_embeddings = getattr(model.config, 'max_position_embeddings', 1024)
    safe_max_input = max_pos_embeddings - 100
    if enc_ids.shape[1] >= safe_max_input:
        enc_ids = enc_ids[:, :safe_max_input]
        prompt_length = enc_ids.shape[1]
    
    # 시 생성 (산문이 아닌 시를 생성하도록 파라미터 조정)
    # 원문의 주제를 더 잘 반영하도록 temperature를 더 낮춤
    with torch.no_grad():
        output = model.generate(
            enc_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.6,  # 0.65 → 0.6 (더 일관성 있게, 원문 주제 유지 강화)
            top_p=0.8,  # 0.85 → 0.8 (더 집중적인 샘플링으로 원문 주제 일관성 향상)
            top_k=30,  # 35 → 30 (더 제한적인 토큰 선택으로 원문 내용 반영 강화)
            repetition_penalty=1.6,  # 산문 반복 방지
            no_repeat_ngram_size=5,  # 더 긴 반복 방지
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,  # EOS 토큰 감지 시 즉시 중단
        )
    
    # 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 프롬프트 제거 (토큰 기준으로 제거)
    if len(output[0]) > prompt_length:
        generated_tokens = output[0][prompt_length:]
        poem = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # 토큰 기준 제거가 안 되면 텍스트 기준으로 시도
        if "시: " in generated_text:
            # 마지막 "시: " 이후만 추출
            parts = generated_text.split("시: ")
            if len(parts) > 1:
                poem = parts[-1].strip()  # 마지막 부분만 사용
            else:
                poem = generated_text.strip()
        else:
            poem = generated_text.strip()
    
    # 프롬프트 패턴 제거 (혹시 모를 경우를 대비)
    # "시: "로 시작하는 줄 제거, 반복되는 "시: " 패턴 제거
    prompt_patterns = [
        r'^시:\s*',  # 줄 시작의 "시: " 제거
        r'시:\s*시:\s*',  # 반복되는 "시: 시: " 제거
        r'산문:.*?\n',  # "산문: ..." 패턴 제거
        r'\s*시:\s*',  # 중간에 삽입된 "시: " 제거 (앞뒤 공백 포함)
        r'시:\s*',  # 모든 "시: " 패턴 제거 (앞뒤 공백 없이도)
    ]
    
    for pattern in prompt_patterns:
        poem = re.sub(pattern, '', poem, flags=re.IGNORECASE | re.MULTILINE)
    
    poem = poem.strip()
    
    # 후처리: 빈 줄 제거 및 프롬프트가 포함된 줄 제거
    poem_lines = []
    for line in poem.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # "시:" 패턴이 중간에 있으면 제거
        # "시: " 또는 "시:" 패턴을 모두 제거
        line = re.sub(r'시:\s*', '', line)  # "시: " 또는 "시:" 제거
        line = re.sub(r'\s*시:\s*', '', line)  # 앞뒤 공백 포함 "시:" 제거
        
        # 프롬프트가 포함된 줄 제거
        if any(keyword in line for keyword in ['산문:', 'Write a Korean poem', 'Poem:', '**CRITICAL']):
            # 프롬프트 패턴은 제거
            continue
        
        # "시:" 키워드만 남은 경우 제거
        if line.strip() in ['시', '시:', '시: ']:
            continue
        
        if line:  # 내용이 있으면 추가
            poem_lines.append(line)
    
    poem = '\n'.join(poem_lines)
    
    # 문장 끝 및 쉼표에서 자동 줄바꿈 처리
    # 한 줄에 여러 문장이 있는 경우 문장 단위로 분리하여 각 문장을 한 줄씩 배치
    # 문장 끝 패턴: 마침표, 느낌표, 물음표, "~다", "~요", "~라", "~아", "~어", "~오", "~우" 등
    # 쉼표에서도 줄바꿈 추가
    sentence_end_markers = ['.', '!', '?', '다', '요', '라', '아', '어', '오', '우', '네', '죠']
    comma_markers = [',', '，']  # 쉼표 (한글, 영문)
    
    # 문장 단위로 분리
    sentences = []
    current_sentence = ""
    
    # 공백으로 분리된 단어들을 순회하면서 문장 끝을 찾음
    words = poem.split()
    for word in words:
        # "시:" 패턴이 포함된 단어는 제거
        if '시:' in word:
            word = word.replace('시:', '').replace('시', '')
            if not word:  # 단어가 모두 제거되면 스킵
                continue
        
        current_sentence += word + " "
        
        # 쉼표 마커 확인 (문장 끝보다 먼저 체크)
        is_comma = False
        for marker in comma_markers:
            if word.endswith(marker):
                is_comma = True
                break
        
        # 문장 끝 마커 확인
        is_sentence_end = False
        for marker in sentence_end_markers:
            if word.endswith(marker):
                is_sentence_end = True
                break
        
        # 쉼표가 있으면 줄바꿈 (문장 끝보다 우선)
        if is_comma:
            cleaned_sentence = current_sentence.strip()
            # "시:" 패턴이 남아있으면 제거
            cleaned_sentence = re.sub(r'\s*시:\s*', '', cleaned_sentence)
            if cleaned_sentence:  # 내용이 있으면 추가
                sentences.append(cleaned_sentence)
            current_sentence = ""
        # 문장이 끝나면 리스트에 추가하고 새 문장 시작
        elif is_sentence_end:
            cleaned_sentence = current_sentence.strip()
            # "시:" 패턴이 남아있으면 제거
            cleaned_sentence = re.sub(r'\s*시:\s*', '', cleaned_sentence)
            if cleaned_sentence:  # 내용이 있으면 추가
                sentences.append(cleaned_sentence)
            current_sentence = ""
    
    # 마지막 문장이 있으면 추가
    if current_sentence.strip():
        cleaned_sentence = current_sentence.strip()
        # "시:" 패턴이 남아있으면 제거
        cleaned_sentence = re.sub(r'\s*시:\s*', '', cleaned_sentence)
        if cleaned_sentence:
            sentences.append(cleaned_sentence)
    
    # 각 문장을 한 줄씩 배치
    if sentences:
        poem = '\n'.join(sentences)
    else:
        # 문장 분리가 안 되면 원본 유지
        pass
    
    # 산문 패턴 제거 및 시 형식 강화
    # 산문 특징: 긴 문장, "~다", "~요", "~이다" 등으로 끝나는 문장, 쉼표가 많은 문장
    prose_indicators = [
        r'[가-힣]{10,}다\.',  # 10자 이상 + "다." 패턴 (예: "오늘은 정말 좋은 하루였다.")
        r'[가-힣]{10,}요\.',  # 10자 이상 + "요." 패턴
        r'[가-힣]{10,}이다\.',  # 10자 이상 + "이다." 패턴
        r'[가-힣]{15,}[,，]',  # 15자 이상 + 쉼표 (산문 특징)
        r'[가-힣]{20,}',  # 20자 이상의 긴 문장 (시는 보통 짧음)
    ]
    
    # 뉴스 기사나 실제 정보를 담은 문장 패턴 (제거 대상)
    news_info_keywords = [
        '한국도로공사', '서울톨게이트', '에 따르면', '이날', '오후', '오전', '시부터', '시까지',
        '고속도로', '분기점', '방향', '톨게이트', '공사', '건설', '발표', '발표에 따르면',
        '보도에 따르면', '관계자', '당국', '기관', '부서', '청', '시청', '구청',
        '경부', '경인', '서해안', '중앙', '영동', '호남', '중부', '동해',
        '부산방향', '서울방향', '인천방향', '대전방향', '대구방향',
        '신갈', '안성', '한남', '판교', '기흥', '수원', '용인',
        'km', 'm', 'km/h', '원', '만원', '억원',
        '년', '월', '일', '시', '분',
        '전망', '예상', '계획', '추진', '검토', '논의',
    ]
    
    # 산문처럼 보이는 문장 및 뉴스 기사 스타일 문장 제거
    filtered_lines = []
    for line in poem.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 산문 패턴 체크
        is_prose = False
        is_news_info = False
        
        # 1. 뉴스 기사나 실제 정보를 담은 문장 체크 (우선 처리)
        for keyword in news_info_keywords:
            if keyword in line:
                is_news_info = True
                break
        
        # 2. 너무 긴 문장 (30자 이상으로 완화)
        if len(line) > 30:  # 20 → 30 (더 긴 줄 허용)
            # 산문 패턴이 있는지 확인
            for pattern in prose_indicators:
                if re.search(pattern, line):
                    is_prose = True
                    break
            
            # 3. 쉼표가 3개 이상인 긴 문장도 산문으로 간주 (2개 → 3개로 완화)
            if not is_prose and line.count(',') >= 3 and len(line) > 20:  # 2개 → 3개, 15자 → 20자
                is_prose = True
        
        # 뉴스 기사 스타일이거나 산문이 아닌 경우만 추가
        if not is_news_info and not is_prose:
            filtered_lines.append(line)
    
    poem = '\n'.join(filtered_lines)
    
    # 빈 결과 처리
    if not poem.strip():
        # 필터링이 너무 강해서 모든 줄이 제거된 경우
        # 원본에서 짧은 줄만 선택
        original_lines = '\n'.join(poem_lines).split('\n')
        short_lines = [line.strip() for line in original_lines if line.strip() and len(line.strip()) <= 30]  # 20 → 30
        if short_lines:
            poem = '\n'.join(short_lines[:15])  # 최대 10줄 → 15줄
    
    # 시 길이 제한 (너무 길면 자동으로 잘라내기)
    # 보통 시는 15-20줄 정도가 적당
    max_lines = 20  # 15 → 20 (더 긴 시 허용)
    poem_lines_final = poem.split('\n')[:max_lines]
    poem = '\n'.join(poem_lines_final).strip()
    
    # 최대 문자 수 제한 (약 800자로 증가)
    # 문장이 중간에 잘리지 않도록 처리
    if len(poem) > 800:  # 500 → 800 (더 긴 시 허용)
        lines = poem.split('\n')
        result_lines = []
        total_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # 줄바꿈 포함
            if total_length + line_length > 500:
                # 제한을 넘으면 중단 (마지막 줄은 제외하여 불완전한 문장 방지)
                break
            result_lines.append(line)
            total_length += line_length
        
        poem = '\n'.join(result_lines).strip()
    
    # 마지막 문장이 불완전하게 잘렸는지 확인 및 처리
    # 문장 끝 패턴: 마침표, 느낌표, 물음표, 줄바꿈, 또는 시적 표현으로 끝나는 경우
    if poem:
        lines = poem.split('\n')
        if lines:
            last_line = lines[-1].strip()
            
            # 불완전한 문장 패턴 체크
            incomplete_patterns = [
                r'[가-힣]+[은는이가을를]$',  # 조사로 끝나는 경우 (예: "나는", "그는")
                r'[가-힣]+[와과]$',  # 접속조사로 끝나는 경우
                r'[가-힣]+[에에서]$',  # 부사격 조사로 끝나는 경우
                r'[가-힣]+[의]$',  # 관형격 조사로 끝나는 경우
                r'[가-힣]+[도]$',  # 보조사로 끝나는 경우
            ]
            
            is_incomplete = False
            for pattern in incomplete_patterns:
                if re.search(pattern, last_line):
                    is_incomplete = True
                    break
            
            # 마지막 줄이 불완전하면 제거
            if is_incomplete and len(lines) > 1:
                lines = lines[:-1]
                poem = '\n'.join(lines).strip()
            
            # 마지막 줄이 너무 짧고(5자 이하) 불완전해 보이면 제거
            elif len(last_line) <= 5 and not any(last_line.endswith(c) for c in ['.', '!', '?', '다', '요', '라', '아', '어', '오', '우']):
                if len(lines) > 1:
                    lines = lines[:-1]
                    poem = '\n'.join(lines).strip()
    
    return poem


def main():
    """메인 함수"""
    # 모델 경로 확인
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ 모델 경로를 찾을 수 없습니다: {MODEL_PATH}")
        print(f"\n💡 사용 방법:")
        print(f"   1. Colab에서 학습한 모델 폴더를 다운로드")
        print(f"   2. 이 스크립트의 MODEL_PATH 변수를 모델 폴더 경로로 변경")
        print(f"   3. 다시 실행")
        return
    
    # 모델 로드
    try:
        tokenizer, model, device = load_trained_model(str(model_path))
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 인터랙티브 모드
    print(f"\n{'='*80}")
    print("시 생성 시작 (종료하려면 'quit' 또는 'exit' 입력)")
    print(f"{'='*80}\n")
    
    while True:
        try:
            # 사용자 입력
            prose_text = input("산문을 입력하세요: ").strip()
            
            if prose_text.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n종료합니다.")
                break
            
            if not prose_text:
                print("⚠️ 텍스트를 입력하세요.")
                continue
            
            # 시 생성
            print(f"\n시 생성 중...")
            poem = generate_poem_from_prose(prose_text, tokenizer, model, device)
            
            # 결과 출력
            print(f"\n{'='*80}")
            print("생성된 시:")
            print(f"{'='*80}")
            print(poem)
            print(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

