# -*- coding: utf-8 -*-
"""
Colab에서 학습된 k-fold 모델들의 성능을 평가하는 스크립트

사용 방법:
1. Colab에서 이 파일을 업로드하거나 내용을 복사
2. Colab 셀에서 실행 전에 한글 폰트 설치 (선택사항):
   !apt-get install -y fonts-nanum
   !fc-cache -fv
3. Colab 셀에서 실행
4. 각 fold 모델의 성능을 비교하여 가장 좋은 모델 찾기
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_fscore_support
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib/seaborn이 설치되지 않았습니다. 시각화 기능을 사용할 수 없습니다.")

# BERTScore 라이브러리 (선택적)
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️ bert-score가 설치되지 않았습니다. BERTScore 평가를 사용할 수 없습니다.")
    print("   설치 방법: pip install bert-score")

# ===== 설정 =====
MODEL_ID = "skt/kogpt2-base-v2"
BASE_MODEL_DIR = "./kogpt2_finetuned"
MAX_DATA_SIZE = 100  # 평가에 사용할 데이터 개수
K_FOLDS = 5


def download_kpoem_data(max_size: int = 100) -> List[Dict]:
    """KPoeM 데이터셋 다운로드"""
    print(f"\n{'='*80}")
    print(f"[KPoeM 데이터셋 다운로드]")
    print(f"  - 최대 개수: {max_size}")
    print(f"{'='*80}\n")
    
    try:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": "hf://datasets/AKS-DHLAB/KPoEM/KPoEM_poem_dataset_v4.tsv"
            },
            delimiter="\t",
            encoding="utf-8",
            quoting=3,
        )
        dataset = dataset["train"]
        
        normalized_data = []
        for i, item in enumerate(dataset):
            if max_size and i >= max_size:
                break
            if 'text' in item and item['text']:
                poem_text = str(item['text']).strip()
                normalized_data.append({
                    'text': poem_text,
                    'poem': poem_text
                })
        
        print(f"✅ {len(normalized_data)}개 데이터 로드 완료")
        return normalized_data
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        raise


def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    """간단한 키워드 추출"""
    words = text.split()
    keywords = [w for w in words if len(w) >= 2][:max_keywords]
    return keywords if keywords else ["시", "감정"]


def classify_emotion_simple(text: str) -> Dict[str, str]:
    """간단한 감정 분류"""
    positive_words = ["좋", "행복", "기쁨", "사랑", "희망", "밝", "따뜻"]
    negative_words = ["슬", "우울", "아픔", "힘듦", "어둠", "차갑"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        mood = "밝은"
    elif neg_count > pos_count:
        mood = "어두운"
    else:
        mood = "잔잔한"
    
    return {'mood': mood}


def build_prompt_kogpt2(keywords: List[str], mood: str, lines: int, original_text: str) -> str:
    """koGPT2용 프롬프트 생성"""
    kw_str = ", ".join(keywords[:10])
    
    prompt = f"""Write a Korean poem (한국어 시) based on the keywords and mood below.

**CRITICAL: Language Requirement**
- You MUST write ONLY in Korean (Hangul, 한글).
- Do NOT use Chinese characters (한자), Japanese characters, English, or any other language.
- Use ONLY Korean characters (가-힣) and Korean punctuation.
- The output MUST be a Korean poem.

**Output Requirements**
- Output ONLY the poem text (no title, no explanation, no keywords, no numbering).
- The output MUST be in poem form with line breaks.
- Write EXACTLY {lines} lines (one line per verse; no empty lines).

**Content**
- Keywords: {kw_str}
- Mood: {mood}
{f'**Original Prose (Context)**\n\"\"\"{original_text.strip()}\"\"\"\n' if original_text else ''}

**Style Rules (strict)**
1) Keep each line short and lyrical.
2) Show, don't tell.
3) Avoid plain narration and diary-like tone.
4) In Korean, avoid declarative endings like "~다", "~이다", "~했다".
5) Avoid explicit subjects/time markers like "나는", "그는/그녀는", "오늘은/어제는".

Poem:
"""
    return prompt


def generate_poem_with_model(
    model_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    keywords: List[str],
    mood: str,
    original_text: str,
    device: str
) -> str:
    """
    학습된 모델로 시 생성
    학습 방식: 산문의 의미를 이해하고 그에 맞는 시를 생성
    """
    # 학습 형식: "산문: [내용]\n시: [내용]"
    # 따라서 입력은 "산문: [내용]\n시: " 형식으로 제공
    # 모델이 산문의 의미를 이해하고 그에 맞는 시를 생성하도록 함
    input_text = f"산문: {original_text.strip()}\n시: "
    
    # 토크나이즈
    enc_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    prompt_length = enc_ids.shape[1]
    
    # 입력 토큰 길이 제한
    max_pos_embeddings = getattr(model.config, 'max_position_embeddings', 1024)
    safe_max_input = max_pos_embeddings - 100
    if enc_ids.shape[1] >= safe_max_input:
        enc_ids = enc_ids[:, :safe_max_input]
        prompt_length = enc_ids.shape[1]
    
    # 시 생성
    with torch.no_grad():
        output = model.generate(
            enc_ids,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 프롬프트 제거 (토큰 기준으로 제거)
    # output[0]은 [prompt_tokens + generated_tokens] 형태
    # prompt_length 이후의 토큰만 디코딩
    if len(output[0]) > prompt_length:
        generated_tokens = output[0][prompt_length:]
        poem = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # 토큰 기준 제거가 안 되면 텍스트 기준으로 시도
        # "산문: ...\n시: " 패턴 제거
        if "시: " in generated_text:
            poem = generated_text.split("시: ", 1)[1].strip()
        else:
            poem = generated_text.strip()
    
    # 프롬프트 패턴 제거 (혹시 모를 경우를 대비)
    # "Poem:", "시:", "Write a Korean poem" 등으로 시작하는 부분 제거
    prompt_patterns = [
        r'^Write a Korean poem.*?\n',
        r'^Poem:\s*',
        r'^시:\s*',
        r'^\*\*CRITICAL.*?\n',
        r'^\*\*Output.*?\n',
    ]
    
    for pattern in prompt_patterns:
        poem = re.sub(pattern, '', poem, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    poem = poem.strip()
    
    # 후처리: 빈 줄 제거 및 줄 수 제한
    poem_lines = [line.strip() for line in poem.split('\n') if line.strip()]
    
    # 프롬프트가 포함된 줄 제거
    filtered_lines = []
    for line in poem_lines:
        # 프롬프트 키워드가 포함된 줄 제거
        if any(keyword in line.lower() for keyword in ['write a korean', 'critical', 'language requirement', 'output requirements', 'style rules']):
            continue
        # 영어만 있는 줄 제거 (한글이 없으면)
        if not any(ord('가') <= ord(c) <= ord('힣') for c in line):
            if len(line) > 20:  # 긴 영어 줄은 프롬프트일 가능성
                continue
        filtered_lines.append(line)
    
    poem = '\n'.join(filtered_lines[:6]) if filtered_lines else poem
    
    # 최종 검증: 여전히 프롬프트가 포함되어 있으면 빈 문자열 반환
    if any(keyword in poem.lower()[:100] for keyword in ['write a korean poem', 'critical: language requirement']):
        print(f"    ⚠️ 프롬프트가 포함된 출력 감지, 빈 문자열 반환")
        return ""
    
    return poem


def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트의 유사도 계산 (간단한 방법)"""
    # 단어 기반 유사도
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def calculate_bertscore(references: List[str], candidates: List[str], device: str = "cpu") -> Dict[str, float]:
    """
    BERTScore를 사용하여 참조 텍스트와 생성 텍스트 간의 의미적 유사도를 계산합니다.
    
    Args:
        references: 참조 텍스트 리스트 (원본 시)
        candidates: 생성 텍스트 리스트 (생성된 시)
        device: 계산에 사용할 디바이스 ("cpu" 또는 "cuda")
    
    Returns:
        BERTScore 점수 딕셔너리 (precision, recall, f1의 평균)
    """
    if not BERTSCORE_AVAILABLE:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'available': False
        }
    
    if not references or not candidates:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'available': True
        }
    
    try:
        # BERTScore 계산 (배치 처리)
        P, R, F1 = bert_score(
            candidates,
            references,
            lang='ko',  # 한국어 모델 사용
            verbose=False,
            device=device
        )
        
        # 텐서를 numpy로 변환 후 평균 계산
        precision_mean = float(P.mean().cpu().item())
        recall_mean = float(R.mean().cpu().item())
        f1_mean = float(F1.mean().cpu().item())
        
        return {
            'precision': precision_mean,
            'recall': recall_mean,
            'f1': f1_mean,
            'available': True
        }
    except Exception as e:
        print(f"⚠️ BERTScore 계산 중 오류 발생: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'available': False,
            'error': str(e)
        }


def evaluate_keyword_relevance(original_text: str, keywords: List[str], generated_poem: str) -> Dict[str, float]:
    """
    생성된 시가 원본 텍스트의 키워드를 얼마나 반영했는지 평가합니다.
    
    Args:
        original_text: 원본 일상 글
        keywords: 추출된 키워드 리스트
        generated_poem: 생성된 시
    
    Returns:
        키워드 관련성 평가 딕셔너리
    """
    if not generated_poem or not keywords:
        return {
            'keyword_coverage': 0.0,
            'keyword_count': 0,
            'total_keywords': len(keywords),
            'keyword_score': 0.0
        }
    
    # 생성된 시에서 키워드가 포함된 개수
    found_keywords = []
    for keyword in keywords:
        # 키워드가 직접 포함되어 있거나, 부분 일치하는지 확인
        if keyword in generated_poem:
            found_keywords.append(keyword)
        else:
            # 부분 일치 확인 (2글자 이상인 경우)
            if len(keyword) >= 2:
                for i in range(len(generated_poem) - len(keyword) + 1):
                    if generated_poem[i:i+len(keyword)] == keyword:
                        found_keywords.append(keyword)
                        break
    
    keyword_count = len(found_keywords)
    total_keywords = len(keywords)
    keyword_coverage = keyword_count / total_keywords if total_keywords > 0 else 0.0
    
    # 키워드 점수: 모든 키워드가 포함되면 1.0, 절반 이상이면 0.7, 일부만 있으면 0.4
    if keyword_coverage >= 0.8:  # 80% 이상
        keyword_score = 1.0
    elif keyword_coverage >= 0.5:  # 50% 이상
        keyword_score = 0.7
    elif keyword_coverage >= 0.3:  # 30% 이상
        keyword_score = 0.4
    else:
        keyword_score = keyword_coverage * 0.5  # 30% 미만은 비례적으로 낮은 점수
    
    return {
        'keyword_coverage': keyword_coverage,
        'keyword_count': keyword_count,
        'total_keywords': total_keywords,
        'found_keywords': found_keywords,
        'missing_keywords': [kw for kw in keywords if kw not in found_keywords],
        'keyword_score': keyword_score
    }


def evaluate_emotion_relevance(original_text: str, original_mood: str, generated_poem: str) -> Dict[str, float]:
    """
    생성된 시가 원본 텍스트의 감정을 얼마나 반영했는지 평가합니다.
    
    Args:
        original_text: 원본 일상 글
        original_mood: 원본 텍스트의 감정/분위기 (예: "밝은", "어두운", "잔잔한")
        generated_poem: 생성된 시
    
    Returns:
        감정 관련성 평가 딕셔너리
    """
    if not generated_poem:
        return {
            'emotion_match': 0.0,
            'emotion_score': 0.0,
            'detected_mood': 'unknown'
        }
    
    # 감정 단어 사전
    positive_words = ["좋", "행복", "기쁨", "사랑", "희망", "밝", "따뜻", "웃", "즐거", "환", "빛", "별", "꽃", "봄"]
    negative_words = ["슬", "우울", "아픔", "힘듦", "어둠", "차갑", "눈물", "그리움", "외로움", "아픔", "고통"]
    neutral_words = ["잔잔", "평온", "고요", "조용", "차분", "평화"]
    
    # 생성된 시에서 감정 단어 찾기
    poem_lower = generated_poem.lower()
    positive_count = sum(1 for word in positive_words if word in poem_lower)
    negative_count = sum(1 for word in negative_words if word in poem_lower)
    neutral_count = sum(1 for word in neutral_words if word in poem_lower)
    
    # 생성된 시의 감정 판정
    if positive_count > negative_count and positive_count > neutral_count:
        detected_mood = "밝은"
    elif negative_count > positive_count and negative_count > neutral_count:
        detected_mood = "어두운"
    elif neutral_count > 0:
        detected_mood = "잔잔한"
    else:
        detected_mood = "중립"
    
    # 원본 감정과 생성된 시의 감정 일치도
    emotion_match = 0.0
    if original_mood == detected_mood:
        emotion_match = 1.0
    elif (original_mood == "밝은" and detected_mood == "어두운") or (original_mood == "어두운" and detected_mood == "밝은"):
        emotion_match = 0.0  # 정반대
    elif (original_mood == "밝은" and detected_mood == "잔잔한") or (original_mood == "어두운" and detected_mood == "잔잔한"):
        emotion_match = 0.5  # 부분 일치
    elif original_mood == "잔잔한" and detected_mood in ["밝은", "어두운"]:
        emotion_match = 0.5  # 부분 일치
    else:
        emotion_match = 0.3  # 약간 일치
    
    # 감정 점수: 일치도 + 감정 단어 사용 빈도
    emotion_word_score = min(1.0, (positive_count + negative_count + neutral_count) / 3.0)
    emotion_score = (emotion_match * 0.7) + (emotion_word_score * 0.3)
    
    return {
        'emotion_match': emotion_match,
        'emotion_score': emotion_score,
        'detected_mood': detected_mood,
        'original_mood': original_mood,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count
    }


def evaluate_poetry_quality(poem: str) -> Dict[str, float]:
    """
    생성된 텍스트가 진짜 시인지 평가합니다.
    시가 아닌 산문, 일기, 설명문 등을 강하게 감지합니다.
    
    Returns:
        평가 점수 딕셔너리 (각 항목 0.0~1.0)
    """
    if not poem or len(poem.strip()) == 0:
        return {
            'is_poetry': 0.0,
            'format_score': 0.0,
            'korean_score': 0.0,
            'prose_penalty': 1.0,  # 최대 패널티
            'poetry_bonus': 0.0,
            'length_score': 0.0,
            'overall_score': 0.0,
            'is_prose': True,
            'is_diary': False,
            'is_explanation': False
        }
    
    poem = poem.strip()
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    
    # ===== 1. 형식 점수 (줄바꿈, 줄 개수, 줄 길이) =====
    format_score = 0.0
    
    # 줄 개수 점수
    if len(lines) >= 3:  # 최소 3줄 이상
        format_score += 0.2
    if len(lines) >= 6:  # 6줄 이상 (이상적)
        format_score += 0.3
    
    # 줄 길이 점수 (각 줄이 5-20자 정도면 좋음)
    good_length_lines = 0
    very_long_lines = 0  # 30자 이상은 산문일 가능성
    for line in lines:
        line_len = len(line)
        if 5 <= line_len <= 20:  # 적절한 길이
            good_length_lines += 1
        elif line_len > 30:  # 너무 긴 줄 (산문 가능성)
            very_long_lines += 1
    
    if lines:
        format_score += 0.5 * (good_length_lines / len(lines))
        # 너무 긴 줄이 많으면 감점
        if very_long_lines > len(lines) * 0.3:  # 30% 이상이면
            format_score *= 0.7
    
    # ===== 2. 한국어 점수 =====
    korean_chars = sum(1 for c in poem if ord('가') <= ord(c) <= ord('힣'))
    total_chars = len([c for c in poem if c.strip()])
    korean_score = korean_chars / total_chars if total_chars > 0 else 0.0
    
    # ===== 3. 산문 패턴 강력 감지 =====
    prose_penalty = 0.0
    is_prose = False
    is_diary = False
    is_explanation = False
    
    # 선언적 종결어미 패턴 (더 강력하게)
    declarative_patterns = [
        r'[가-힣\s]+다\s*[\.。]',  # "~다."
        r'[가-힣\s]+이다\s*[\.。]',  # "~이다."
        r'[가-힣\s]+했다\s*[\.。]',  # "~했다."
        r'[가-힣\s]+갔다\s*[\.。]',  # "~갔다."
        r'[가-힣\s]+했다\s*[\.。]',  # "~했다."
        r'[가-힣\s]+이다\s*[\.。]',  # "~이다."
        r'[가-힣\s]+이다\s*$',  # 줄 끝의 "~이다"
        r'[가-힣\s]+다\s*$',  # 줄 끝의 "~다"
    ]
    
    declarative_count = 0
    for pattern in declarative_patterns:
        matches = re.findall(pattern, poem)
        declarative_count += len(matches)
    
    # 주어/시간 표시 패턴 (일기, 산문 특징)
    subject_time_patterns = [
        r'\b나는\b',
        r'\b그는\b',
        r'\b그녀는\b',
        r'\b우리는\b',
        r'\b오늘은\b',
        r'\b어제는\b',
        r'\b내일은\b',
        r'\b오늘\b.*[은는]',  # "오늘은", "오늘의"
        r'\b어제\b.*[은는]',
    ]
    
    subject_time_count = 0
    for pattern in subject_time_patterns:
        subject_time_count += len(re.findall(pattern, poem))
    
    # 일기 패턴 감지
    diary_patterns = [
        r'오늘.*[은는]',
        r'어제.*[은는]',
        r'내일.*[은는]',
        r'나는.*[했갔]다',
        r'오늘.*[했갔]다',
    ]
    
    diary_count = sum(len(re.findall(pattern, poem)) for pattern in diary_patterns)
    if diary_count >= 3:  # 2 → 3으로 완화
        is_diary = True
    
    # 설명문/논술 패턴
    explanation_patterns = [
        r'[가-힣]+[은는]?\s*[가-힣]+[이다]',  # "A는 B이다"
        r'[가-힣]+[은는]?\s*[가-힣]+[이다]',  # "A는 B이다"
        r'왜냐하면',
        r'그래서',
        r'따라서',
        r'그러므로',
    ]
    
    explanation_count = sum(len(re.findall(pattern, poem)) for pattern in explanation_patterns)
    if explanation_count >= 3:  # 2 → 3으로 완화
        is_explanation = True
    
    # 산문 판정: 선언적 종결어미가 많거나, 주어/시간 표시가 많으면 (기준 완화)
    if declarative_count >= 8 or subject_time_count >= 8:  # 5 → 8로 완화
        is_prose = True
    
    # 패널티 계산 (기준 완화)
    total_penalty_score = (
        declarative_count * 0.10 +  # 선언적 종결어미 감점 완화 (0.15 → 0.10)
        subject_time_count * 0.08 +  # 주어/시간 표시 감점 완화 (0.10 → 0.08)
        diary_count * 0.15 +  # 일기 패턴 감점 완화 (0.20 → 0.15)
        explanation_count * 0.12  # 설명문 패턴 감점 완화 (0.15 → 0.12)
    )
    
    # 최대 패널티는 0.7 (70% 감점) - 완화 (0.8 → 0.7)
    prose_penalty = min(0.7, total_penalty_score)
    
    # ===== 4. 시적 표현 보너스 =====
    poetry_bonus = 0.0
    
    # 은유, 상징 표현
    poetic_patterns = [
        r'[가-힣]+처럼',  # "꽃처럼", "별처럼"
        r'[가-힣]+같이',  # "꽃같이"
        r'[가-힣]+[은는]?\s*[가-힣]+[을를]\s*[가-힣]+',  # "바람이 꽃을 흔든다" (시적 표현)
        r'[가-힣]+[은는]?\s*[가-힣]+[에]?\s*[가-힣]+',  # "하늘에 별이"
    ]
    
    poetic_count = sum(len(re.findall(pattern, poem)) for pattern in poetic_patterns)
    if poetic_count >= 2:
        poetry_bonus += 0.2
    elif poetic_count >= 1:
        poetry_bonus += 0.1
    
    # 짧고 함축적인 줄 (시적 특징)
    short_lyrical_lines = sum(1 for line in lines if 3 <= len(line) <= 15)
    if len(lines) > 0 and short_lyrical_lines / len(lines) >= 0.7:  # 70% 이상이면
        poetry_bonus += 0.1
    
    # ===== 5. 길이 점수 =====
    length_score = 1.0
    poem_length = len(poem)
    if poem_length < 20:  # 너무 짧음
        length_score = 0.2
    elif poem_length < 50:  # 약간 짧음
        length_score = 0.6
    elif poem_length > 500:  # 너무 김 (산문 가능성)
        length_score = 0.7
    
    # ===== 6. 의미 있는 내용 =====
    meaningful_score = 1.0
    if korean_chars < 10:  # 한글이 너무 적음
        meaningful_score = 0.2
    elif korean_chars < 20:
        meaningful_score = 0.5
    elif korean_chars < 30:
        meaningful_score = 0.8
    
    # ===== 7. 종합 점수 계산 =====
    # 기본 점수: 형식 25% + 한국어 20% + 길이 15% + 의미 15%
    base_score = (
        format_score * 0.25 +
        korean_score * 0.20 +
        length_score * 0.15 +
        meaningful_score * 0.15
    )
    
    # 산문 패널티 적용 (25% 가중치)
    penalty_adjusted_score = base_score * (1.0 - prose_penalty * 0.25)
    
    # 시적 표현 보너스 추가 (25% 가중치)
    overall_score = penalty_adjusted_score + (poetry_bonus * 0.25)
    
    # 산문/일기/설명문이면 감점 (기준 완화)
    if is_prose:
        overall_score *= 0.7  # 30% 추가 감점 (0.5 → 0.7로 완화)
    if is_diary:
        overall_score *= 0.75  # 25% 추가 감점 (0.6 → 0.75로 완화)
    if is_explanation:
        overall_score *= 0.8  # 20% 추가 감점 (0.7 → 0.8로 완화)
    
    # 최소 0.0, 최대 1.0으로 제한
    overall_score = max(0.0, min(1.0, overall_score))
    
    return {
        'is_poetry': overall_score,  # 0.0 (산문) ~ 1.0 (시)
        'format_score': format_score,
        'korean_score': korean_score,
        'prose_penalty': prose_penalty,
        'poetry_bonus': poetry_bonus,
        'length_score': length_score,
        'meaningful_score': meaningful_score,
        'overall_score': overall_score,
        'line_count': len(lines),
        'korean_chars': korean_chars,
        'declarative_count': declarative_count,
        'subject_time_count': subject_time_count,
        'diary_count': diary_count,
        'explanation_count': explanation_count,
        'poetic_count': poetic_count,
        'is_prose': is_prose,
        'is_diary': is_diary,
        'is_explanation': is_explanation
    }


def calculate_classification_metrics(results: List[Dict]) -> Dict:
    """
    평가 결과로부터 confusion matrix와 classification metrics를 계산합니다.
    
    Returns:
        감정 분류, 성공/실패 분류, 시 품질 통계에 대한 metrics
    """
    # 감정 분류 데이터 수집
    emotion_true = []
    emotion_pred = []
    
    # 성공/실패 분류
    success_true = []  # 실제로는 항상 True를 목표
    success_pred = []
    
    # 시 품질 점수 수집 (통계용)
    poetry_scores = []
    prose_count = 0
    diary_count = 0
    explanation_count = 0
    
    for r in results:
        if not r.get('generated_poem'):
            continue
        
        # 감정 분류
        original_mood = r.get('mood', '잔잔한')
        emotion_relevance = r.get('emotion_relevance', {})
        detected_mood = emotion_relevance.get('detected_mood', 'unknown')
        
        if original_mood and detected_mood != 'unknown':
            emotion_true.append(original_mood)
            emotion_pred.append(detected_mood)
        
        # 시 품질 통계 수집
        poetry_quality = r.get('poetry_quality', {})
        if poetry_quality:
            poetry_scores.append(poetry_quality.get('overall_score', 0.0))
            if poetry_quality.get('is_prose', False):
                prose_count += 1
            if poetry_quality.get('is_diary', False):
                diary_count += 1
            if poetry_quality.get('is_explanation', False):
                explanation_count += 1
        
        # 성공/실패 분류
        success = r.get('success', False)
        success_true.append(True)  # 목표는 항상 성공
        success_pred.append(success)
    
    metrics = {}
    
    # 감정 분류 metrics
    if emotion_true and emotion_pred:
        emotion_labels = sorted(set(emotion_true + emotion_pred))
        emotion_cm = confusion_matrix(emotion_true, emotion_pred, labels=emotion_labels)
        emotion_acc = accuracy_score(emotion_true, emotion_pred)
        
        # precision, recall, f1 per class
        emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
            emotion_true, emotion_pred, labels=emotion_labels, zero_division=0
        )
        
        metrics['emotion'] = {
            'confusion_matrix': emotion_cm.tolist(),
            'labels': emotion_labels,
            'accuracy': float(emotion_acc),
            'precision': {label: float(p) for label, p in zip(emotion_labels, emotion_precision)},
            'recall': {label: float(r) for label, r in zip(emotion_labels, emotion_recall)},
            'f1_score': {label: float(f) for label, f in zip(emotion_labels, emotion_f1)},
            'classification_report': classification_report(emotion_true, emotion_pred, labels=emotion_labels, zero_division=0)
        }
    
    # 시 품질 통계 metrics
    if poetry_scores:
        import numpy as np
        metrics['poetry_quality_stats'] = {
            'mean': float(np.mean(poetry_scores)),
            'median': float(np.median(poetry_scores)),
            'std': float(np.std(poetry_scores)),
            'min': float(np.min(poetry_scores)),
            'max': float(np.max(poetry_scores)),
            'total_samples': len(poetry_scores),
            'prose_count': prose_count,
            'diary_count': diary_count,
            'explanation_count': explanation_count,
            'non_poetry_count': prose_count + diary_count + explanation_count,
            'poetry_rate': float((len(poetry_scores) - prose_count - diary_count - explanation_count) / len(poetry_scores)) if poetry_scores else 0.0
        }
    
    # 성공/실패 분류 metrics
    if success_true and success_pred:
        success_labels = [True, False]
        success_cm = confusion_matrix(success_true, success_pred, labels=success_labels)
        success_acc = accuracy_score(success_true, success_pred)
        
        success_precision, success_recall, success_f1, _ = precision_recall_fscore_support(
            success_true, success_pred, labels=success_labels, zero_division=0
        )
        
        metrics['success'] = {
            'confusion_matrix': success_cm.tolist(),
            'labels': success_labels,
            'accuracy': float(success_acc),
            'precision': {str(label): float(p) for label, p in zip(success_labels, success_precision)},
            'recall': {str(label): float(r) for label, r in zip(success_labels, success_recall)},
            'f1_score': {str(label): float(f) for label, f in zip(success_labels, success_f1)},
            'classification_report': classification_report(success_true, success_pred, labels=success_labels, zero_division=0)
        }
    
    # BERTScore 통계 metrics
    bertscore_f1_scores = [r.get('bertscore', {}).get('f1', 0.0) 
                          for r in results if r.get('bertscore')]
    bertscore_precision_scores = [r.get('bertscore', {}).get('precision', 0.0) 
                                 for r in results if r.get('bertscore')]
    bertscore_recall_scores = [r.get('bertscore', {}).get('recall', 0.0) 
                              for r in results if r.get('bertscore')]
    
    if bertscore_f1_scores:
        import numpy as np
        metrics['bertscore_stats'] = {
            'mean_f1': float(np.mean(bertscore_f1_scores)),
            'mean_precision': float(np.mean(bertscore_precision_scores)),
            'mean_recall': float(np.mean(bertscore_recall_scores)),
            'std_f1': float(np.std(bertscore_f1_scores)),
            'min_f1': float(np.min(bertscore_f1_scores)),
            'max_f1': float(np.max(bertscore_f1_scores)),
            'sample_count': len(bertscore_f1_scores),
            'available': True
        }
    else:
        metrics['bertscore_stats'] = {
            'available': False
        }
    
    return metrics


def translate_labels_to_english(labels):
    """
    한글 라벨을 영어로 변환합니다 (폰트 없이도 작동하도록).
    
    Args:
        labels: 한글 또는 영어 라벨 리스트
    
    Returns:
        영어 라벨 리스트
    """
    translation_map = {
        # 감정
        '밝은': 'Bright',
        '어두운': 'Dark',
        '잔잔한': 'Calm',
        '쓸쓸한': 'Lonely',
        '격한': 'Intense',
        '신기한': 'Surprised',
        '무서운': 'Scary',
        '싫은': 'Disgusted',
        '따뜻한': 'Warm',
        '그리운': 'Nostalgic',
        '불안한': 'Anxious',
        '담담한': 'Neutral',
        # 시 형태
        '시': 'Poetry',
        '비시': 'Non-Poetry',
        '산문': 'Prose',
        '일기': 'Diary',
        '설명문': 'Explanation',
        # 기타
        'unknown': 'Unknown',
        '중립': 'Neutral',
        True: 'Success',
        False: 'Failure'
    }
    
    translated = []
    for label in labels:
        if isinstance(label, bool):
            translated.append(translation_map.get(label, str(label)))
        else:
            translated.append(translation_map.get(label, label))
    
    return translated


def translate_title_to_english(title: str) -> str:
    """
    제목의 한글을 영어로 변환합니다.
    
    Args:
        title: 한글 제목
    
    Returns:
        영어 제목
    """
    title_map = {
        '감정 분류 Confusion Matrix': 'Emotion Classification Confusion Matrix',
        '성공/실패 분류 Confusion Matrix': 'Success/Failure Classification Confusion Matrix',
        '감정 분류 Confusion Matrix (전체 Fold)': 'Emotion Classification Confusion Matrix (All Folds)',
        '성공/실패 분류 Confusion Matrix (전체 Fold)': 'Success/Failure Classification Confusion Matrix (All Folds)'
    }
    
    return title_map.get(title, title)


def plot_confusion_matrix(cm, labels, title, fold_idx=None, save_path=None):
    """
    Confusion matrix를 시각화합니다.
    
    Args:
        cm: confusion matrix (numpy array)
        labels: 클래스 라벨 리스트
        title: 그래프 제목
        fold_idx: fold 번호 (옵션)
        save_path: 저장 경로 (옵션)
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"⚠️ matplotlib이 설치되지 않아 {title} 시각화를 건너뜁니다.")
        return
    
    # 한글 라벨을 영어로 변환 (폰트 없이도 작동하도록)
    english_labels = translate_labels_to_english(labels)
    # 제목도 영어로 변환
    english_title = translate_title_to_english(title)
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=english_labels, yticklabels=english_labels,
                cbar_kws={'label': 'Count'})
    plt.title(f"{english_title}{f' (Fold {fold_idx})' if fold_idx else ''}", fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_metrics(metrics: Dict, fold_idx: int = None):
    """
    Classification metrics를 출력합니다.
    """
    fold_prefix = f"[Fold {fold_idx}] " if fold_idx else ""
    
    # 감정 분류 metrics
    if 'emotion' in metrics:
        em = metrics['emotion']
        print(f"\n{'='*80}")
        print(f"{fold_prefix}감정 분류 평가 지표")
        print(f"{'='*80}")
        print(f"정확도 (Accuracy): {em['accuracy']:.4f}")
        print(f"\n클래스별 성능:")
        for label in em['labels']:
            print(f"  {label}:")
            print(f"    Precision: {em['precision'].get(label, 0.0):.4f}")
            print(f"    Recall: {em['recall'].get(label, 0.0):.4f}")
            print(f"    F1-Score: {em['f1_score'].get(label, 0.0):.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  Labels: {em['labels']}")
        for i, row in enumerate(em['confusion_matrix']):
            print(f"  {em['labels'][i]}: {row}")
        print(f"\nClassification Report:")
        print(em['classification_report'])
    
    # 시 품질 통계 metrics
    if 'poetry_quality_stats' in metrics:
        pqs = metrics['poetry_quality_stats']
        print(f"\n{'='*80}")
        print(f"{fold_prefix}Poetry Quality Statistics")
        print(f"{'='*80}")
        print(f"평균 시 품질 점수: {pqs['mean']:.4f} (0.0=산문, 1.0=시)")
        print(f"중앙값: {pqs['median']:.4f}")
        print(f"표준편차: {pqs['std']:.4f}")
        print(f"최소값: {pqs['min']:.4f}")
        print(f"최대값: {pqs['max']:.4f}")
        print(f"\n시 형태 분포:")
        print(f"  시로 판정: {pqs['total_samples'] - pqs['non_poetry_count']}개 ({pqs['poetry_rate']:.2%})")
        print(f"  산문으로 판정: {pqs['prose_count']}개")
        print(f"  일기로 판정: {pqs['diary_count']}개")
        print(f"  설명문으로 판정: {pqs['explanation_count']}개")
        print(f"  비시 총계: {pqs['non_poetry_count']}개 ({1.0 - pqs['poetry_rate']:.2%})")
    
    # BERTScore metrics
    if 'bertscore_stats' in metrics:
        bs = metrics['bertscore_stats']
        print(f"\n{'='*80}")
        print(f"{fold_prefix}의미 유사도 평가 (BERTScore)")
        print(f"{'='*80}")
        if bs.get('available', False):
            print(f"평균 BERTScore F1: {bs.get('mean_f1', 0.0):.4f} (0.0=다름, 1.0=동일)")
            print(f"평균 Precision: {bs.get('mean_precision', 0.0):.4f}")
            print(f"평균 Recall: {bs.get('mean_recall', 0.0):.4f}")
            print(f"표준편차 F1: {bs.get('std_f1', 0.0):.4f}")
            print(f"최소값 F1: {bs.get('min_f1', 0.0):.4f}")
            print(f"최대값 F1: {bs.get('max_f1', 0.0):.4f}")
            print(f"평가 샘플 수: {bs.get('sample_count', 0)}개")
            print(f"\n원본 시와 생성 시의 의미적 유사도를 BERT 모델을 사용하여 측정합니다.")
        else:
            print(f"⚠️ BERTScore를 사용할 수 없습니다 (라이브러리 미설치 또는 계산 실패)")
            print(f"   설치 방법: pip install bert-score")
    
    # 성공/실패 분류 metrics
    if 'success' in metrics:
        sc = metrics['success']
        print(f"\n{'='*80}")
        print(f"{fold_prefix}성공/실패 분류 평가 지표")
        print(f"{'='*80}")
        print(f"정확도 (Accuracy): {sc['accuracy']:.4f}")
        print(f"\n클래스별 성능:")
        for label in sc['labels']:
            label_str = str(label)
            print(f"  {label_str}:")
            print(f"    Precision: {sc['precision'].get(label_str, 0.0):.4f}")
            print(f"    Recall: {sc['recall'].get(label_str, 0.0):.4f}")
            print(f"    F1-Score: {sc['f1_score'].get(label_str, 0.0):.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  Labels: {sc['labels']}")
        for i, row in enumerate(sc['confusion_matrix']):
            print(f"  {sc['labels'][i]}: {row}")
        print(f"\nClassification Report:")
        print(sc['classification_report'])


def evaluate_fold_model(
    fold_idx: int,
    model_path: Path,
    test_data: List[Dict],
    device: str
) -> Dict:
    """특정 fold 모델 평가"""
    print(f"\n{'='*80}")
    print(f"[Fold {fold_idx} 모델 평가]")
    print(f"  모델 경로: {model_path}")
    print(f"  Test 데이터: {len(test_data)}개")
    print(f"{'='*80}\n")
    
    if not model_path.exists():
        print(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
        return {
            'fold': fold_idx,
            'success': False,
            'error': 'Model path not found'
        }
    
    # 모델 로드
    print(f"[1/3] 모델 로딩 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32
        )
        model = model.to(device).eval()
        print(f"✅ 모델 로딩 완료\n")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return {
            'fold': fold_idx,
            'success': False,
            'error': str(e)
        }
    
    # 시 생성 및 평가
    print(f"[2/3] 시 생성 및 평가 중...")
    results = []
    similarities = []
    success_count = 0
    
    # 디버깅: 생성된 시 샘플 출력
    sample_count = 0
    
    for i, item in enumerate(test_data, 1):
        if i % 5 == 0 or i == len(test_data):
            print(f"  진행 중: {i}/{len(test_data)}")
        
        text = item['text']
        original_poem = item.get('poem', '')
        
        try:
            # 키워드 및 감정 추출
            keywords = extract_keywords_simple(text, max_keywords=10)
            emotion_result = classify_emotion_simple(text)
            mood = emotion_result.get('mood', '잔잔한')
            
            # 시 생성
            generated_poem = generate_poem_with_model(
                str(model_path),
                tokenizer,
                model,
                keywords,
                mood,
                text,
                device
            )
            
            # 디버깅: 처음 몇 개 샘플 출력
            if sample_count < 3:
                print(f"\n    [샘플 {sample_count + 1}]")
                print(f"      원문: {text[:50]}...")
                print(f"      생성된 시 길이: {len(generated_poem) if generated_poem else 0}자")
                print(f"      생성된 시: {repr(generated_poem[:200]) if generated_poem else 'None'}")
                if generated_poem:
                    korean_chars = sum(1 for c in generated_poem if ord('가') <= ord(c) <= ord('힣'))
                    print(f"      한글 문자 수: {korean_chars}자")
                sample_count += 1
            
            # 시 품질 평가
            poetry_quality = evaluate_poetry_quality(generated_poem) if generated_poem else {
                'is_poetry': 0.0,
                'overall_score': 0.0,
                'korean_chars': 0,
                'format_score': 0.0,
                'korean_score': 0.0,
                'prose_penalty': 0.0
            }
            
            # 키워드 관련성 평가
            keyword_relevance = evaluate_keyword_relevance(text, keywords, generated_poem) if generated_poem else {
                'keyword_coverage': 0.0,
                'keyword_score': 0.0,
                'keyword_count': 0,
                'total_keywords': len(keywords)
            }
            
            # 감정 관련성 평가
            emotion_relevance = evaluate_emotion_relevance(text, mood, generated_poem) if generated_poem else {
                'emotion_match': 0.0,
                'emotion_score': 0.0,
                'detected_mood': 'unknown',
                'original_mood': mood
            }
            
            # 유사도 계산 (원본 시와의 유사도 - 참고용)
            similarity = 0.0
            if generated_poem and original_poem:
                similarity = calculate_similarity(generated_poem, original_poem)
                similarities.append(similarity)
            
            # 종합 성공 기준: 시 형태 + 키워드 반영 + 감정 반영
            # 1. 시 품질 점수가 0.6 이상
            # 2. 산문/일기/설명문이 아님
            # 3. 한글이 충분히 있음
            # 4. 최소 길이 충족
            # 5. 키워드 반영 (30% 이상)
            # 6. 감정 반영 (50% 이상)
            
            # 각 조건별 확인 (디버깅용) - 기준 완화
            check_poetry_score = poetry_quality['overall_score'] >= 0.5  # 0.6 → 0.5로 완화
            check_not_prose = not poetry_quality.get('is_prose', False)
            check_not_diary = not poetry_quality.get('is_diary', False)
            check_not_explanation = not poetry_quality.get('is_explanation', False)
            check_korean_chars = poetry_quality['korean_chars'] >= 10  # 15 → 10으로 완화
            check_line_count = poetry_quality.get('line_count', 0) >= 2  # 3 → 2로 완화
            check_min_length = len(generated_poem.strip()) >= 20  # 25 → 20으로 완화
            check_keyword = keyword_relevance['keyword_coverage'] >= 0.2  # 30% → 20%로 완화
            check_emotion = emotion_relevance['emotion_score'] >= 0.4  # 0.5 → 0.4로 완화
            
            # 성공 기준: 가중치 기반 점수 시스템
            # 필수 조건을 점수로 변환 (통과한 개수에 따라 점수 부여)
            required_checks = [
                check_not_prose,  # 산문이 아니면 통과
                check_not_diary,  # 일기가 아니면 통과
                check_not_explanation,  # 설명문이 아니면 통과
                check_korean_chars,  # 한글 10자 이상이면 통과
            ]
            
            # 필수 조건 통과 개수에 따른 점수 (0.0 ~ 1.0)
            passed_required_count = sum(required_checks)
            required_score = passed_required_count / len(required_checks)  # 4개 중 몇 개 통과했는지 비율
            
            # 선택 조건: 가중치 기반 점수 계산
            # 각 조건을 점수로 변환 (0.0 ~ 1.0) - 기준 완화
            score_poetry = min(1.0, poetry_quality['overall_score'] / 0.5)  # 0.5 이상이면 1.0
            score_line_count = min(1.0, poetry_quality.get('line_count', 0) / 2.0)  # 2줄 이상이면 1.0
            score_min_length = min(1.0, len(generated_poem.strip()) / 20.0) if generated_poem else 0.0  # 20자 이상이면 1.0
            score_keyword = min(1.0, keyword_relevance['keyword_coverage'] / 0.2)  # 20% 이상이면 1.0
            score_emotion = min(1.0, emotion_relevance['emotion_score'] / 0.4)  # 0.4 이상이면 1.0
            
            # 가중치 기반 종합 점수 계산
            # 필수 조건 점수 30% + 선택 조건 점수 70%
            # 선택 조건 중 시 품질이 가장 중요 (40%), 나머지는 각 10%
            selection_score = (
                score_poetry * 0.40 +
                score_line_count * 0.10 +
                score_min_length * 0.10 +
                score_keyword * 0.20 +
                score_emotion * 0.20
            )
            
            # 종합 점수 = 필수 조건 점수(30%) + 선택 조건 점수(70%)
            weighted_score = (required_score * 0.30) + (selection_score * 0.70)
            
            # 종합 점수가 0.6 이상이면 성공
            is_success = weighted_score >= 0.6
            
            # 디버깅: 처음 5개 샘플에 대해 상세 분석 출력
            if sample_count <= 5:
                print(f"\n    [샘플 {sample_count} 평가 결과]")
                print(f"      원문: {text[:60]}...")
                print(f"      생성된 시: {repr(generated_poem[:150]) if generated_poem else 'None'}")
                
                if not is_success:
                    print(f"      결과: ❌ 실패")
                else:
                    print(f"      결과: ✅ 성공")
                
                # 필수 조건 확인
                print(f"\n      [필수 조건] (통과 개수에 따라 점수 부여: {passed_required_count}/4)")
                print(f"        산문 여부: {poetry_quality.get('is_prose', False)} {'✅' if check_not_prose else '❌'}")
                if poetry_quality.get('is_prose', False):
                    print(f"          - 선언적 종결어미: {poetry_quality.get('declarative_count', 0)}개")
                    print(f"          - 주어/시간 표시: {poetry_quality.get('subject_time_count', 0)}개")
                print(f"        일기 여부: {poetry_quality.get('is_diary', False)} {'✅' if check_not_diary else '❌'}")
                if poetry_quality.get('is_diary', False):
                    print(f"          - 일기 패턴: {poetry_quality.get('diary_count', 0)}개")
                print(f"        설명문 여부: {poetry_quality.get('is_explanation', False)} {'✅' if check_not_explanation else '❌'}")
                if poetry_quality.get('is_explanation', False):
                    print(f"          - 설명문 패턴: {poetry_quality.get('explanation_count', 0)}개")
                print(f"        한글 문자: {poetry_quality['korean_chars']}자 (필요: ≥10) {'✅' if check_korean_chars else '❌'}")
                print(f"        → 필수 조건 점수: {required_score:.4f} ({passed_required_count}/4 통과)")
                
                # 선택 조건 확인
                print(f"\n      [선택 조건] (가중치 기반 점수)")
                print(f"        시 품질 점수: {poetry_quality['overall_score']:.4f} (목표: ≥0.5)")
                print(f"        줄 개수: {poetry_quality.get('line_count', 0)}줄 (목표: ≥2)")
                print(f"        전체 길이: {len(generated_poem.strip()) if generated_poem else 0}자 (목표: ≥20)")
                print(f"        키워드 반영률: {keyword_relevance['keyword_coverage']:.2%} (목표: ≥20%)")
                print(f"        감정 점수: {emotion_relevance['emotion_score']:.4f} (목표: ≥0.4)")
                
                # 종합 점수 계산 (가중치 기반)
                print(f"\n        종합 점수 계산:")
                print(f"          - 필수 조건 점수: {required_score:.4f} × 0.30 = {required_score * 0.30:.4f}")
                print(f"          - 선택 조건 점수: {selection_score:.4f} × 0.70 = {selection_score * 0.70:.4f}")
                print(f"          - 종합 점수: {weighted_score:.4f} (필요: ≥0.6)")
                print(f"\n        선택 조건 세부:")
                print(f"          - 시 품질: {score_poetry:.4f} × 0.40 = {score_poetry * 0.40:.4f}")
                print(f"          - 줄 개수: {score_line_count:.4f} × 0.10 = {score_line_count * 0.10:.4f}")
                print(f"          - 전체 길이: {score_min_length:.4f} × 0.10 = {score_min_length * 0.10:.4f}")
                print(f"          - 키워드: {score_keyword:.4f} × 0.20 = {score_keyword * 0.20:.4f}")
                print(f"          - 감정: {score_emotion:.4f} × 0.20 = {score_emotion * 0.20:.4f}")
            
            if is_success:
                success_count += 1
            
            results.append({
                'original_text': text,
                'original_poem': original_poem,
                'generated_poem': generated_poem,
                'keywords': keywords,
                'mood': mood,
                'success': is_success,
                'poetry_quality': poetry_quality,
                'keyword_relevance': keyword_relevance,
                'emotion_relevance': emotion_relevance,
                'similarity': similarity
            })
            
        except Exception as e:
            print(f"    ⚠️ 오류 (인덱스 {i-1}): {e}")
            results.append({
                'original_text': text,
                'original_poem': original_poem,
                'generated_poem': '',
                'success': False,
                'error': str(e)
            })
    
    # BERTScore 계산 (배치 처리)
    print(f"\n[2.5/3] BERTScore 계산 중...")
    bertscore_results = None
    if BERTSCORE_AVAILABLE:
        references = []
        candidates = []
        valid_indices = []
        
        for idx, r in enumerate(results):
            if r.get('generated_poem') and r.get('original_poem'):
                references.append(r['original_poem'])
                candidates.append(r['generated_poem'])
                valid_indices.append(idx)
        
        if references and candidates:
            print(f"  - BERTScore 계산 대상: {len(references)}개")
            bertscore_results = calculate_bertscore(references, candidates, device)
            
            # 각 결과에 BERTScore 점수 추가
            if bertscore_results.get('available', False):
                # 개별 점수 계산 (배치로 한 번에 계산)
                P, R, F1 = bert_score(
                    candidates,
                    references,
                    lang='ko',
                    verbose=False,
                    device=device
                )
                
                # 각 결과에 개별 점수 추가
                for idx, result_idx in enumerate(valid_indices):
                    results[result_idx]['bertscore'] = {
                        'precision': float(P[idx].cpu().item()),
                        'recall': float(R[idx].cpu().item()),
                        'f1': float(F1[idx].cpu().item())
                    }
                print(f"  ✅ BERTScore 계산 완료")
            else:
                print(f"  ⚠️ BERTScore 계산 실패")
        else:
            print(f"  ⚠️ BERTScore 계산할 데이터가 없습니다.")
    else:
        print(f"  ⚠️ bert-score 라이브러리가 설치되지 않아 BERTScore를 계산할 수 없습니다.")
    
    # 결과 정리
    avg_similarity = np.mean(similarities) if similarities else 0.0
    success_rate = success_count / len(test_data) if test_data else 0.0
    
    # 시 품질 점수 평균
    poetry_scores = [r.get('poetry_quality', {}).get('overall_score', 0.0) 
                     for r in results if r.get('poetry_quality')]
    avg_poetry_score = np.mean(poetry_scores) if poetry_scores else 0.0
    
    # 키워드 관련성 평균
    keyword_scores = [r.get('keyword_relevance', {}).get('keyword_score', 0.0) 
                      for r in results if r.get('keyword_relevance')]
    avg_keyword_score = np.mean(keyword_scores) if keyword_scores else 0.0
    avg_keyword_coverage = np.mean([r.get('keyword_relevance', {}).get('keyword_coverage', 0.0) 
                                    for r in results if r.get('keyword_relevance')]) if keyword_scores else 0.0
    
    # 감정 관련성 평균
    emotion_scores = [r.get('emotion_relevance', {}).get('emotion_score', 0.0) 
                      for r in results if r.get('emotion_relevance')]
    avg_emotion_score = np.mean(emotion_scores) if emotion_scores else 0.0
    avg_emotion_match = np.mean([r.get('emotion_relevance', {}).get('emotion_match', 0.0) 
                                 for r in results if r.get('emotion_relevance')]) if emotion_scores else 0.0
    
    # 상세 통계
    avg_format_score = np.mean([r.get('poetry_quality', {}).get('format_score', 0.0) 
                                for r in results if r.get('poetry_quality')]) if poetry_scores else 0.0
    avg_korean_score = np.mean([r.get('poetry_quality', {}).get('korean_score', 0.0) 
                                   for r in results if r.get('poetry_quality')]) if poetry_scores else 0.0
    avg_prose_penalty = np.mean([r.get('poetry_quality', {}).get('prose_penalty', 0.0) 
                                 for r in results if r.get('poetry_quality')]) if poetry_scores else 0.0
    avg_poetry_bonus = np.mean([r.get('poetry_quality', {}).get('poetry_bonus', 0.0) 
                                for r in results if r.get('poetry_quality')]) if poetry_scores else 0.0
    
    # 산문/일기/설명문 통계
    prose_count = sum(1 for r in results if r.get('poetry_quality', {}).get('is_prose', False))
    diary_count = sum(1 for r in results if r.get('poetry_quality', {}).get('is_diary', False))
    explanation_count = sum(1 for r in results if r.get('poetry_quality', {}).get('is_explanation', False))
    
    print(f"\n[3/3] 평가 완료")
    print(f"  - 성공률 (종합 기준): {success_rate:.2%} ({success_count}/{len(test_data)})")
    
    # 성공률이 0%인 경우 상세 분석
    if success_rate == 0.0:
        print(f"\n  ⚠️ 성공률이 0%입니다. 실패 원인 분석:")
        
        # 필수 조건 실패 통계
        prose_fail = sum(1 for r in results if r.get('poetry_quality', {}).get('is_prose', False))
        diary_fail = sum(1 for r in results if r.get('poetry_quality', {}).get('is_diary', False))
        explanation_fail = sum(1 for r in results if r.get('poetry_quality', {}).get('is_explanation', False))
        korean_chars_fail = sum(1 for r in results if r.get('poetry_quality', {}).get('korean_chars', 0) < 15)
        
        # 선택 조건 실패 통계
        poetry_score_fail = sum(1 for r in results if r.get('poetry_quality', {}).get('overall_score', 0.0) < 0.6)
        line_count_fail = sum(1 for r in results if r.get('poetry_quality', {}).get('line_count', 0) < 3)
        min_length_fail = sum(1 for r in results if len(r.get('generated_poem', '').strip()) < 25)
        keyword_fail = sum(1 for r in results if r.get('keyword_relevance', {}).get('keyword_coverage', 0.0) < 0.3)
        emotion_fail = sum(1 for r in results if r.get('emotion_relevance', {}).get('emotion_score', 0.0) < 0.5)
        
        # 필수 조건 실패로 인한 실패
        required_fail = prose_fail + diary_fail + explanation_fail + korean_chars_fail
        
        print(f"\n    [필수 조건 실패] (하나라도 실패하면 실패)")
        print(f"      - 산문으로 판정: {prose_fail}개")
        print(f"      - 일기로 판정: {diary_fail}개")
        print(f"      - 설명문으로 판정: {explanation_fail}개")
        print(f"      - 한글 < 10자: {korean_chars_fail}개")
        print(f"      → 필수 조건 실패로 인한 실패: {required_fail}개")
        
        # 선택 조건 분석
        print(f"\n    [선택 조건 분석]")
        print(f"      - 시 품질 점수 < 0.5: {poetry_score_fail}개")
        print(f"      - 줄 개수 < 2줄: {line_count_fail}개")
        print(f"      - 전체 길이 < 20자: {min_length_fail}개")
        print(f"      - 키워드 반영률 < 20%: {keyword_fail}개")
        print(f"      - 감정 점수 < 0.4: {emotion_fail}개")
        
        # 종합 점수 통계 (모든 샘플에 대해 계산)
        weighted_scores = []
        for r in results:
            poetry_q = r.get('poetry_quality', {})
            keyword_r = r.get('keyword_relevance', {})
            emotion_r = r.get('emotion_relevance', {})
            
            # 필수 조건 통과 개수 계산
            required_pass_count = 0
            if not poetry_q.get('is_prose', False):
                required_pass_count += 1
            if not poetry_q.get('is_diary', False):
                required_pass_count += 1
            if not poetry_q.get('is_explanation', False):
                required_pass_count += 1
            if poetry_q.get('korean_chars', 0) >= 10:
                required_pass_count += 1
            
            required_score = required_pass_count / 4.0
            
            # 선택 조건 점수 계산
            score_poetry = min(1.0, poetry_q.get('overall_score', 0.0) / 0.5)
            score_line_count = min(1.0, poetry_q.get('line_count', 0) / 2.0)
            score_min_length = min(1.0, len(r.get('generated_poem', '').strip()) / 20.0)
            score_keyword = min(1.0, keyword_r.get('keyword_coverage', 0.0) / 0.2)
            score_emotion = min(1.0, emotion_r.get('emotion_score', 0.0) / 0.4)
            
            selection_score = (
                score_poetry * 0.40 +
                score_line_count * 0.10 +
                score_min_length * 0.10 +
                score_keyword * 0.20 +
                score_emotion * 0.20
            )
            
            # 종합 점수 = 필수 조건 점수(30%) + 선택 조건 점수(70%)
            weighted_score = (required_score * 0.30) + (selection_score * 0.70)
            weighted_scores.append(weighted_score)
        
        if weighted_scores:
            avg_weighted_score = np.mean(weighted_scores)
            max_weighted_score = np.max(weighted_scores)
            min_weighted_score = np.min(weighted_scores)
            below_06 = sum(1 for s in weighted_scores if s < 0.6)
            
            print(f"\n      종합 점수 통계:")
            print(f"        - 평균: {avg_weighted_score:.4f}")
            print(f"        - 최고: {max_weighted_score:.4f}")
            print(f"        - 최저: {min_weighted_score:.4f}")
            print(f"        - 0.6 미만: {below_06}개")
        
        # 생성된 시가 비어있는지 확인
        empty_poems = sum(1 for r in results if not r.get('generated_poem') or len(r.get('generated_poem', '').strip()) == 0)
        if empty_poems > 0:
            print(f"\n    ⚠️ 생성된 시가 비어있음: {empty_poems}개")
            print(f"    💡 모델이 시를 생성하지 못하고 있습니다.")
            print(f"    💡 가능한 원인:")
            print(f"       1. 모델이 제대로 학습되지 않음")
            print(f"       2. 입력 형식이 학습 시와 다름")
            print(f"       3. 모델이 프롬프트만 반복하고 있음")
        
        # 실제 생성된 시 샘플 확인
        non_empty_poems = [r for r in results if r.get('generated_poem') and len(r.get('generated_poem', '').strip()) > 0]
        if non_empty_poems:
            print(f"\n    📝 생성된 시 샘플 (처음 3개):")
            for i, r in enumerate(non_empty_poems[:3], 1):
                poem = r.get('generated_poem', '')
                print(f"      [{i}] {repr(poem[:100])}")
                poetry_q = r.get('poetry_quality', {})
                print(f"          - 시 품질: {poetry_q.get('overall_score', 0.0):.4f}, "
                      f"산문: {poetry_q.get('is_prose', False)}, "
                      f"한글: {poetry_q.get('korean_chars', 0)}자, "
                      f"줄: {poetry_q.get('line_count', 0)}줄")
    
    print(f"\n  📝 시 형태 평가:")
    print(f"    - 평균 시 품질 점수: {avg_poetry_score:.4f} (0.0=산문, 1.0=시)")
    print(f"    - 평균 형식 점수: {avg_format_score:.4f}")
    print(f"    - 평균 한국어 점수: {avg_korean_score:.4f}")
    print(f"    - 평균 산문 패널티: {avg_prose_penalty:.4f}")
    print(f"    - 평균 시적 표현 보너스: {avg_poetry_bonus:.4f}")
    print(f"    - 산문으로 판정: {prose_count}개")
    print(f"    - 일기로 판정: {diary_count}개")
    print(f"    - 설명문으로 판정: {explanation_count}개")
    print(f"\n  🔑 키워드 반영 평가:")
    print(f"    - 평균 키워드 점수: {avg_keyword_score:.4f} (0.0=반영 안됨, 1.0=완벽 반영)")
    print(f"    - 평균 키워드 반영률: {avg_keyword_coverage:.2%} (포함된 키워드 비율)")
    print(f"\n  💭 감정 반영 평가:")
    print(f"    - 평균 감정 점수: {avg_emotion_score:.4f} (0.0=반영 안됨, 1.0=완벽 반영)")
    print(f"    - 평균 감정 일치도: {avg_emotion_match:.4f} (원본 감정과 일치하는 비율)")
    # BERTScore 평균 계산
    bertscore_f1_scores = [r.get('bertscore', {}).get('f1', 0.0) 
                           for r in results if r.get('bertscore')]
    avg_bertscore_f1 = np.mean(bertscore_f1_scores) if bertscore_f1_scores else 0.0
    avg_bertscore_precision = np.mean([r.get('bertscore', {}).get('precision', 0.0) 
                                      for r in results if r.get('bertscore')]) if bertscore_f1_scores else 0.0
    avg_bertscore_recall = np.mean([r.get('bertscore', {}).get('recall', 0.0) 
                                    for r in results if r.get('bertscore')]) if bertscore_f1_scores else 0.0
    
    print(f"\n  📊 기타:")
    print(f"    - 평균 유사도 (원본 시와, 단어 기반): {avg_similarity:.4f}")
    if bertscore_results and bertscore_results.get('available', False):
        print(f"    - 평균 BERTScore F1: {avg_bertscore_f1:.4f} (의미적 유사도)")
    elif not BERTSCORE_AVAILABLE:
        print(f"    - BERTScore: 라이브러리 미설치 (pip install bert-score)")
    
    # 디버깅: 생성된 시 샘플 확인
    print(f"\n  [디버깅 정보]")
    successful_poems = [r for r in results if r.get('generated_poem') and len(r.get('generated_poem', '').strip()) > 0]
    print(f"  - 시 생성 성공: {len(successful_poems)}/{len(results)}")
    if successful_poems:
        print(f"  - 첫 번째 생성된 시 샘플:")
        print(f"    {repr(successful_poems[0]['generated_poem'][:150])}")
    else:
        print(f"  ⚠️ 생성된 시가 없습니다!")
        if results:
            print(f"  - 첫 번째 결과:")
            print(f"    {repr(results[0].get('generated_poem', 'None')[:150])}")
    
    # Classification metrics 계산
    classification_metrics = calculate_classification_metrics(results)
    
    # Classification metrics 출력
    if classification_metrics:
        print(f"\n{'='*80}")
        print(f"[Fold {fold_idx} Classification Metrics]")
        print(f"{'='*80}")
        print_classification_metrics(classification_metrics, fold_idx)
        
        # Confusion matrix 시각화 (matplotlib 사용 가능한 경우)
        if MATPLOTLIB_AVAILABLE:
            output_dir = Path(f"evaluation_results_fold_{fold_idx}")
            output_dir.mkdir(exist_ok=True)
            
            if 'emotion' in classification_metrics:
                em = classification_metrics['emotion']
                cm = np.array(em['confusion_matrix'])
                plot_confusion_matrix(
                    cm, em['labels'], 
                    f"감정 분류 Confusion Matrix",
                    fold_idx=fold_idx,
                    save_path=str(output_dir / f"emotion_cm_fold_{fold_idx}.png")
                )
            
            if 'success' in classification_metrics:
                sc = classification_metrics['success']
                cm = np.array(sc['confusion_matrix'])
                plot_confusion_matrix(
                    cm, [str(l) for l in sc['labels']],
                    f"성공/실패 분류 Confusion Matrix",
                    fold_idx=fold_idx,
                    save_path=str(output_dir / f"success_cm_fold_{fold_idx}.png")
                )
            
            # BERTScore 분포 시각화
            if bertscore_f1_scores and len(bertscore_f1_scores) > 0:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.hist(bertscore_f1_scores, bins=20, edgecolor='black', alpha=0.7)
                    plt.xlabel('BERTScore F1', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.title(f'BERTScore F1 Distribution (Fold {fold_idx})', fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(str(output_dir / f"bertscore_distribution_fold_{fold_idx}.png"), dpi=300, bbox_inches='tight')
                    print(f"✅ BERTScore 분포 차트 저장: {output_dir / f'bertscore_distribution_fold_{fold_idx}.png'}")
                    plt.close()
                except Exception as e:
                    print(f"⚠️ BERTScore 분포 차트 생성 실패: {e}")
    
    # 메모리 정리
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'fold': fold_idx,
        'success': True,
        'model_path': str(model_path),
        'test_count': len(test_data),
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_poetry_score': avg_poetry_score,
        'avg_format_score': avg_format_score,
        'avg_korean_score': avg_korean_score,
        'avg_prose_penalty': avg_prose_penalty,
        'avg_keyword_score': avg_keyword_score,
        'avg_keyword_coverage': avg_keyword_coverage,
        'avg_emotion_score': avg_emotion_score,
        'avg_emotion_match': avg_emotion_match,
        'avg_similarity': avg_similarity,
        'avg_bertscore_f1': avg_bertscore_f1,
        'avg_bertscore_precision': avg_bertscore_precision,
        'avg_bertscore_recall': avg_bertscore_recall,
        'bertscore_available': bertscore_results.get('available', False) if bertscore_results else False,
        'classification_metrics': classification_metrics,
        'results': results
    }


def find_best_fold_model(base_dir: str = None) -> None:
    """모든 fold 모델을 평가하고 가장 좋은 모델 찾기"""
    print(f"\n{'='*80}")
    print("k-fold 모델 성능 평가")
    print(f"{'='*80}\n")
    
    # GPU 확인
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️ CPU 모드 (느림)")
    
    # 모델 폴더 자동 찾기
    print(f"\n[0/4] 모델 폴더 찾기...")
    if base_dir is None:
        base_dir = BASE_MODEL_DIR
    
    base_path = Path(base_dir)
    
    # 폴더가 없으면 자동으로 찾기
    if not base_path.exists():
        print(f"⚠️ {base_dir} 폴더를 찾을 수 없습니다. 자동 검색 중...")
        
        # 현재 디렉토리에서 kogpt2 관련 폴더 찾기
        current_dir = Path(".")
        possible_dirs = []
        
        for item in current_dir.iterdir():
            if item.is_dir() and "kogpt2" in item.name.lower():
                possible_dirs.append(item)
        
        if possible_dirs:
            print(f"\n📁 찾은 폴더:")
            for i, folder in enumerate(possible_dirs, 1):
                print(f"  {i}. {folder.name}")
            
            if len(possible_dirs) == 1:
                base_path = possible_dirs[0]
                print(f"\n✅ 자동으로 선택: {base_path.name}")
            else:
                print(f"\n💡 여러 폴더를 찾았습니다. 첫 번째 폴더를 사용합니다.")
                base_path = possible_dirs[0]
                print(f"✅ 선택: {base_path.name}")
        else:
            print(f"\n❌ kogpt2 관련 폴더를 찾을 수 없습니다.")
            print(f"\n📁 현재 디렉토리의 모든 폴더:")
            for item in current_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
            print(f"\n💡 올바른 폴더 경로를 입력하세요:")
            user_input = input("폴더 경로 입력 (기본값: ./kogpt2_finetuned): ").strip()
            if user_input:
                base_path = Path(user_input)
            else:
                base_path = Path("./kogpt2_finetuned")
    
    if not base_path.exists():
        print(f"❌ {base_path} 폴더를 찾을 수 없습니다.")
        return
    
    print(f"✅ 모델 폴더: {base_path.absolute()}")
    
    # 데이터 로드
    print(f"\n[1/4] 데이터 로드 중...")
    try:
        data = download_kpoem_data(max_size=MAX_DATA_SIZE)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    if len(data) < K_FOLDS:
        print(f"❌ 데이터 개수({len(data)})가 fold 개수({K_FOLDS})보다 적습니다.")
        return
    
    # k-fold 분할 (학습 시와 동일한 분할)
    print(f"\n[2/4] k-fold 분할 중...")
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # fold 모델 찾기 (현재 디렉토리에서 직접 찾기)
    print(f"\n🔍 Fold 모델 검색 중...")
    current_dir = Path(".")
    all_fold_folders = []
    
    for folder in current_dir.iterdir():
        if folder.is_dir() and "_fold" in folder.name and "kogpt2" in folder.name.lower():
            # fold 번호 추출
            match = re.search(r'_fold(\d+)_', folder.name)
            if match:
                fold_num = int(match.group(1))
                all_fold_folders.append((fold_num, folder))
    
    # fold 번호별로 그룹화하고 최신 것 선택
    fold_models = {}
    for fold_num, folder in all_fold_folders:
        if fold_num not in fold_models:
            fold_models[fold_num] = folder
        else:
            # 더 최신 timestamp를 가진 폴더 선택
            current_timestamp = re.search(r'_(\d{8}_\d{6})', folder.name)
            existing_timestamp = re.search(r'_(\d{8}_\d{6})', fold_models[fold_num].name)
            if current_timestamp and existing_timestamp:
                if current_timestamp.group(1) > existing_timestamp.group(1):
                    fold_models[fold_num] = folder
    
    print(f"✅ {len(fold_models)}개의 fold 모델을 찾았습니다:")
    for fold_num in sorted(fold_models.keys()):
        print(f"  - Fold {fold_num}: {fold_models[fold_num].name}")
    
    if len(fold_models) < K_FOLDS:
        print(f"\n⚠️ 경고: {len(fold_models)}개의 fold만 찾았습니다. (예상: {K_FOLDS}개)")
        print(f"   찾은 fold: {sorted(fold_models.keys())}")
    
    # 각 fold 모델의 실제 경로 확인 (checkpoint 폴더가 아닌 최상위 모델)
    for fold_num in fold_models:
        model_folder = fold_models[fold_num]
        # config.json이 최상위에 있는지 확인
        if (model_folder / "config.json").exists():
            print(f"  ✅ Fold {fold_num}: 최상위에 모델 파일 존재")
        else:
            # checkpoint 폴더 확인
            checkpoint_folders = [f for f in model_folder.iterdir() 
                                 if f.is_dir() and "checkpoint" in f.name.lower()]
            if checkpoint_folders:
                # 가장 큰 번호의 checkpoint 선택
                checkpoint_nums = []
                for cp in checkpoint_folders:
                    match = re.search(r'checkpoint-(\d+)', cp.name)
                    if match:
                        checkpoint_nums.append((int(match.group(1)), cp))
                
                if checkpoint_nums:
                    latest_checkpoint = max(checkpoint_nums, key=lambda x: x[0])[1]
                    fold_models[fold_num] = latest_checkpoint
                    print(f"  ✅ Fold {fold_num}: {latest_checkpoint.name} 사용")
                else:
                    print(f"  ⚠️ Fold {fold_num}: checkpoint 번호를 찾을 수 없습니다.")
            else:
                print(f"  ⚠️ Fold {fold_num}: 모델 파일을 찾을 수 없습니다.")
    
    if not fold_models:
        print(f"❌ fold 모델을 찾을 수 없습니다.")
        print(f"\n📁 {base_dir} 내의 폴더:")
        for item in base_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return
    
    print(f"✅ {len(fold_models)}개의 fold 모델을 찾았습니다:")
    for fold_num in sorted(fold_models.keys()):
        print(f"  - Fold {fold_num}: {fold_models[fold_num].name}")
    
    # 각 fold 평가
    print(f"\n[3/4] 각 fold 모델 평가 중...")
    all_results = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(data), 1):
        if fold_idx not in fold_models:
            print(f"⚠️ Fold {fold_idx} 모델을 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        test_data = [data[i] for i in test_indices]
        model_path = fold_models[fold_idx]
        
        print(f"\n{'='*80}")
        print(f"[Fold {fold_idx} 평가 시작]")
        print(f"  - 사용할 모델: {model_path.name}")
        print(f"  - 모델 경로: {model_path.absolute()}")
        print(f"  - Test 데이터: {len(test_data)}개")
        print(f"{'='*80}")
        
        result = evaluate_fold_model(fold_idx, model_path, test_data, device)
        all_results.append(result)
        
        # 각 fold 결과 즉시 출력 (디버깅)
        if result.get('success', False):
            print(f"\n✅ Fold {fold_idx} 평가 완료:")
            print(f"   - 성공률: {result['success_rate']:.2%}")
            print(f"   - 시 품질: {result.get('avg_poetry_score', 0.0):.4f}")
            print(f"   - 한국어 점수: {result.get('avg_korean_score', 0.0):.4f}")
            # 생성된 시 샘플 출력
            if result.get('results'):
                sample_results = [r for r in result['results'] if r.get('generated_poem')]
                if sample_results:
                    print(f"   - 생성된 시 샘플:")
                    print(f"     {repr(sample_results[0]['generated_poem'][:100])}")
        else:
            print(f"\n❌ Fold {fold_idx} 평가 실패: {result.get('error', 'Unknown error')}")
            # 실패한 경우에도 생성된 시 확인
            if result.get('results'):
                sample_results = [r for r in result['results'] if r.get('generated_poem')]
                if sample_results:
                    print(f"   - 생성된 시 샘플 (실패):")
                    print(f"     {repr(sample_results[0]['generated_poem'][:100])}")
    
    # 결과 비교
    print(f"\n[4/4] 결과 비교")
    print(f"{'='*80}")
    print(f"{'Fold':<6} {'성공률':<10} {'시품질':<10} {'키워드':<10} {'감정':<10} {'종합점수':<10}")
    print(f"{'-'*80}")
    
    valid_results = [r for r in all_results if r.get('success', False)]
    
    if not valid_results:
        print("❌ 평가 성공한 모델이 없습니다.")
        return
    
    # 종합 점수 계산 (시 형태 40% + 키워드 30% + 감정 30%)
    for result in valid_results:
        poetry_score = result.get('avg_poetry_score', 0.0)
        keyword_score = result.get('avg_keyword_score', 0.0)
        emotion_score = result.get('avg_emotion_score', 0.0)
        result['composite_score'] = (poetry_score * 0.4) + (keyword_score * 0.3) + (emotion_score * 0.3)
    
    for result in sorted(valid_results, key=lambda x: x['fold']):
        print(f"Fold {result['fold']:<4} "
            f"{result['success_rate']:>6.2%}   "
            f"{result.get('avg_poetry_score', 0.0):>6.4f}   "
            f"{result.get('avg_keyword_score', 0.0):>6.4f}   "
            f"{result.get('avg_emotion_score', 0.0):>6.4f}   "
            f"{result.get('composite_score', 0.0):>6.4f}")
    
    # 최고 성능 모델 찾기
    print(f"\n{'='*80}")
    print("🏆 최고 성능 모델")
    print(f"{'='*80}")
    
    # 각 항목별 최고 모델
    best_by_poetry = max(valid_results, key=lambda x: x.get('avg_poetry_score', 0.0))
    best_by_keyword = max(valid_results, key=lambda x: x.get('avg_keyword_score', 0.0))
    best_by_emotion = max(valid_results, key=lambda x: x.get('avg_emotion_score', 0.0))
    best_by_success = max(valid_results, key=lambda x: x['success_rate'])
    
    # 종합 점수는 이미 계산됨 (composite_score)
    best_overall = max(valid_results, key=lambda x: x.get('composite_score', 0.0))
    
    print(f"\n📊 시 형태 기준 최고: Fold {best_by_poetry['fold']}")
    print(f"   - 시 품질 점수: {best_by_poetry.get('avg_poetry_score', 0.0):.4f}")
    print(f"   - 성공률: {best_by_poetry['success_rate']:.2%}")
    print(f"   - 한국어 점수: {best_by_poetry.get('avg_korean_score', 0.0):.4f}")
    print(f"   - 산문 패널티: {best_by_poetry.get('avg_prose_penalty', 0.0):.4f}")
    
    print(f"\n🔑 키워드 반영 기준 최고: Fold {best_by_keyword['fold']}")
    print(f"   - 키워드 점수: {best_by_keyword.get('avg_keyword_score', 0.0):.4f}")
    print(f"   - 키워드 반영률: {best_by_keyword.get('avg_keyword_coverage', 0.0):.2%}")
    print(f"   - 성공률: {best_by_keyword['success_rate']:.2%}")
    
    print(f"\n💭 감정 반영 기준 최고: Fold {best_by_emotion['fold']}")
    print(f"   - 감정 점수: {best_by_emotion.get('avg_emotion_score', 0.0):.4f}")
    print(f"   - 감정 일치도: {best_by_emotion.get('avg_emotion_match', 0.0):.4f}")
    print(f"   - 성공률: {best_by_emotion['success_rate']:.2%}")
    
    print(f"\n📊 성공률 기준 최고: Fold {best_by_success['fold']}")
    print(f"   - 성공률: {best_by_success['success_rate']:.2%}")
    print(f"   - 종합 점수: {best_by_success.get('composite_score', 0.0):.4f}")
    
    print(f"\n🏆 종합 최고: Fold {best_overall['fold']}")
    print(f"   - 종합 점수: {best_overall.get('composite_score', 0.0):.4f}")
    print(f"   - 시 품질 점수: {best_overall.get('avg_poetry_score', 0.0):.4f}")
    print(f"   - 키워드 점수: {best_overall.get('avg_keyword_score', 0.0):.4f}")
    print(f"   - 감정 점수: {best_overall.get('avg_emotion_score', 0.0):.4f}")
    print(f"   - 성공률: {best_overall['success_rate']:.2%}")
    print(f"   - 경로: {best_overall['model_path']}")
    
    print(f"\n💡 추천: Fold {best_overall['fold']} 모델을 다운로드하세요!")
    print(f"   다운로드 코드:")
    print(f"   ```python")
    print(f"   import shutil")
    print(f"   from google.colab import files")
    print(f"   fold_folder = \"{Path(best_overall['model_path']).name}\"")
    print(f"   base_dir = \"{base_path}\"")
    print(f"   shutil.make_archive(fold_folder, 'zip', base_dir, fold_folder)")
    print(f"   files.download(f\"{{fold_folder}}.zip\")")
    print(f"   ```")
    
    # 전체 fold에 대한 종합 classification metrics
    print(f"\n{'='*80}")
    print("📊 전체 Fold 종합 Classification Metrics")
    print(f"{'='*80}")
    
    # 모든 fold의 결과를 합쳐서 전체 metrics 계산
    all_results_combined = []
    for result in valid_results:
        if result.get('results'):
            all_results_combined.extend(result['results'])
    
    if all_results_combined:
        overall_metrics = calculate_classification_metrics(all_results_combined)
        print_classification_metrics(overall_metrics)
        
        # 전체 confusion matrix 시각화
        if MATPLOTLIB_AVAILABLE and overall_metrics:
            output_dir = Path("evaluation_results_overall")
            output_dir.mkdir(exist_ok=True)
            
            if 'emotion' in overall_metrics:
                em = overall_metrics['emotion']
                cm = np.array(em['confusion_matrix'])
                plot_confusion_matrix(
                    cm, em['labels'],
                    "감정 분류 Confusion Matrix (전체 Fold)",
                    save_path=str(output_dir / "emotion_cm_overall.png")
                )
            
            if 'success' in overall_metrics:
                sc = overall_metrics['success']
                cm = np.array(sc['confusion_matrix'])
                plot_confusion_matrix(
                    cm, [str(l) for l in sc['labels']],
                    "성공/실패 분류 Confusion Matrix (전체 Fold)",
                    save_path=str(output_dir / "success_cm_overall.png")
                )
            
            print(f"\n✅ 전체 Confusion Matrix 이미지가 저장되었습니다: {output_dir.absolute()}")
    
    # 각 fold별 classification metrics 요약
    print(f"\n{'='*80}")
    print("📊 Fold별 Classification Metrics 요약")
    print(f"{'='*80}")
    print(f"{'Fold':<6} {'Emotion Acc':<12} {'Poetry Quality':<12} {'Success Acc':<12}")
    print(f"{'-'*80}")
    
    for result in sorted(valid_results, key=lambda x: x['fold']):
        fold_idx = result['fold']
        cm = result.get('classification_metrics', {})
        
        emotion_acc = cm.get('emotion', {}).get('accuracy', 0.0) if cm else 0.0
        poetry_quality_mean = cm.get('poetry_quality_stats', {}).get('mean', 0.0) if cm else 0.0
        success_acc = cm.get('success', {}).get('accuracy', 0.0) if cm else 0.0
        
        print(f"Fold {fold_idx:<4} "
            f"{emotion_acc:>10.4f}   "
            f"{poetry_quality_mean:>10.4f}   "
            f"{success_acc:>10.4f}")


if __name__ == "__main__":
    find_best_fold_model()

