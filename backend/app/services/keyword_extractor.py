# -*- coding: utf-8 -*-
import warnings
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

# sklearn 경고 억제 (tokenizer 사용 시 token_pattern은 무시되지만 정상 동작)
# 전역 필터링으로 모든 sklearn 경고 억제
warnings.filterwarnings("ignore", message=".*token_pattern.*will not be used.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")


def _remove_josa(word: str) -> str:
    """
    한국어 단어에서 조사를 제거합니다.
    예: "사과를" -> "사과", "책이" -> "책", "친구와" -> "친구"
    """
    if not word or len(word) < 2:
        return word
    
    # 한국어가 아닌 경우 그대로 반환
    if not re.search(r'[가-힣]', word):
        return word
    
    # 일반적인 조사 패턴 제거 (받침 유무에 따라 다르게 처리)
    # 을/를, 이/가, 은/는, 와/과, 에/에서, 의, 도, 만, 조차, 까지, 부터 등
    josa_patterns = [
        r'을$', r'를$',  # 을/를
        r'이$', r'가$',  # 이/가
        r'은$', r'는$',  # 은/는
        r'과$', r'와$',  # 과/와
        r'에서$', r'에$',  # 에/에서
        r'의$',  # 의
        r'도$',  # 도
        r'만$',  # 만
        r'조차$',  # 조차
        r'까지$',  # 까지
        r'부터$',  # 부터
        r'로$', r'으로$',  # 로/으로
        r'라$', r'이라$',  # 라/이라
        r'나$', r'이나$',  # 나/이나
    ]
    
    cleaned = word
    for pattern in josa_patterns:
        cleaned = re.sub(pattern, '', cleaned)
        if cleaned != word:
            break  # 조사가 제거되었으면 중단
    
    # 조사 제거 후 최소 길이 확인 (1글자 이상이어야 함)
    if len(cleaned) >= 1:
        return cleaned
    else:
        return word  # 조사 제거 후 너무 짧아지면 원본 반환


def _tok_ko(text: str) -> List[str]:
    """한글/영문/숫자 토큰, 2글자 이상 (조사 제거 후)"""
    tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", text) if len(t) >= 2]
    # 조사 제거
    cleaned_tokens = [_remove_josa(t) for t in tokens]
    # 조사 제거 후에도 2글자 이상인 것만 반환
    return [t for t in cleaned_tokens if len(t) >= 2]


def _split_to_docs(ctx: str) -> List[str]:
    """TF-IDF용 소문서 쪼개기(문단/문장 기준 단순 분할)"""
    docs = [s.strip() for s in re.split(r"[.\n!?]+", ctx) if s.strip()]
    if len(docs) < 3:
        # 너무 적으면 문장을 슬라이딩 윈도우로 나누어 다양성 확보
        # 복제 대신 슬라이딩 윈도우 사용하여 TF-IDF 점수 개선
        if len(ctx) > 20:
            # 텍스트가 충분히 길면 슬라이딩 윈도우로 나눔
            window_size = max(10, len(ctx) // 3)
            docs = []
            for i in range(0, len(ctx), window_size // 2):
                window = ctx[i:i + window_size].strip()
                if window:
                    docs.append(window)
            if len(docs) < 3:
                docs = [ctx.strip()] * 3
        else:
            # 짧은 텍스트는 원본만 사용 (복제하지 않음)
            docs = [ctx.strip()]
    return docs[:10]  # 과도한 길이 방지


def _remove_similar_keywords(keywords: List[str]) -> List[str]:
    """
    의미적으로 유사하거나 중복되는 키워드를 제거합니다.
    - 한 키워드가 다른 키워드에 포함되는 경우 제거
    - 너무 짧은 키워드(1-2자) 중복 제거
    """
    if not keywords:
        return []
    
    # 길이순 정렬 (긴 것부터) - 긴 키워드가 더 의미있음
    sorted_kws = sorted(keywords, key=len, reverse=True)
    result = []
    seen = set()
    
    for kw in sorted_kws:
        kw_lower = kw.lower()
        # 이미 포함된 키워드인지 확인
        is_duplicate = False
        for existing in result:
            existing_lower = existing.lower()
            # 한 키워드가 다른 키워드에 포함되거나 동일한 경우
            if kw_lower in existing_lower or existing_lower in kw_lower:
                if len(kw) < len(existing):
                    # 더 짧은 키워드는 스킵
                    is_duplicate = True
                    break
                elif len(kw) > len(existing) and kw_lower != existing_lower:
                    # 더 긴 키워드로 교체
                    result.remove(existing)
                    seen.discard(existing_lower)
                    break
        
        if not is_duplicate and kw_lower not in seen:
            result.append(kw)
            seen.add(kw_lower)
    
    return result


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    TF-IDF를 사용하여 텍스트에서 키워드를 추출합니다.
    개선사항:
    - 더 많은 키워드 추출 (기본값: 10개)
    - 의미적으로 유사한 키워드 제거
    - 중복 제거 강화
    
    Args:
        text: 입력 텍스트
        max_keywords: 추출할 최대 키워드 개수 (기본값: 10)
    
    Returns:
        추출된 키워드 리스트 (중복 제거됨)
    """
    if not text or len(text.strip()) == 0:
        return []
    
    try:
        docs = _split_to_docs(text)
        # tokenizer를 사용하므로 token_pattern 파라미터를 전달하지 않음
        # warnings.catch_warnings()로 경고 억제
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*token_pattern.*will not be used.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            # tokenizer를 사용할 때는 token_pattern 파라미터 자체를 전달하지 않음
            # max_df를 1.0으로 설정하여 모든 단어 포함 (짧은 텍스트에서도 단어가 제외되지 않도록)
            vec = TfidfVectorizer(tokenizer=_tok_ko, min_df=1, max_df=1.0)  # 모든 단어 포함
        # 원본 텍스트가 docs에 포함되어 있지 않은 경우에만 추가
        if text.strip() not in docs:
            vec.fit(docs + [text.strip()])
        else:
            vec.fit(docs)
        scores = vec.transform([text]).toarray()[0]
        terms = vec.get_feature_names_out()
        idx = scores.argsort()[::-1]
        
        # 더 많은 키워드 추출 (나중에 필터링)
        candidate_kws = []
        seen = set()
        for i in idx:
            w = terms[i]
            # 조사 제거
            w_cleaned = _remove_josa(w)
            
            # 숫자만 있는 키워드 제외
            if w_cleaned.isdigit():
                continue
            # 너무 짧은 단어 제외 (1글자는 제외, 2글자 이상만)
            if len(w_cleaned) < 2:
                continue
            # 중복 확인
            if w_cleaned.lower() in seen:
                continue
            seen.add(w_cleaned.lower())
            candidate_kws.append(w_cleaned)
            # 충분한 후보를 모으면 중단 (필터링 전에 더 많이 수집)
            if len(candidate_kws) >= max_keywords * 2:
                break
        
        # 의미적으로 유사한 키워드 제거
        filtered_kws = _remove_similar_keywords(candidate_kws)
        
        # 최대 개수로 제한
        return filtered_kws[:max_keywords]
    
    except Exception as e:
        # 오류 발생 시 간단한 단어 추출
        korean_words = re.findall(r'[가-힣]+', text)
        # 조사 제거
        cleaned_words = [_remove_josa(w) for w in korean_words if len(w) >= 2]
        unique_words = list(dict.fromkeys([w for w in cleaned_words if len(w) >= 2]))
        # 중복 제거
        filtered = _remove_similar_keywords(unique_words)
        return filtered[:max_keywords]


def make_keyword_tag(keywords: List[str]) -> str:
    """키워드를 태그 형식으로 변환"""
    return f"<k>{', '.join(keywords)}</k>"
