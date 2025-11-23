# -*- coding: utf-8 -*-
"""
시 생성 후 텍스트 후처리 관련 함수
"""

import re


def _clean_broken_chars(text: str) -> str:
    """
    깨진 문자와 제어 문자를 제거합니다.
    - 디코딩 에러로 인한 문자 ( 등) 제거
    - 제어 문자 제거 (탭, 캐리지 리턴 등 제외하고 유지)
    - 비정상적인 유니코드 문자 필터링
    """
    if not text:
        return text
    
    result = []
    for char in text:
        # 제어 문자 처리 (탭, 줄바꿈, 캐리지 리턴은 유지)
        if ord(char) < 32:
            if char in ['\n', '\r', '\t']:
                result.append(char)
            # 다른 제어 문자는 제거
            continue
        
        # 대체 문자 () 제거 - 디코딩 실패로 인한 문자
        if char == '\ufffd' or char == '':
            continue
        
        # 비정상적인 유니코드 범위 필터링
        # 일반적으로 사용되는 유니코드 범위만 허용
        code = ord(char)
        
        # 허용되는 범위:
        # - ASCII (0x00-0x7F)
        # - 라틴-1 보충 (0x80-0xFF)
        # - 일반 구두점 및 기호 (0x2000-0x206F)
        # - CJK 통합 한자 (0x4E00-0x9FFF)
        # - 한글 완성형 (0xAC00-0xD7AF)
        # - 한글 자모 (0x1100-0x11FF, 0x3130-0x318F, 0xA960-0xA97F)
        # - 히라가나/가타카나 (0x3040-0x30FF)
        # - 일반 구두점 (0x3000-0x303F)
        # - 기타 일반적으로 사용되는 문자들
        
        if (code <= 0x7F or  # ASCII
            0x80 <= code <= 0xFF or  # Latin-1 Supplement
            0x2000 <= code <= 0x206F or  # General Punctuation
            0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
            0xAC00 <= code <= 0xD7AF or  # Hangul Syllables
            0x1100 <= code <= 0x11FF or  # Hangul Jamo
            0x3130 <= code <= 0x318F or  # Hangul Compatibility Jamo
            0xA960 <= code <= 0xA97F or  # Hangul Jamo Extended-A
            0x3040 <= code <= 0x30FF or  # Hiragana/Katakana
            0x3000 <= code <= 0x303F or  # CJK Symbols and Punctuation
            0xFF00 <= code <= 0xFFEF):  # Halfwidth and Fullwidth Forms
            result.append(char)
        # 그 외의 문자는 제거 (깨진 문자 가능성)
        else:
            # 디버깅용: 제거되는 문자 로그 (너무 많으면 스킵)
            if len(result) < 1000:  # 초기 부분만 로그
                print(f"[clean] 깨진 문자 제거: {repr(char)} (U+{code:04X})", flush=True)
    
    return ''.join(result)


def _postprocess_poem(raw: str, min_lines: int, max_lines: int) -> str:
    """
    후처리:
    - 깨진 문자 제거
    - 명백한 지시줄(시:, Answer:, Output:, <|user|> 등) 제거
    - SOLAR 모델 특정 패턴 제거 (assistant 태그 등)
    - 공백 줄 제거
    - 행 수 min~max 범위로 제한 (내용을 과도하게 지우지 않음)
    """
    print(f"[postprocess] 원문 길이: {len(raw)}자")
    print(f"[postprocess] 원문 내용 (앞 500자): {repr(raw[:500])}")
    
    if not raw or len(raw.strip()) == 0:
        print("[postprocess] ❌ 원문이 비어있음")
        raise Exception("후처리할 원문이 비어있습니다.")
    
    # 깨진 문자 제거
    raw = _clean_broken_chars(raw)
    print(f"[postprocess] 깨진 문자 제거 후 길이: {len(raw)}자")
    
    # SOLAR 모델이 생성할 수 있는 패턴들 제거
    remove_patterns = [
        r"^시\s*[:：]\s*",  # "시:", "시 :" 등
        r"^Answer\s*[:：]\s*",
        r"^Output\s*[:：]\s*",
        r"^시 생성\s*[:：]\s*",
        r"^<\|assistant\|>\s*",  # SOLAR chat template 패턴
        r"^<\|user\|>\s*",
        r"^assistant\s*[:：]\s*",
        r"^user\s*[:：]\s*",
    ]
    
    # 번역 주석 패턴 제거 (괄호, 대괄호 모두)
    translation_comment_patterns = [
        r"\(번역\s*[:：]\s*[^)]+\)",  # (번역: ...)
        r"\(Translation\s*[:：]\s*[^)]+\)",  # (Translation: ...)
        r"\[번역\s*[:：]\s*[^\]]+\]",  # [번역: ...]
        r"\[Translation\s*[:：]\s*[^\]]+\]",  # [Translation: ...]
        r"\(번역\s*[^)]+\)",  # (번역 ...)
        r"\(Translation\s*[^)]+\)",  # (Translation ...)
        r"\[번역\s*[^\]]+\]",  # [번역 ...]
        r"\[Translation\s*[^\]]+\]",  # [Translation ...]
        r"번역\s*[:：]\s*[^\n]+",  # 줄 시작의 "번역: ..."
        r"Translation\s*[:：]\s*[^\n]+",  # Translation: ...]
    ]
    
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        
        # 패턴으로 지시어 제거
        original_s = s
        for pattern in remove_patterns:
            s = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
        
        # 번역 주석 패턴 제거
        for pattern in translation_comment_patterns:
            s = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
        
        # 제거 후 비어있으면 스킵
        if not s:
            if original_s != s:
                print(f"[postprocess] 지시줄/주석 제거: {repr(original_s)}")
            continue
            
        # 번역 주석으로만 구성된 줄은 제거
        if re.match(r"^[\(\[].*번역.*[\)\]]$", s, re.IGNORECASE):
            print(f"[postprocess] 번역 주석 줄 제거: {repr(s)}")
            continue
            
        # 해시태그, URL, 브랜드명 등 이상한 패턴 제거
        if re.search(r'#[가-힣a-zA-Z0-9_]+|@[가-힣a-zA-Z0-9_]+|https?://|www\.|[a-z]+_[a-z]+', s):
            print(f"[postprocess] 이상한 패턴 제거: {repr(s)}")
            continue
        
        # 방송사, 미디어 관련 텍스트 제거
        if re.search(r'\b(SBS|KBS|MBC|JTBC|TVN|Mnet|EBS|YTN|연합뉴스|뉴스|방송|스페셜|특집)\b', s, re.IGNORECASE):
            print(f"[postprocess] 방송사/미디어 관련 텍스트 제거: {repr(s)}")
            continue
        
        # 등호(=)가 포함된 수식 같은 패턴 제거
        if '=' in s and not any(ord('가') <= ord(c) <= ord('힣') for c in s):
            print(f"[postprocess] 수식 패턴 제거: {repr(s)}")
            continue
        
        # 의미 없는 대문자 단어 조합 제거 (예: "Keyp = Freezardience")
        # 대문자로 시작하는 단어가 2개 이상이고 한글이 없으면 제거
        uppercase_words = re.findall(r'\b[A-Z][a-z]+\b', s)
        if len(uppercase_words) >= 2 and not any(ord('가') <= ord(c) <= ord('힣') for c in s):
            print(f"[postprocess] 의미 없는 대문자 단어 조합 제거: {repr(s)}")
            continue
        
        # 이상한 영어 단어 패턴 (의미 없는 조합) 제거
        # 영어만 있고 한글이 없으며, 단어가 이상하게 조합된 경우
        english_only = re.match(r'^[a-zA-Z\s=!\[\]]+$', s)
        if english_only:
            # 일반적인 영어 단어가 아닌 이상한 조합인지 확인
            words = s.split()
            # 단어가 3개 이상이고 모두 대문자로 시작하거나 이상한 패턴이면 제거
            if len(words) >= 3 and all(w[0].isupper() if w else False for w in words if w and w[0].isalpha()):
                print(f"[postprocess] 이상한 영어 단어 조합 제거: {repr(s)}")
                continue
        
        # 영어가 너무 많은 줄 제거 (한국어 시가 아니므로)
        english_chars = sum(1 for c in s if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z'))
        korean_chars = sum(1 for c in s if ord('가') <= ord(c) <= ord('힣'))
        total_alpha = sum(1 for c in s if c.isalpha())
        
        # 영어 비율이 80% 이상이고 한글이 3자 미만이면 제거
        if total_alpha > 0 and (english_chars / total_alpha) > 0.8 and korean_chars < 3:
            print(f"[postprocess] 영어가 너무 많은 줄 제거 (영어: {english_chars}, 한글: {korean_chars}): {repr(s)}")
            continue
            
        # 너무 짧은 줄은 제외 (1-2자)
        if len(s) < 3 and not any(ord('가') <= ord(c) <= ord('힣') for c in s):
            continue
            
        lines.append(s)

    if not lines:
        print("[postprocess] ⚠️ 필터링 후 모든 줄이 제거됨 → 원문 반환")
        # 원문이 있으면 원문 반환 (최후의 수단)
        return raw.strip()

    if len(lines) < min_lines:
        print(f"[postprocess] ⚠️ 줄 수 부족({len(lines)}<{min_lines}) → 있는 만큼 사용")
        result = "\n".join(lines)
        if len(result.strip()) < 1:
            # 결과가 비어있으면 원문 반환
            print("[postprocess] 결과가 비어있어 원문 반환")
            return raw.strip()
        return result

    print(f"[postprocess] ✓ 줄 수 제한: {min_lines}~{max_lines}, 현재={len(lines)}")
    result = "\n".join(lines[:max_lines])
    
    # 최종 결과에서 번역 주석 패턴 한 번 더 제거 (중간에 포함된 경우 대비)
    for pattern in translation_comment_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    
    result = result.strip()
    
    # 결과가 비어있으면 원본 사용
    if not result or len(result.strip()) < 1:
        print("[postprocess] ⚠️ 제한 후 결과가 비어있어 원문 반환")
        return raw.strip()
    return result


__all__ = ['_clean_broken_chars', '_postprocess_poem']

