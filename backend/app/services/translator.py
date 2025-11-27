# -*- coding: utf-8 -*-
"""
translator.py

시 텍스트에서 '비한국어(한자/영어/기타 문자)'가 섞인 부분을
최대한 한국어로 번역해서 돌려준다.

최종 보장:
- 결과 문자열에는 한글/숫자/기본 문장부호/공백만 남는다.
- 원래 시의 "줄 수"는 그대로 유지된다. (줄을 통째로 버리지 않음)
"""

import os
import re
import time
import html
from typing import Optional

import requests

# =============================================================================
# 환경변수에서 API 키 읽기
# =============================================================================

GOOGLE_TRANSLATE_API_KEY = (
    os.getenv("GOOGLE_TRANSLATE_API_KEY")
    or os.getenv("GOOGLE_TRANSLATION_API_KEY")
)

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


# =============================================================================
# 내부 유틸
# =============================================================================

def _has_non_korean(text: str) -> bool:
    """
    한글/숫자/기본 문장부호/공백을 제외한 문자가 하나라도 있으면 True.
    (영어, 한자, 일본어, 이모지 등 포함)
    """
    if not text:
        return False
    # 허용: 한글, 자모, 숫자, 공백, 기본 문장부호
    return bool(re.search(r"[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s.,?!~…·\-()]", text))


def _google_translate(text: str, target: str = "ko", source: Optional[str] = None) -> str:
    """
    Google Translation API를 사용하여 text를 target 언어로 번역.
    - 실패하면 원문 그대로 돌려줌.
    """
    if not GOOGLE_TRANSLATE_API_KEY:
        print("[translator] GOOGLE_TRANSLATE_API_KEY / GOOGLE_TRANSLATION_API_KEY 미설정", flush=True)
        return text

    if not text:
        return text

    params = {
        "key": GOOGLE_TRANSLATE_API_KEY,
        "q": text,
        "target": target,
    }
    if source:
        params["source"] = source

    resp = None
    try:
        resp = requests.post(TRANSLATE_URL, data=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        translated = data["data"]["translations"][0]["translatedText"]
        translated = html.unescape(translated)
        return translated
    except Exception as e:
        print(f"[translator] Google 번역 호출 오류: {e}", flush=True)
        if resp is not None:
            try:
                print(f"[translator] raw response: {resp.text}", flush=True)
            except Exception:
                pass
        return text


# =============================================================================
# 비한글 문자 강제 정리 + 덩어리 번역
# =============================================================================

# 자주 나올 수 있는 한자 몇 개는 의미를 살짝 매핑
HANJA_MAP = {
    "心": "마음",
    "熱": "열",
    "愛": "사랑",
    "夢": "꿈",
    "光": "빛",
    "星": "별",
    "花": "꽃",
}


def _translate_foreign_chunks(text: str) -> str:
    """
    한 줄 안에 남아 있는 '외국어 덩어리'만 골라서 다시 번역 시도.
    - 영어 단어, 한자 연속, 일본어(히라가나/가타카나) 등
    - chunk 단위로 Google 번역을 호출해서 한글로 치환
    """

    if not text or not _has_non_korean(text):
        return text

    # 영어 / 한자 / 일본어(히라가나, 가타카나) 덩어리들을 매칭
    pattern = re.compile(
        r"[A-Za-z]+"
        r"|[一-龯㐀-䶵々〆ヵヶ]+"  # 한자
        r"|[ぁ-ん]+"
        r"|[ァ-ンー]+"
    )

    def repl(match: re.Match) -> str:
        chunk = match.group(0)

        # 1) 한 글자 한자인데 HANJA_MAP에 있으면 바로 매핑
        if chunk in HANJA_MAP:
            return HANJA_MAP[chunk]

        # 2) 그 외는 Google 번역에 한 번 더 맡김
        translated = _google_translate(chunk, target="ko")
        # 번역 결과가 여전히 chunk랑 같고, 비한국어면 그냥 지움
        if translated == chunk and _has_non_korean(translated):
            return ""
        return translated

    return pattern.sub(repl, text)


def _force_korean_only(text: str) -> str:
    """
    최종 안전장치:
    1) HANJA_MAP에 있는 한자는 한글로 치환
    2) _translate_foreign_chunks()로 외국어 덩어리 번역 시도
    3) 그 후에도 남아 있는 비허용 문자는 제거
       (숫자와 기본 문장부호는 유지)

    이 함수가 반환하는 문자열 안에는
    '한글/숫자/공백/기본 문장부호' 외의 문자는 없다.
    """
    if not text:
        return text

    # 1) 단일 한자 매핑
    for src, tgt in HANJA_MAP.items():
        text = text.replace(src, tgt)

    # 2) 덩어리 단위 번역
    text = _translate_foreign_chunks(text)

    # 3) 남은 비허용 문자 제거
    cleaned = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s.,?!~…·\-()]", "", text)
    # 연속 공백 정리
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


# =============================================================================
# 줄 단위 번역 (핵심)
# =============================================================================

def translate_poem_with_retry_linewise(poem: str, max_retry: int = 2) -> str:
    """
    시 전체 문자열을 받아서, '비한국어가 포함된 줄만' 골라 번역.
    1차: 줄 전체를 ko로 번역
    2차: 결과에 남은 외국어 덩어리들을 다시 chunk 번역
    3차: 그래도 남은 비허용 문자는 강제 정리(_force_korean_only)

    보장:
    - 원래 줄 수는 그대로 유지 (빈 줄이라도 남김)
    - 각 줄 안의 문자는 모두 허용 문자(한글/숫자/기본 문장부호/공백)만 남음
    """
    poem = (poem or "").strip("\n")
    if not poem:
        return poem

    lines = poem.splitlines()
    new_lines = []

    for ln in lines:
        original_line = ln  # 줄 자체는 반드시 하나 넣는다
        s = ln.strip()

        if not s:
            # 완전히 빈 줄이면 그대로 추가 (줄 수 유지)
            new_lines.append("")
            continue

        # 이 줄이 전부 한국어(허용 문자)라면, 마지막에 한 번만 정리
        if not _has_non_korean(s):
            cleaned = _force_korean_only(original_line)
            # 한 글자도 안 남으면 그냥 빈 줄로라도 유지
            new_lines.append(cleaned if cleaned is not None else "")
            continue

        # 비한국어가 포함된 줄 → 번역 시도
        translated_line = s
        last_err = None

        for attempt in range(1, max_retry + 1):
            try:
                print(f"[translator] 줄 전체 번역 시도 {attempt}/{max_retry}: {repr(s)}", flush=True)
                translated_line = _google_translate(s, target="ko")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[translator] 줄 번역 실패({attempt}): {e}", flush=True)
                time.sleep(1.0)

        if last_err is not None:
            print(f"[translator] 줄 번역 완전 실패 → 원문 유지: {repr(s)}", flush=True)
            candidate = original_line
        else:
            candidate = translated_line

        # 번역 결과에서 남은 외국어 덩어리 재번역 + 최종 정리
        cleaned_line = _force_korean_only(candidate)

        # ✅ 여기서 절대 줄을 버리지 않는다
        if cleaned_line is None:
            cleaned_line = ""
        new_lines.append(cleaned_line)

    # 라인들 다시 합치기 (줄 구조는 유지)
    return "\n".join(new_lines)


# =============================================================================
# 기존 이름과의 호환용 래퍼
# =============================================================================

def translate_poem_with_retry(poem: str, max_retry: int = 2) -> str:
    """
    기존 코드에서 사용하던 이름을 유지하기 위한 래퍼.
    내부 구현은 '줄 단위 번역 + 덩어리 재번역 + 한글 강제 정리' 방식.
    """
    return translate_poem_with_retry_linewise(poem, max_retry=max_retry)


__all__ = [
    "translate_poem_with_retry",
    "translate_poem_with_retry_linewise",
]