# -*- coding: utf-8 -*-
"""
한국어 번역 모듈
Google Cloud Translation API v3를 사용하여 비한국어 텍스트를 한국어로 번역합니다.
"""

import os
import re
import json
import time
from typing import Optional

# Google Cloud Translation API v3 라이브러리 (선택 사항)
try:
    from google.cloud import translate_v3 as translate  # type: ignore
    from google.auth import default  # type: ignore
    _google_translate_available = True
except ImportError:
    translate = None
    default = None
    _google_translate_available = False

# 번역 설정 (환경 변수에서 로드)
GOOGLE_CLOUD_PROJECT_ID: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
USE_GOOGLE_TRANSLATION_API: bool = bool(GOOGLE_CLOUD_PROJECT_ID)


def detect_language(text: str) -> tuple[str, str]:
    """
    텍스트의 언어를 감지합니다.
    Returns: (언어명, 언어코드) 튜플
    """
    # 한국어 감지
    korean_chars = sum(1 for c in text if ord('가') <= ord(c) <= ord('힣'))
    # 중국어 감지
    chinese_chars = sum(1 for c in text if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
    # 일본어 감지
    japanese_hiragana = sum(1 for c in text if ord('\u3040') <= ord(c) <= ord('\u309f'))
    japanese_katakana = sum(1 for c in text if ord('\u30a0') <= ord(c) <= ord('\u30ff'))
    # 영어 감지 (간단한 휴리스틱: 알파벳 비율이 높고 한글이 적으면 영어로 간주)
    english_chars = sum(1 for c in text if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z'))
    total_chars = len([c for c in text if c.strip()])
    
    # 언어 우선순위 판단
    if korean_chars > 0 and korean_chars > chinese_chars and korean_chars > (japanese_hiragana + japanese_katakana):
        return ("한국어", "ko")
    elif chinese_chars > korean_chars * 2:
        return ("중국어", "zh-CN")
    elif (japanese_hiragana + japanese_katakana) > korean_chars:
        return ("일본어", "ja")
    elif english_chars > total_chars * 0.5 and korean_chars < total_chars * 0.3:
        return ("영어", "en")
    else:
        # 자동 감지 필요
        return ("자동감지", None)


def translate_to_korean(text: str, source_lang: str = None) -> str:
    """
    Google Cloud Translation API v3를 사용하여 텍스트를 한국어로 번역 (ADC 사용, API 키 불필요)
    source_lang이 None이면 자동으로 언어를 감지합니다.
    """
    if not USE_GOOGLE_TRANSLATION_API:
        raise Exception("Google Cloud Translation API 설정이 없습니다. GOOGLE_CLOUD_PROJECT_ID 환경 변수를 설정하세요.")
    
    if not _google_translate_available:
        raise Exception("google-cloud-translate 라이브러리가 설치되지 않았습니다. 'pip install google-cloud-translate'를 실행하세요.")
    
    if translate is None or default is None:
        raise Exception("google-cloud-translate 라이브러리를 import할 수 없습니다.")
    
    try:
        # 언어 감지
        detected_lang_name, detected_lang_code = detect_language(text)
        print(f"[translator] 언어 감지: {detected_lang_name}", flush=True)
        
        # 한국어면 번역 불필요
        if detected_lang_code == "ko":
            print(f"[translator] ✓ 이미 한국어입니다. 번역 불필요", flush=True)
            return text
        
        # 소스 언어 설정 (None이면 자동 감지)
        source_language_code = source_lang if source_lang else detected_lang_code
        
        print(f"[translator] Google Cloud Translation API v3로 {detected_lang_name} 텍스트를 한국어로 번역 중... (길이: {len(text)}자)", flush=True)
        
        # ADC(Application Default Credentials) 자동 사용
        # 로컬: gcloud auth application-default login 실행 필요
        # 클라우드: 자동으로 서비스 계정 사용
        credentials, project = default()
        
        # 서비스 계정 키 파일에서 프로젝트 ID 읽기 시도
        if not project and GOOGLE_APPLICATION_CREDENTIALS:
            try:
                with open(GOOGLE_APPLICATION_CREDENTIALS, 'r') as f:
                    creds_json = json.load(f)
                    file_project_id = creds_json.get("project_id")
                    if file_project_id:
                        project = file_project_id
                        print(f"[translator] 서비스 계정 키 파일에서 프로젝트 ID 확인: {project}", flush=True)
            except Exception as e:
                print(f"[translator] 키 파일에서 프로젝트 ID 읽기 실패: {e}", flush=True)
        
        # 환경 변수에서 프로젝트 ID 확인
        if not project and GOOGLE_CLOUD_PROJECT_ID:
            project = GOOGLE_CLOUD_PROJECT_ID
            print(f"[translator] 환경 변수에서 프로젝트 ID 확인: {project}", flush=True)
        
        if not project:
            raise Exception("프로젝트 ID를 찾을 수 없습니다. GOOGLE_CLOUD_PROJECT_ID 환경 변수를 설정하거나 서비스 계정 키 파일의 project_id를 확인하세요.")
        
        # Translation API v3 클라이언트 생성
        client = translate.TranslationServiceClient(credentials=credentials)
        
        # 번역 요청 (source_language_code가 None이면 자동 감지)
        parent = f"projects/{project}/locations/global"
        translate_kwargs = {
            "parent": parent,
            "contents": [text],
            "target_language_code": "ko",  # 한국어
            "mime_type": "text/plain"
        }
        
        # 소스 언어가 있으면 지정, 없으면 자동 감지
        if source_language_code:
            translate_kwargs["source_language_code"] = source_language_code
        
        response = client.translate_text(**translate_kwargs)
        
        # 번역 결과 추출
        translations = response.translations
        if not translations:
            raise Exception("번역 결과가 비어있습니다.")
        
        translated = translations[0].translated_text
        
        if not translated:
            raise Exception("번역 결과가 비어있습니다.")
        
        # 감지된 언어 정보 확인
        detected_language = translations[0].detected_language_code if hasattr(translations[0], 'detected_language_code') else source_language_code
        print(f"[translator] ✓ 번역 완료 ({detected_language} → 한국어, 번역된 길이: {len(translated)}자)", flush=True)
        return translated
        
    except Exception as e:
        error_msg = str(e)
        print(f"[translator] ❌ 번역 실패: {error_msg}", flush=True)
        
        # 일반적인 에러 메시지 제공
        if "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
            raise Exception(f"인증 실패: gcloud auth application-default login을 실행하거나 GOOGLE_APPLICATION_CREDENTIALS를 설정하세요. {error_msg[:150]}")
        elif "has not been used" in error_msg.lower() or "disabled" in error_msg.lower() or "enable" in error_msg.lower():
            # API 활성화 오류
            project_id_in_error = ""
            api_url = ""
            
            if "project" in error_msg.lower():
                # 에러 메시지에서 프로젝트 번호 추출 시도
                match = re.search(r'project (\d+)', error_msg)
                if not match:
                    match = re.search(r'project=(\d+)', error_msg)
                if match:
                    project_id_in_error = match.group(1)
                    api_url = f"https://console.cloud.google.com/apis/api/translate.googleapis.com/overview?project={project_id_in_error}"
            
            # 현재 사용 중인 프로젝트 ID 확인
            current_project_id = GOOGLE_CLOUD_PROJECT_ID
            if not current_project_id and GOOGLE_APPLICATION_CREDENTIALS:
                try:
                    with open(GOOGLE_APPLICATION_CREDENTIALS, 'r') as f:
                        creds_json = json.load(f)
                        current_project_id = creds_json.get("project_id")
                except:
                    pass
            
            error_guide = "\n\n" + "="*70 + "\n"
            error_guide += "❌ Cloud Translation API가 활성화되지 않았습니다!\n"
            error_guide += "="*70 + "\n\n"
            
            if api_url:
                error_guide += f"🔗 바로 가기: {api_url}\n"
                error_guide += "   위 링크를 클릭하면 API 활성화 페이지로 이동합니다!\n\n"
            
            error_guide += "📋 해결 방법:\n\n"
            error_guide += "방법 1: 직접 링크로 활성화 (가장 빠름)\n"
            if api_url:
                error_guide += f"   1. {api_url}\n"
                error_guide += "   2. 페이지에서 '사용 설정' 버튼 클릭\n"
            else:
                error_guide += "   1. https://console.cloud.google.com/apis/library/translate.googleapis.com 접속\n"
                if current_project_id:
                    error_guide += f"   2. 프로젝트가 '{current_project_id}'인지 확인\n"
                error_guide += "   3. '사용 설정' 버튼 클릭\n"
            error_guide += "   4. 활성화 완료 후 1-2분 대기 (전파 시간)\n"
            error_guide += "   5. 다시 시도\n\n"
            
            error_guide += "방법 2: 수동으로 활성화\n"
            error_guide += "   1. https://console.cloud.google.com/ 접속\n"
            if project_id_in_error:
                error_guide += f"   2. 프로젝트 '{project_id_in_error}' 선택\n"
            elif current_project_id:
                error_guide += f"   2. 프로젝트 '{current_project_id}' 선택\n"
            error_guide += "   3. 'API 및 서비스' → '라이브러리'\n"
            error_guide += "   4. 'Cloud Translation API' 검색\n"
            error_guide += "   5. '사용 설정' 클릭\n\n"
            
            if project_id_in_error and current_project_id and project_id_in_error != str(current_project_id):
                error_guide += f"⚠️ 주의: 프로젝트 ID 불일치 감지!\n"
                error_guide += f"   - 에러의 프로젝트: {project_id_in_error}\n"
                error_guide += f"   - 설정된 프로젝트: {current_project_id}\n"
                error_guide += f"   - 서비스 계정 키 파일의 project_id를 확인하세요!\n\n"
            
            error_guide += "="*70 + "\n"
            
            raise Exception(f"Cloud Translation API 활성화 필요: {error_msg[:200]}{error_guide}")
        elif "project" in error_msg.lower():
            raise Exception(f"프로젝트 설정 실패: GOOGLE_CLOUD_PROJECT_ID 환경 변수를 설정하세요. {error_msg[:150]}")
        else:
            raise Exception(f"텍스트를 한국어로 번역하는데 실패했습니다: {error_msg[:200]}")


def translate_poem_with_retry(poem: str, max_retries: int = 5) -> str:
    """
    시를 한국어로 번역합니다. 재시도 로직이 포함되어 있습니다.
    
    Args:
        poem: 번역할 시 텍스트
        max_retries: 최대 재시도 횟수
    
    Returns:
        번역된 한국어 시 텍스트
    """
    # 언어 분석 먼저 수행
    korean_chars = sum(1 for c in poem if ord('가') <= ord(c) <= ord('힣'))
    chinese_chars = sum(1 for c in poem if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
    japanese_hiragana = sum(1 for c in poem if ord('\u3040') <= ord(c) <= ord('\u309f'))
    japanese_katakana = sum(1 for c in poem if ord('\u30a0') <= ord(c) <= ord('\u30ff'))
    japanese_chars = japanese_hiragana + japanese_katakana
    total_chars = len([c for c in poem if c.strip()])
    
    # 한국어만 포함되어 있는 경우 번역 불필요
    non_korean_chars = chinese_chars + japanese_chars
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
    
    # 한국어 비율이 높고 비한국어 문자가 거의 없으면 번역 불필요
    if korean_ratio > 0.7 and non_korean_chars == 0:
        print(f"[translator] ✓ 시가 이미 한국어로만 구성되어 있습니다. (한국어: {korean_chars}자, 비한국어: {non_korean_chars}자)", flush=True)
        return poem
    
    # 비한국어 문자가 있지만 번역 API가 없는 경우
    if not USE_GOOGLE_TRANSLATION_API:
        if non_korean_chars > 0:
            # API가 없으면 설정 가이드 출력
            print(f"[translator] ❌ Google Cloud Translation API가 설정되지 않았습니다. 번역이 필수입니다!", flush=True)
            print(f"[translator]", flush=True)
            print(f"[translator] ============================================================", flush=True)
            print(f"[translator] Google Cloud Translation API v3 설정이 필요합니다!", flush=True)
            print(f"[translator] ============================================================", flush=True)
            print(f"[translator]", flush=True)
            print(f"[translator] 📋 설정 방법:", flush=True)
            print(f"[translator]", flush=True)
            print(f"[translator] 방법 1: 서비스 계정 키 파일 사용 (Colab 권장)", flush=True)
            print(f"[translator]   1. Google Cloud Console 접속: https://console.cloud.google.com/", flush=True)
            print(f"[translator]   2. 프로젝트 생성/선택", flush=True)
            print(f"[translator]   3. Cloud Translation API 활성화", flush=True)
            print(f"[translator]   4. 서비스 계정 생성 → JSON 키 다운로드", flush=True)
            print(f"[translator]   5. COLAB_FINAL.py에서 다음 설정:", flush=True)
            print(f"[translator]      os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/key.json'", flush=True)
            print(f"[translator]      os.environ['GOOGLE_CLOUD_PROJECT_ID'] = 'your-project-id'", flush=True)
            print(f"[translator]", flush=True)
            print(f"[translator] 방법 2: gcloud CLI 사용", flush=True)
            print(f"[translator]   !gcloud auth application-default login", flush=True)
            print(f"[translator]   os.environ['GOOGLE_CLOUD_PROJECT_ID'] = 'your-project-id'", flush=True)
            print(f"[translator]", flush=True)
            print(f"[translator] ============================================================", flush=True)
            print(f"[translator]", flush=True)
            raise Exception(f"비한국어 문자가 포함된 시가 생성되었지만 번역 API가 설정되지 않았습니다. (한국어: {korean_chars}자, 중국어: {chinese_chars}자, 일본어: {japanese_chars}자) 위의 로그를 참고하여 API를 설정하세요.")
        else:
            # 한국어만 있으면 그대로 반환
            print(f"[translator] ✓ 시가 한국어로만 구성되어 있습니다. 번역 불필요.", flush=True)
            return poem
    
    # 번역 API가 있는 경우 - 영어 분석 추가
    english_chars = sum(1 for c in poem if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z'))
    
    print(f"[translator] ============================================================", flush=True)
    print(f"[translator] 비한국어 시가 감지되었습니다. 한국어로 번역을 시작합니다.", flush=True)
    print(f"[translator] 원본 시 - 한국어: {korean_chars}자, 중국어: {chinese_chars}자, 일본어: {japanese_chars}자, 전체: {total_chars}자", flush=True)
    print(f"[translator] ============================================================", flush=True)
    
    translation_success = False
    last_error = None
    translated_poem = poem
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
    
    for attempt in range(max_retries):
        try:
            print(f"[translator] 시도 {attempt + 1}/{max_retries}: 전체 시를 한국어로 번역 중...", flush=True)
            
            # 번역 실행
            translated = translate_to_korean(poem)
            
            # 번역 결과 검증
            translated_korean = sum(1 for c in translated if ord('가') <= ord(c) <= ord('힣'))
            translated_chinese = sum(1 for c in translated if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
            translated_japanese = sum(1 for c in translated if (ord('\u3040') <= ord(c) <= ord('\u309f') or ord('\u30a0') <= ord(c) <= ord('\u30ff')))
            translated_english = sum(1 for c in translated if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z'))
            translated_total = len([c for c in translated if c.strip()])
            
            print(f"[translator] 번역 결과 분석:", flush=True)
            print(f"[translator]   - 한국어: {translated_korean}자 (이전: {korean_chars}자)", flush=True)
            print(f"[translator]   - 중국어: {translated_chinese}자 (이전: {chinese_chars}자)", flush=True)
            print(f"[translator]   - 일본어: {translated_japanese}자 (이전: {japanese_chars}자)", flush=True)
            print(f"[translator]   - 영어: {translated_english}자 (이전: {english_chars}자)", flush=True)
            print(f"[translator]   - 전체: {translated_total}자", flush=True)
            
            # 비한국어가 없는지 확인
            non_korean_total = translated_chinese + translated_japanese
            non_korean_ratio = non_korean_total / translated_total if translated_total > 0 else 0
            english_ratio = translated_english / translated_total if translated_total > 0 else 0
            translated_korean_ratio = translated_korean / translated_total if translated_total > 0 else 0
            
            # 완벽한 번역: 한국어 비율이 90% 이상이고 비한국어가 거의 없음
            is_perfect_translation = (
                translated_korean > 0 and 
                translated_korean_ratio >= 0.9 and 
                non_korean_total == 0 and 
                english_ratio < 0.1
            )
            
            if is_perfect_translation:
                print(f"[translator] ✓✓✓ 번역 완료! 완전히 한국어로 변환되었습니다!", flush=True)
                print(f"[translator] 한국어: {translated_korean}자 ({translated_korean_ratio:.1%}), 비한국어: 0자", flush=True)
                translated_poem = translated
                translation_success = True
                print(f"[translator] ============================================================", flush=True)
                break
            elif translated_korean > korean_chars * 1.1 and (translated_chinese + translated_japanese) < (chinese_chars + japanese_chars) * 0.8:
                # 번역 개선: 한국어가 10% 이상 늘고 비한국어가 20% 이상 줄음
                improvement = ((translated_korean - korean_chars) / max(korean_chars, 1)) * 100
                print(f"[translator] ✓ 번역 개선됨! 한국어 {improvement:.1f}% 증가", flush=True)
                translated_poem = translated
                korean_chars = translated_korean
                chinese_chars = translated_chinese
                japanese_chars = translated_japanese
                english_chars = translated_english
                total_chars = len([c for c in translated_poem if c.strip()])
                korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
                
                # 비한국어가 조금이라도 남아있으면 무조건 재시도
                has_remaining_foreign = (
                    translated_chinese > 0 or 
                    translated_japanese > 0 or 
                    english_ratio > 0.15 or
                    translated_korean_ratio < 0.85
                )
                
                if has_remaining_foreign and attempt < max_retries - 1:
                    remaining = f"중국어: {translated_chinese}자, 일본어: {translated_japanese}자, 영어: {translated_english}자 (영어 비율: {english_ratio:.1%})"
                    print(f"[translator] ⚠️ 아직 비한국어가 남아있습니다 ({remaining}). 재시도합니다...", flush=True)
                    time.sleep(1.5)
                    continue
                elif translated_korean_ratio >= 0.85:
                    print(f"[translator] ✓ 번역 완료: 한국어 비율 {translated_korean_ratio:.1%} (수용 가능)", flush=True)
                    translation_success = True
                    print(f"[translator] ============================================================", flush=True)
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"[translator] ⚠️ 한국어 비율이 낮습니다 ({translated_korean_ratio:.1%}). 재시도합니다...", flush=True)
                        time.sleep(1.5)
                        continue
                    else:
                        translation_success = True
                        print(f"[translator] ⚠️ 최종 시도: 한국어 비율 {translated_korean_ratio:.1%}", flush=True)
                        print(f"[translator] ============================================================", flush=True)
                        break
            elif translated_korean > 0:
                print(f"[translator] ⚠️ 부분 번역: 한국어 {translated_korean}자 포함, 비한국어도 남아있음", flush=True)
                
                if (translated_chinese > 0 or translated_japanese > 0 or english_ratio > 0.2) and attempt < max_retries - 1:
                    remaining = f"중국어: {translated_chinese}자, 일본어: {translated_japanese}자, 영어: {translated_english}자"
                    print(f"[translator] 비한국어가 많습니다 ({remaining}). 재시도합니다...", flush=True)
                    translated_poem = translated
                    korean_chars = translated_korean
                    chinese_chars = translated_chinese
                    japanese_chars = translated_japanese
                    english_chars = translated_english
                    total_chars = len([c for c in translated_poem if c.strip()])
                    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
                    time.sleep(1.5)
                    continue
                else:
                    translated_poem = translated
                    korean_chars = translated_korean
                    chinese_chars = translated_chinese
                    japanese_chars = translated_japanese
                    english_chars = translated_english
                    total_chars = len([c for c in translated_poem if c.strip()])
                    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
                    translation_success = True
                    print(f"[translator] ============================================================", flush=True)
                    break
            else:
                print(f"[translator] ❌ 번역 결과에 한국어가 없습니다 (한국어: {translated_korean}자). 재시도...", flush=True)
                last_error = f"번역 결과에 한국어가 포함되지 않았습니다 (한국어: {translated_korean}자)"
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise Exception(f"번역이 실패했습니다. (원본 한국어: {korean_chars}자, 번역 한국어: {translated_korean}자)")

        except Exception as e:
            error_msg = str(e)
            last_error = error_msg
            print(f"[translator] ❌ 번역 시도 {attempt + 1} 실패: {error_msg}", flush=True)
            
            if attempt < max_retries - 1:
                print(f"[translator] 1초 후 재시도합니다...", flush=True)
                time.sleep(1)
                continue
            else:
                print(f"[translator] ❌ 모든 번역 시도 실패 (최대 {max_retries}회)!", flush=True)
                print(f"[translator] 마지막 오류: {error_msg[:300]}", flush=True)
                print(f"[translator] ============================================================", flush=True)
                raise Exception(f"시를 한국어로 번역하는데 실패했습니다 (최대 {max_retries}회 시도). 마지막 오류: {error_msg[:200]}")
    
    # 최종 검증
    if not translation_success:
        raise Exception(f"번역에 실패했습니다. 최대 {max_retries}회 시도했지만 한국어로 변환하지 못했습니다. 마지막 오류: {last_error[:200] if last_error else '알 수 없음'}")
    
    # 번역 성공 후 최종 확인
    final_chinese = sum(1 for c in translated_poem if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
    final_japanese = sum(1 for c in translated_poem if (ord('\u3040') <= ord(c) <= ord('\u309f') or ord('\u30a0') <= ord(c) <= ord('\u30ff')))
    final_english = sum(1 for c in translated_poem if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z'))
    final_korean = sum(1 for c in translated_poem if ord('가') <= ord(c) <= ord('힣'))
    final_total = len([c for c in translated_poem if c.strip()])
    final_korean_ratio = final_korean / final_total if final_total > 0 else 0
    final_non_korean_total = final_chinese + final_japanese
    
    print(f"[translator] 최종 검증:", flush=True)
    print(f"[translator]   - 한국어: {final_korean}자 ({final_korean_ratio:.1%})", flush=True)
    print(f"[translator]   - 중국어: {final_chinese}자", flush=True)
    print(f"[translator]   - 일본어: {final_japanese}자", flush=True)
    print(f"[translator]   - 영어: {final_english}자", flush=True)
    
    # 여러 언어가 섞여있는지 확인
    if final_non_korean_total > 0 or final_english > final_total * 0.15:
        remaining_langs = []
        if final_chinese > 0:
            remaining_langs.append(f"중국어 {final_chinese}자")
        if final_japanese > 0:
            remaining_langs.append(f"일본어 {final_japanese}자")
        if final_english > final_total * 0.15:
            remaining_langs.append(f"영어 {final_english}자 ({final_english/final_total:.1%})")
        
        remaining_str = ", ".join(remaining_langs)
        print(f"[translator] ⚠️ 경고: 번역 후에도 비한국어가 남아있습니다 ({remaining_str})", flush=True)
        
        # 비한국어가 많으면 마지막으로 한 번 더 번역 시도
        if final_korean_ratio < 0.8 and final_non_korean_total > final_total * 0.1:
            print(f"[translator] 비한국어 비율이 높습니다 ({final_korean_ratio:.1%}). 마지막 번역 시도를 진행합니다...", flush=True)
            try:
                final_translated = translate_to_korean(translated_poem)
                final_translated_korean = sum(1 for c in final_translated if ord('가') <= ord(c) <= ord('힣'))
                final_translated_chinese = sum(1 for c in final_translated if ord('\u4e00') <= ord(c) <= ord('\u9fff'))
                final_translated_japanese = sum(1 for c in final_translated if (ord('\u3040') <= ord(c) <= ord('\u309f') or ord('\u30a0') <= ord(c) <= ord('\u30ff')))
                final_translated_english = sum(1 for c in final_translated if c.isalpha() and ord('a') <= ord(c.lower()) <= ord('z'))
                final_translated_total = len([c for c in final_translated if c.strip()])
                final_translated_korean_ratio = final_translated_korean / final_translated_total if final_translated_total > 0 else 0
                final_translated_non_korean = final_translated_chinese + final_translated_japanese
                final_translated_non_korean_ratio = final_translated_non_korean / final_translated_total if final_translated_total > 0 else 0
                final_translated_english_ratio = final_translated_english / final_translated_total if final_translated_total > 0 else 0
                
                if final_translated_korean_ratio > final_korean_ratio:
                    print(f"[translator] ✓ 최종 번역 개선: 한국어 비율 {final_korean_ratio:.1%} → {final_translated_korean_ratio:.1%}", flush=True)
                    translated_poem = final_translated
                else:
                    print(f"[translator] 최종 번역 시도로는 개선되지 않았습니다.", flush=True)
            except Exception as e:
                print(f"[translator] 최종 번역 시도 실패: {e}", flush=True)
        
        # 최종 검증: 비한국어가 너무 많이 남아있으면 예외 발생
        final_non_korean_ratio = final_non_korean_total / final_total if final_total > 0 else 0
        final_english_ratio = final_english / final_total if final_total > 0 else 0
        total_foreign_ratio = final_non_korean_ratio + final_english_ratio
        
        if total_foreign_ratio >= 0.2 or final_korean_ratio < 0.7:
            error_msg = (
                f"번역 후에도 비한국어가 너무 많이 남아있습니다. "
                f"한국어 비율: {final_korean_ratio:.1%}, "
                f"비한국어 비율: {total_foreign_ratio:.1%} "
                f"(중국어: {final_chinese}자, 일본어: {final_japanese}자, 영어: {final_english}자). "
                f"시를 다시 생성하거나 번역 API 설정을 확인하세요."
            )
            print(f"[translator] ❌ {error_msg}", flush=True)
            raise Exception(error_msg)
        elif total_foreign_ratio >= 0.1 or final_korean_ratio < 0.85:
            print(f"[translator] ⚠️ 경고: 비한국어가 일부 남아있습니다 (비율: {total_foreign_ratio:.1%}, 한국어: {final_korean_ratio:.1%})", flush=True)
            print(f"[translator]    수용 가능한 수준이지만, 이상적이지는 않습니다.", flush=True)
        else:
            print(f"[translator] ✓ 번역 완료: 비한국어가 적게 남아있습니다 (비율: {total_foreign_ratio:.1%})", flush=True)
    else:
        print(f"[translator] ✓ 완벽: 모든 텍스트가 한국어로 변환되었습니다!", flush=True)
    
    return translated_poem

