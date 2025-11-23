# poem_prompt_builder.py
# -*- coding: utf-8 -*-
"""
시 생성 프롬프트 구성 관련 함수
"""

from typing import List, Optional


def _build_messages_kogpt2(
    keywords: List[str], 
    mood: str, 
    lines: int, 
    original_text: str = "",
    banned_words: Optional[List[str]] = None,
    use_rhyme: bool = False,
    acrostic: Optional[str] = None,
) -> str:
    """
    koGPT2용 텍스트 프롬프트 생성 (chat template 없음)
    - 학습 시 사용한 패턴:  "산문: ...\\n시: ..."
    - 추론 시에도 같은 패턴을 유지하되,
        '원문에 최대한 붙어서' 표현만 시적으로 바꾸도록 강하게 제한
    """
    # 1) 키워드 정리
    kw_list = keywords[:6] if keywords else []
    kw_str = ", ".join(kw_list) if kw_list else "일상"

    # 2) 원본 텍스트 정리 (너무 길면 400자 정도로 자르기)
    prose = (original_text or "").strip()
    if prose and len(prose) > 400:
        prose = prose[:400].rstrip() + "..."

    # 3) 옵션(금지 단어, 두운/운율, 두문시) 한국어로 간단히
    constraint_lines = []
    if banned_words:
        constraint_lines.append(
            f"- 다음 단어들은 사용하지 마세요: {', '.join(banned_words)}"
        )
    if use_rhyme:
        constraint_lines.append(
            "- 가능한 한 비슷한 발음이나 반복되는 소리를 사용해 리듬감을 주세요."
        )
    if acrostic:
        acrostic_chars = " ".join(list(acrostic))
        constraint_lines.append(
            f"- 첫 글자 시: 각 줄의 첫 글자가 '{acrostic_chars}' 순서대로 오도록 해주세요 "
            f"(총 {len(acrostic)}줄)."
        )
    constraint_text = "\n".join(constraint_lines)

    # 4) 안내 문구 (핵심: 요약 금지 + 새 인물/사건 추가 금지 + 문장 하나하나 유지)
    guide = f"""아래 산문을 한국어 시로 바꿔주세요.

요구 사항 (매우 중요):
- **요약 금지**: 산문 속 문장/구절 하나하나의 정보(시간, 장소, 인물, 사건, 감정)를 가능한 한 모두 살리세요.
- **삭제 금지**: 특별한 이유가 없다면 원문에 있는 문장을 빼지 마세요.
- **추가 금지**: 원문에 없는 인물, 장소, 사건, 사물, 감정을 새로 만들지 마세요.
  (예: 원문에 없는 '아빠', 새로운 도시, 새로운 사고/죽음 등의 내용을 추가하지 마세요.)
- 등장인물(엄마, 나 등), 도시 이름(서울, 안동 등), 시간 정보(오늘, 12월 등)의 의미를 바꾸지 마세요.
- 말하고 있는 “상황/이야기 흐름”은 그대로 두고, **표현만 더 시처럼** 바꾸는 것이 목표입니다.

표현 방식:
- 각 원문 문장을 1~2줄 정도의 짧은 시 구절로 나누어 주세요.
- 원래 문장에 있던 단어들을 최대한 그대로 사용하되,
  조사, 어순, 약간의 단어만 바꿔서 자연스럽고 서정적인 느낌을 만들어 주세요.
- 분위기(톤): {mood}
- 키워드(참고용): {kw_str}
- 형식: 줄바꿈이 있는 한국어 자유시, **정확히 {lines}줄** (비어 있는 줄 없이).
- 가능한 한 한국어(한글)만 사용하고, 영어/숫자는 꼭 필요할 때만 제한적으로 사용하세요.
- 문장을 너무 설명조로 쓰지 말고, 이미지와 비유를 가볍게 사용해 주세요
  (예: 꽃처럼, 별처럼, 바람처럼, 그림자처럼).

금지/추가 조건:
{constraint_text if constraint_text else "- 특별한 추가 조건은 없습니다."}

요약:
- 내용/정보/인물/장소/시간은 그대로 유지
- **새로운 설정·사건·인물 추가 금지**
- 긴 설명을 짧은 시 구절로만 바꾸기

이제 아래 산문을 시로 바꿔주세요.
"""

    # 5) 학습 패턴과 맞추기: "산문: ...\\n\\n시:"
    prompt = guide

    if prose:
        prompt += f"\n산문: {prose}\n\n시:"
    else:
        # 원본 텍스트가 없는 경우: 키워드/분위기 기반 자유 창작
        prompt += (
            "\n산문: (원본 산문이 없습니다. "
            "위 키워드와 분위기를 바탕으로 새로 시를 지어주세요.)\n\n시:"
        )

    return prompt


def _build_messages(
    keywords: List[str], 
    mood: str, 
    lines: int, 
    original_text: str = "",
    banned_words: Optional[List[str]] = None,
    use_rhyme: bool = False,
    acrostic: Optional[str] = None,
) -> list:
    """
    Instruct 모델에 맞는 대화 템플릿 구성.
    - system: 역할 명시 (시인)
    - user: 요구조건(분위기, 줄 수, 키워드, 원본 텍스트 맥락, 시만 출력)
    """
    # 키워드를 명확하게 구성 (최대 6개까지 사용)
    kw_list = keywords[:6] if keywords else []
    if not kw_list:
        kw_str = "일상"
        print("[_build_messages] ⚠️ 키워드가 비어있어 '일상'을 사용합니다.")
    else:
        # 키워드를 명확하게 나열하고 강조
        kw_str = ", ".join(kw_list)
        print(f"[_build_messages] 사용할 키워드: {kw_list}")
    
    # 원본 텍스트의 핵심 문장 추출 (최대 100자)
    context_hint = ""
    if original_text:
        # 원본 텍스트의 앞부분 일부를 맥락으로 제공 (너무 길지 않게)
        context_preview = original_text[:100].strip()
        if len(original_text) > 100:
            context_preview += "..."
        context_hint = f"\nOriginal text context: {context_preview}\n"
    
    # 제약 조건 구성 (영어)
    constraints = []
    if banned_words:
        constraints.append(f"- Banned words: {', '.join(banned_words)} (DO NOT use these words)")
    if use_rhyme:
        constraints.append("- Use alliteration or rhyme to create rhythm")
    if acrostic:
        # 아크로스틱: 각 줄의 첫 글자가 지정된 문자여야 함
        acrostic_chars = " ".join(list(acrostic))
        constraints.append(f"- Acrostic: First letter of each line should be '{acrostic_chars}' in order (total {len(acrostic)} lines)")
    
    constraint_text = "\n".join(constraints) if constraints else ""
    
    # SOLAR-Instruct 모델에 최적화된 프롬프트 (키워드와 맥락 강조, 영어 프롬프트)
    user_msg = (
        f"Write a Korean poem (한국어 시) with at least {lines} lines based on the following keywords and context.\n\n"
        f"**IMPORTANT**: Write ONLY in Korean (Hangul). Do NOT use Chinese, Japanese, English, or any other language.\n\n"
        f"**What is Poetry?**\n"
        f"Poetry is NOT prose or diary. Poetry uses:\n"
        f"- Short, lyrical lines (5-12 characters per line)\n"
        f"- Imagery, metaphor, and symbolism (꽃처럼, 별처럼, 바람처럼)\n"
        f"- Emotional expression through concrete images\n"
        f"- Line breaks to create rhythm and space\n\n"
        f"**Poetry Examples (Write like this):**\n\n"
        f"Example 1:\n"
        f"봄날 꽃잎이\n"
        f"희망처럼 피어나고\n"
        f"따뜻한 바람이\n"
        f"새로운 시작을 안고\n"
        f"꽃향기 속에\n"
        f"미래가 흐른다\n\n"
        f"Example 2:\n"
        f"밤하늘 별들이\n"
        f"꿈을 비추고\n"
        f"잔잔한 마음에\n"
        f"희망이 스며든다\n"
        f"별빛 속에서\n"
        f"내일이 기다린다\n\n"
        f"**What NOT to write (Prose - DO NOT write like this):**\n"
        f"오늘은 날씨가 좋았다. 나는 공원에 가서 산책을 했다. 꽃들이 예쁘게 피어있었다.\n\n"
        f"**Required keywords**: {kw_str}\n"
        f"**Mood**: {mood}\n"
        f"{context_hint}"
        f"**Poetry Writing Rules (MUST follow):\n\n"
        f"1. ABSOLUTELY FORBIDDEN:\n"
        f"   - Do NOT use declarative endings: \"~다\", \"~이다\", \"~했다\", \"~을 했다\", \"~을 갔다\"\n"
        f"   - Do NOT use subjects or time markers: \"나는\", \"그는\", \"그녀는\", \"우리는\", \"오늘은\", \"어제는\"\n"
        f"   - Do NOT write long connected sentences (each line should be independent)\n"
        f"   - Do NOT write prose or diary-like text\n\n"
        f"2. How to write poetry:\n"
        f"   - Each line should be short and concise (5-12 characters, max 15 characters)\n"
        f"   - Actively use metaphors, similes, and symbols (e.g., \"꽃처럼\", \"별처럼\", \"바람처럼\", \"눈물처럼\")\n"
        f"   - Express emotions through concrete images, not directly\n"
        f"   - Create rhythm: vary line lengths but keep them short\n"
        f"   - Use line breaks to create space and breathing room\n\n"
        f"3. Style:\n"
        f"   - Use lyrical expressions, not narrative\n"
        f"   - Express actions or states directly without subjects\n"
        f"   - Conjunctions like \"~고\", \"~며\", \"~아서\" are allowed but keep lines short\n\n"
        f"- Write a POEM, NOT prose or diary\n"
        f"- Naturally include the keywords above in the poem\n"
        f"- Reflect the meaning and emotion of the keywords in the poem\n"
        f"- Create a {mood} mood in the poem\n"
        f"- Write ONLY in Korean (other languages are absolutely forbidden)\n"
        f"- Write at least {lines} lines\n"
        f"- Output ONLY the poem lines, nothing else (no title, no explanation)\n"
    )
    
    if constraint_text:
        user_msg += f"{constraint_text}\n"
    
    user_msg += f"- Output only the poem content (no explanations or comments)"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a Korean-language poet. Your ONLY job is to write Korean poems (한국어 시). "
                "Write poems ONLY in Korean (Hangul). Do NOT use Chinese, Japanese, English, or any other language. "
                "Poetry is NOT prose, NOT diary, NOT narrative. Poetry uses short lyrical lines, imagery, metaphor, and symbolism. "
                "Write emotional Korean poems that reflect the given keywords and original text context. "
                "Understand the meaning and emotion of the keywords and naturally incorporate them into the poem. "
                "Do NOT use declarative endings like \"~다\", \"~이다\", \"~했다\", \"~을 했다\". "
                "Do NOT specify subjects or time like \"나는\", \"그는\", \"그녀는\", \"오늘은\". "
                "Use lyrical expressions, not narrative. Actively use poetic techniques like metaphors, similes, and symbols "
                "(꽃처럼, 별처럼, 바람처럼). Divide into short and concise lines (5-12 characters, max 15 characters) "
                "to create rhythmic poetry. Express emotions through concrete images, not directly. "
                "Express actions or states directly without subjects. Remember: You are writing POETRY, not prose or diary."
            )
        },
        {"role": "user", "content": user_msg},
    ]
    
    # 프롬프트에 키워드가 제대로 들어갔는지 확인
    user_content = user_msg.lower()
    for kw in kw_list[:3]:  # 최대 3개만 확인
        if kw.lower() in user_content:
            print(f"[_build_messages] ✓ 키워드 '{kw}'가 프롬프트에 포함됨")
        else:
            print(f"[_build_messages] ⚠️ 키워드 '{kw}'가 프롬프트에 포함되지 않았습니다!")
    
    return messages


__all__ = ['_build_messages', '_build_messages_kogpt2']