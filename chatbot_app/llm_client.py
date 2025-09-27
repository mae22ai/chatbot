from pathlib import Path
from typing import Optional
from django.conf import settings
from google import genai
from google.genai import types
import time

# ───────────────────────────────
# 클라이언트 초기화
# ───────────────────────────────
def _make_client():
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY가 설정되지 않았습니다. .env와 config/settings.py를 확인하세요.")
    return genai.Client(api_key=api_key)

DEFAULT_MODEL = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")

# ───────────────────────────────
# 프롬프트 로드
# ───────────────────────────────
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "school_morph.md").read_text(encoding="utf-8")

# ───────────────────────────────
# 사용자 프롬프트 빌더
# ───────────────────────────────
def build_user_prompt(sentence: str, std_kr_entry: str = "", pretokenized: str = "", issue_context: str = "", decomposition_info: str = "") -> str:
    prompt = (
        f'문장: "{sentence}"\n'
        f'사전_표준국어: "{std_kr_entry}"\n'
        f'형태소기_결과(raw): "{pretokenized}"\n'
    )
    if issue_context:
        prompt += f'\n[참고 자료: 쟁점 문서]\n{issue_context}\n'
    if decomposition_info:
        prompt += f'\n[단어 분해 정보]\n{decomposition_info}\n'
    return prompt

# ───────────────────────────────
# 학교 문법 전체 분석 (자동 재시도 내장)
# ───────────────────────────────
def analyze_school_grammar(
    sentence: str,
    std_kr_entry: str = "",
    pretokenized: str = "",
    issue_context: str = "",
    decomposition_info: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    thinking_budget: Optional[int] = 0,
    use_system_prompt: bool = True,
) -> str:
    max_retries = 5
    initial_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            client = _make_client()
            _model = model or DEFAULT_MODEL

            config = types.GenerateContentConfig(temperature=temperature)
            if thinking_budget is not None:
                config.thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))

            user_prompt = build_user_prompt(sentence, std_kr_entry, pretokenized, issue_context, decomposition_info)
            
            parts = []
            if use_system_prompt and SYSTEM_PROMPT:
                parts.append(types.Part(text=SYSTEM_PROMPT))
            parts.append(types.Part(text=user_prompt))

            resp = client.models.generate_content(
                model=_model,
                contents=[types.Content(parts=parts)],
                config=config,
            )
            return resp.text
        
        except Exception as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"API 503 Error in analyze_school_grammar. Retrying in {delay}s... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"API 503 Error. Exceeded max retries ({max_retries}).")
                    raise RuntimeError(f"Gemini API error after max retries: {str(e)}") from e
            else:
                # 503이 아닌 다른 오류는 즉시 실패 처리
                raise RuntimeError(f"Gemini API error: {str(e)}") from e

# ───────────────────────────────
# 단어 분해 전용
# ───────────────────────────────
DECOMPOSER_PROMPT = (PROMPT_DIR / "word_decomposer.md").read_text(encoding="utf-8")

def decompose_words(
    sentence: str,
    model: Optional[str] = None,
) -> str:
    client = _make_client()
    _model = model or DEFAULT_MODEL

    prompt = DECOMPOSER_PROMPT.replace("{{SENTENCE}}", sentence)

    try:
        config = types.GenerateContentConfig(temperature=0.0)
        resp = client.models.generate_content(
            model=_model,
            contents=prompt,
            config=config,
        )
        return resp.text.strip()
    except Exception as e:
        print(f"Warning: Word decomposition failed. Error: {str(e)}")
        return "없음"

# ───────────────────────────────
# 쟁점 탐지 전용
# ───────────────────────────────
ISSUE_DETECTOR_PROMPT = (PROMPT_DIR / "school_morph_issue_detector.md").read_text(encoding="utf-8")

def detect_grammatical_issue(
    analysis_markdown: str,
    issue_list: list[str],
    model: Optional[str] = None,
) -> str:
    client = _make_client()
    _model = model or DEFAULT_MODEL

    prompt = ISSUE_DETECTOR_PROMPT.replace("{{ANALYSIS_MARKDOWN}}", analysis_markdown)
    prompt = prompt.replace("{{ISSUE_LIST}}", ", ".join(issue_list))

    try:
        config = types.GenerateContentConfig(temperature=0.1)
        resp = client.models.generate_content(
            model=_model,
            contents=prompt,
            config=config,
        )
        return resp.text.strip()
    except Exception as e:
        # 쟁점 탐지는 실패해도 전체 분석에 영향을 주지 않도록 로깅만 하고 '없음'을 반환
        print(f"Warning: Grammatical issue detection failed. Error: {str(e)}")
        return "없음"

# ───────────────────────────────
# 한자어 판정 전용
# ───────────────────────────────
SINO_KOREAN_PROMPT = (PROMPT_DIR / "sino_korean_detector.md").read_text(encoding="utf-8")

def analyze_sino_korean(
    nouns: list[str],
    model: Optional[str] = None,
) -> str:
    client = _make_client()
    _model = model or DEFAULT_MODEL

    if not nouns:
        return ""

    prompt = SINO_KOREAN_PROMPT.replace("{{NOUN_LIST}}", ", ".join(nouns))

    try:
        config = types.GenerateContentConfig(temperature=0.1)
        resp = client.models.generate_content(
            model=_model,
            contents=prompt,
            config=config,
        )
        return resp.text.strip()
    except Exception as e:
        print(f"Warning: Sino-Korean analysis failed. Error: {str(e)}")
        return ""
