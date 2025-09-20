from pathlib import Path
from typing import Optional
from django.conf import settings
from google import genai
from google.genai import types

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
def build_user_prompt(sentence: str, std_kr_entry: str = "", pretokenized: str = "", issue_context: str = "") -> str:
    prompt = (
        f'문장: "{sentence}"\n'
        f'사전_표준국어: "{std_kr_entry}"\n'
        f'형태소기_결과(raw): "{pretokenized}"\n'
    )
    if issue_context:
        prompt += f'\n[참고 자료: 쟁점 문서]\n{issue_context}\n'
    return prompt

# ───────────────────────────────
# 학교 문법 전체 분석
# ───────────────────────────────
def analyze_school_grammar(
    sentence: str,
    std_kr_entry: str = "",
    pretokenized: str = "",
    issue_context: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    thinking_budget: Optional[int] = 0,
) -> str:
    client = _make_client()
    _model = model or DEFAULT_MODEL

    try:
        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        if thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))

        user_prompt = build_user_prompt(sentence, std_kr_entry, pretokenized, issue_context)
        resp = client.models.generate_content(
            model=_model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=SYSTEM_PROMPT),
                        types.Part(text=user_prompt),
                    ]
                )
            ],
            config=config,
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

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
