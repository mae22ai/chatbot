import os
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types

# ───────────────────────────────
# 클라이언트 초기화
# ───────────────────────────────
def _make_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")
    return genai.Client(api_key=api_key)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ───────────────────────────────
# 프롬프트 로드
# ───────────────────────────────
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "school_morph.md").read_text(encoding="utf-8")
COMPOUND_ONLY_SYSTEM = (PROMPT_DIR / "school_morph_compound.md").read_text(encoding="utf-8")

# ───────────────────────────────
# 사용자 프롬프트 빌더
# ───────────────────────────────
def build_user_prompt(sentence: str, std_kr_entry: str = "", pretokenized: str = "") -> str:
    return (
        f'문장: "{sentence}"\n'
        f'사전_표준국어: "{std_kr_entry}"\n'
        f'형태소기_결과(raw): "{pretokenized}"\n'
    )

# ───────────────────────────────
# 학교 문법 전체 분석
# ───────────────────────────────
def analyze_school_grammar(
    sentence: str,
    std_kr_entry: str = "",
    pretokenized: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    thinking_budget: Optional[int] = 0,
) -> str:
    client = _make_client()
    _model = model or DEFAULT_MODEL

    try:
        config = types.GenerateContentConfig(
            system_instruction=[SYSTEM_PROMPT],
            temperature=temperature,
        )
        if thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))

        resp = client.models.generate_content(
            model=_model,
            contents=build_user_prompt(sentence, std_kr_entry, pretokenized),
            config=config,
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

# ───────────────────────────────
# 합성/파생 판정 전용
# ───────────────────────────────
def analyze_compound_block(
    sentence: str,
    std_kr_entry: str = "",
    pretokenized: str = "",
    candidates: list[str] | None = None,
    posline: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    thinking_budget: Optional[int] = 0,
) -> str:
    client = _make_client()
    _model = model or DEFAULT_MODEL

    user = build_user_prompt(sentence, std_kr_entry, pretokenized)
    if candidates:
        user += "\n의심_후보: [" + ", ".join(candidates) + "]"
    if posline:
        user += "\n형태소기_요약: " + posline

    try:
        config = types.GenerateContentConfig(
            system_instruction=[COMPOUND_ONLY_SYSTEM],
            temperature=temperature,
        )
        if thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))

        resp = client.models.generate_content(
            model=_model,
            contents=user,
            config=config,
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")
