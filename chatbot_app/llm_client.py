from pathlib import Path
from typing import Optional
from itertools import cycle
from django.conf import settings
from google import genai
from google.genai import types
import json
import re
import time

# ───────────────────────────────
# 클라이언트 초기화
# ───────────────────────────────
_GEMINI_KEYS = getattr(settings, "GEMINI_API_KEYS", None) or ([settings.GEMINI_API_KEY] if getattr(settings, "GEMINI_API_KEY", None) else [])
_GEMINI_KEY_CYCLE = cycle(_GEMINI_KEYS) if _GEMINI_KEYS else None

def _next_gemini_api_key():
    if not _GEMINI_KEY_CYCLE:
        raise RuntimeError("❌ GEMINI_API_KEY가 설정되지 않았습니다. .env와 config/settings.py를 확인하세요.")
    return next(_GEMINI_KEY_CYCLE)

def _make_client():
    api_key = _next_gemini_api_key()
    return genai.Client(api_key=api_key)

DEFAULT_MODEL = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
LLM_RETRY_DELAYS = getattr(settings, "GEMINI_RETRY_DELAYS", [15.0, 20.0])

def _retry_delay(attempt: int) -> float:
    if not LLM_RETRY_DELAYS:
        return 0
    idx = min(attempt, len(LLM_RETRY_DELAYS) - 1)
    return LLM_RETRY_DELAYS[idx]

# ───────────────────────────────
# 프롬프트 로드
# ───────────────────────────────
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "school_morph.md").read_text(encoding="utf-8")

# ───────────────────────────────
# 사용자 프롬프트 빌더
# ───────────────────────────────
def build_user_prompt(sentence: str, std_kr_entry: str = "", pretokenized: str = "", issue_context: str = "", decomposition_info: str = "", heuristic_info: str = "") -> str:
    prompt = (
        f'[분석 대상 문장]: "{sentence}"\n'
        f'[사전 정보]: "{std_kr_entry}"\n'
        f'[초기 분석 결과]: "{pretokenized}"\n'
    )
    if heuristic_info and heuristic_info != "없음":
        prompt += f'\n[휴리스틱 분석 정보]\n{heuristic_info}\n'
    if issue_context:
        prompt += f'\n[참고 자료: 쟁점 문서]\n{issue_context}\n'
    if decomposition_info:
        prompt += f'\n[단어 분해 정보]\n{decomposition_info}\n'
    return prompt

# ───────────────────────────────
# 학교 문법 전체 분석 (JSON 파싱 기능 내장)
# ───────────────────────────────
def analyze_school_grammar(
    sentence: str,
    std_kr_entry: str = "",
    pretokenized: str = "",
    issue_context: str = "",
    decomposition_info: str = "",
    heuristic_info: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    use_system_prompt: bool = True,
) -> tuple[str, str, list[str]]:
    max_retries = 3

    for attempt in range(max_retries):
        try:
            client = _make_client()
            _model = model or DEFAULT_MODEL

            config = types.GenerateContentConfig(temperature=temperature)
            user_prompt = build_user_prompt(sentence, std_kr_entry, pretokenized, issue_context, decomposition_info, heuristic_info)
            
            parts = []
            if use_system_prompt and SYSTEM_PROMPT:
                parts.append(types.Part(text=SYSTEM_PROMPT))
            parts.append(types.Part(text=user_prompt))

            resp = client.models.generate_content(
                model=_model,
                contents=[types.Content(parts=parts)],
                config=config,
            )
            
            # LLM 응답에서 JSON 추출 및 파싱
            raw_text = resp.text.strip()
            json_match = re.search(r"```json\n(.*?)\n```", raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = raw_text # 코드 블록이 없는 경우, 전체를 JSON으로 가정

            data = json.loads(json_str)
            
            main_md = data.get("main_analysis_markdown", "")
            sino_md = data.get("sino_korean_analysis_markdown", "")
            issues = data.get("detected_issues", [])
            
            return main_md, sino_md, issues

        except (json.JSONDecodeError, AttributeError) as e:
            # JSON 파싱 실패 시 재시도
            if attempt < max_retries - 1:
                delay = _retry_delay(attempt)
                print(f"JSON parsing failed. Retrying in {delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise RuntimeError(f"LLM did not return valid JSON after max retries: {str(e)}") from e
        except Exception as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                if attempt < max_retries - 1:
                    delay = _retry_delay(attempt)
                    print(f"API 503 Error. Retrying in {delay}s... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Gemini API error after max retries: {str(e)}") from e
            else:
                raise RuntimeError(f"Gemini API error: {str(e)}") from e
    return "", "", [] # Should not be reached

# ───────────────────────────────
# 단어 분해 전용
# ───────────────────────────────
DECOMPOSER_PROMPT = (PROMPT_DIR / "word_decomposer.md").read_text(encoding="utf-8")

def decompose_words(
    sentence: str,
    model: Optional[str] = None,
) -> str:
    max_retries = 3

    for attempt in range(max_retries):
        try:
            client = _make_client()
            _model = model or DEFAULT_MODEL

            prompt = DECOMPOSER_PROMPT.replace("{{SENTENCE}}", sentence)
            config = types.GenerateContentConfig(temperature=0.0)
            resp = client.models.generate_content(
                model=_model,
                contents=prompt,
                config=config,
            )
            return resp.text.strip()

        except Exception as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                if attempt < max_retries - 1:
                    delay = _retry_delay(attempt)
                    print(f"API 503 Error in decompose_words. Retrying in {delay}s... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"API 503 Error in decompose_words. Exceeded max retries ({max_retries}).")
                    # 재시도 실패 시 경고만 남기고 계속 진행
                    print(f"Warning: Word decomposition failed after max retries. Error: {str(e)}")
                    return "없음"
            else:
                # 503이 아닌 다른 오류는 즉시 실패 처리
                print(f"Warning: Word decomposition failed. Error: {str(e)}")
                return "없음"
    return "없음" # Should not be reached

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
