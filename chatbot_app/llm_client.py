import os
from typing import Optional

# Google GenAI SDK (Gemini)
from google import genai
from google.genai import types

# (옵션) settings에서 키를 가져오되, 없으면 ENV를 사용
try:
    # 사용자가 "GEMENI_API_KEY"로 보관했을 수 있으니 둘 다 시도
    from config import settings as _settings  # type: ignore
    _SETTINGS_KEY = getattr(_settings, "GEMINI_API_KEY", None) or getattr(_settings, "GEMENI_API_KEY", None)
except Exception:
    _SETTINGS_KEY = None

# ---- 클라이언트 생성 ----
def _make_client() -> genai.Client:
    # 우선순위: settings의 키 → ENV(GEMINI_API_KEY) → 무인자(ENV 자동 인식)
    api_key = _SETTINGS_KEY or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()  # ENV 자동 인식 지원 :contentReference[oaicite:2]{index=2}

# 환경변수로 모델 바꾸기 가능 (기본값은 속도/비용 우선)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # :contentReference[oaicite:3]{index=3}

# ---- 시스템 프롬프트(그대로 사용) ----
SYSTEM_PROMPT = r"""
# 🧩 학교 문법 형태소 분석 – 시스템 프롬프트 (템플릿)

**역할**

너는 *학교 문법 기준*으로 한국어 형태소를 분석하는 **교육용 전문 챗봇**이다.

입력 문장에 대해 **형태소·품사·실질/형식·자립/의존** 표를 제시하고, **합성어 vs 파생어**를 엄밀히 판정하여 **간단한 근거**를 덧붙인다.

출력은 반드시 아래 **출력 형식**만 사용한다(추가 설명·서문 금지).

---

## 권위/근거 우선순위 (반드시 이 순서로 판정)

1. **고려대한국어대사전 ‘형태’ 표기**(예: `[+새+잎]`, `[=덧+門]`)
2. **국립국어원 표준국어대사전/온라인가나다** 품사·풀이
3. **규칙 기반 휴리스틱**(아래 목록)
    
    ※ 상충 시 1→2→3. 사전 데이터가 없으면 **“사전 검증 불가”**로 명시하고 휴리스틱만 사용.
    

---

## 알고리즘

1. **어절 분해 → 형태소 분절**: 어간·어미·조사·접사 식별(과잉 분절 금지).
2. **어휘 단위 재구성**: 관형사+명사, 연속 명사, 접두/접미 의심 결합을 후보로 추출.
3. **사전 검증**: 제공된 사전 필드에서 ‘형태’ 표기를 먼저 확인.
4. **사전 불가 시 휴리스틱 적용**(아래).
5. **표 생성** + **합성/파생 판정** + **근거(1–2줄)**.
6. 불확실하면 **“판정보류(후보: 합성/파생)”**로 표기하고 한 줄 이유를 적시.

---

## 합성어/파생어 정의

- **합성어**: 자립형태소/어근 결합(통사적: *관형사+명사* 등 / 비통사적: *형용사어근+명사* 등).
- **파생어**: 접두·접미 결합으로 파생(예: **덧-문**, **풋-사과**, 사람**-들**, 사랑**스럽다**).

---

## 휴리스틱(사전 미확인 시만, 보수적으로 적용)

### A. ‘새’ 판정

1. 색채 형용사 어근(빨갛-/파랗-/하얗-/노랗-) 앞 **새-/샛-** → **접두(파생)** *(새빨갛다, 샛노랗다)*
2. 그 외 일반 명사 앞 **새** → **관형사**로 보고 **합성** *(새집, 새해)*

### B. 관형사 계열

- **첫-, 옛-**: 관형사 → **합성** *(첫사랑, 옛이야기)*

### C. 안전 접두사 리스트(파생)

- **덧-, 맨-, 헛-, 풋-, 숫-/암-**, **한-**, **홑-/겹-**, **생-**, **날-**, **왕-**, **돌-**
- 한자계: **무-, 비-, 불-/부-, 미-, 초-, 재-, 준-, 반-, 친-, 남-/여-**
    
    → 원칙적으로 **파생**.
    

### D. 안전 접미사 리스트(파생)

- **하다, -답다, -스럽다**, **-이/-히**, **-어치, -씩, -째, -투성이**
- 직업/성향: **-꾼, -쟁이, -뱅이, -장이, -잡이, -님, -씨, -내기**
- 한자계: **-적(的), -화(化), -성(性), -력(力), -률/-율(率), -가(家), -자(者), -인(人), -식(式), -론(論), -학(學)**

### E. 합성어 지표

- **관형사/관형형 + 명사**, **명사+명사**, **사이시옷**, **용언어근+명사**, **한자 병렬(남녀·강약)**

### F. 판정보류 트리거

- 접두/접미 해석과 합성 해석이 모두 가능한 경우 등

### G. 주의(어미 vs 접미)

- **-게, -도록, -지만, -으면, -으니, -음/-ㅁ, -기** 등은 **어미**.
- **들**은 **복수 접미사(파생)**.
- **하다**: 한자/명사어근+하다는 **접미 파생**.

---

## 출력 형식(마크다운, 이 구조를 절대 변경하지 말 것)

```
### 🔍 단어 분석 및 형태소 분석

* **단어**: {space\_separated\_tokens\_or\_list}
* **형태소**: {comma\_separated\_morphs\_in\_school\_grammar}

### 🔍 학교 문법 기준 형태소 분석 표

| 형태소 | 품사(세부) | 실질/형식 | 자립/의존 |
| --- | ------ | ----- | ----- |
| ... | ...    | ...   | ...   |

### 🔍 실질형태소와 형식형태소

* **실질형태소**: {list}
* **형식형태소**: {list}

### 🔍 자립형태소와 의존형태소

* **자립형태소**: {list}
* **의존형태소**: {list}

### 🧩 합성어/파생어 판정

{for each suspicious item (e.g., '새잎', '한겨울', '사랑스럽다') print block:}

* **항목**: <단어>
* **판정**: 합성어 / 파생어 / 판정보류
* **근거**: <사전표기 요약(예: 고려대 ‘형태’ \[+A+B]) 또는 휴리스틱 규칙 키워드/번호>
* **확신도**: 높음/중간/낮음
* (사전검증: {“고려대 확인” / “표준국어 확인” / “사전 검증 불가”})

```

---

## 입력 데이터(템플릿 변수)

- `{sentence}`, `{korea_u_form}`, `{std_kr_entry}`, `{pretokenized}` (사전 필드 비어 있으면 **“사전 검증 불가”** 명시)
"""

# ---- 사용자 프롬프트 빌더 ----
def build_user_prompt(sentence: str, korea_u_form: str = "", std_kr_entry: str = "", pretokenized: str = "") -> str:
    return (
        f'문장: "{sentence}"\n'
        f'사전_형태_고려대: "{korea_u_form}"\n'
        f'사전_표준국어: "{std_kr_entry}"\n'
        f'형태소기_결과(raw): "{pretokenized}"\n'
    )

# ---- 주 함수: 학교 문법 분석 ----
def analyze_school_grammar(
    sentence: str,
    korea_u_form: str = "",
    std_kr_entry: str = "",
    pretokenized: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    thinking_budget: Optional[int] = 0,  # 2.5 계열은 thinking 옵션 제공(Flash는 끌 수 있음) :contentReference[oaicite:4]{index=4}
) -> str:
    """
    Gemini로 템플릿 시스템 프롬프트 + 사용자 메시지를 보내
    '출력 형식'을 그대로 따르게 생성.
    """
    client = _make_client()
    _model = model or DEFAULT_MODEL

    try:
        config = types.GenerateContentConfig(
            system_instruction=[SYSTEM_PROMPT],  # 시스템 프롬프트 전달(GenAI SDK) :contentReference[oaicite:5]{index=5}
            temperature=temperature,
        )
        # thinking 설정(2.5 Flash에서 비용/지연 줄이고 싶을 때 0으로)
        if thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))  # :contentReference[oaicite:6]{index=6}

        resp = client.models.generate_content(
            model=_model,
            contents=build_user_prompt(sentence, korea_u_form, std_kr_entry, pretokenized),
            config=config,
        )
        return resp.text
    except Exception as e:
        # google-genai는 표준 Exception으로 떨어지는 경우가 많음 — 메시지만 래핑
        raise RuntimeError(f"Gemini API error: {str(e)}")

# === 합성/파생만 뽑는 LLM (Bareun 힌트 사용) ==========================
COMPOUND_ONLY_SYSTEM = r"""
너는 *학교 문법 기준* 한국어 형태소 교육 보조자다.
다음 입력을 보고 **오직** 아래 섹션만 출력하라(기타 설명/표 금지).

### 🧩 합성어/파생어 판정
- 항목: <단어>
- 판정: 합성어 / 파생어 / 판정보류
- 근거: (고려대 ‘형태’ 표기 요약, 표준국어 근거, 또는 휴리스틱 규칙)
- 확신도: 높음/중간/낮음
- (사전검증: “고려대 확인” / “표준국어 확인” / “사전 검증 불가”)

규칙:
- **후보 목록이 제공되면 그 후보들만 판정**한다.
- 우선순위: ① 고려대 ‘형태’ → ② 표준국어/온라인가나다 → ③ 휴리스틱.
"""

def analyze_compound_block(
    sentence: str,
    korea_u_form: str = "",
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

    user = build_user_prompt(sentence, korea_u_form, std_kr_entry, pretokenized)
    # Bareun 힌트(후보, POS 라인) 추가
    hint = ""
    if candidates:
        hint += "\n의심_후보: [" + ", ".join(candidates) + "]"
    if posline:
        hint += "\n형태소기_요약: " + posline

    try:
        config = types.GenerateContentConfig(
            system_instruction=[COMPOUND_ONLY_SYSTEM],
            temperature=temperature,
        )
        if thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))
        resp = client.models.generate_content(
            model=_model,
            contents=user + hint,
            config=config,
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

# ---- 기존 요약기(백워드 호환) ----
def summarize_analysis(analysis_result: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    기존 summarize 함수 유지. Gemini로 대체.
    """
    client = _make_client()
    _model = model or DEFAULT_MODEL
    prompt = f"""다음은 한국어 형태소 분석 결과입니다:
{analysis_result}

이 결과를 사람이 읽기 좋게, 간결한 형태로 정리해주세요.

'안녕하세요' 출력 예시:

'안녕하' : VA
'시' : EP
'어요' : EF
"""
    try:
        resp = client.models.generate_content(
            model=_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

