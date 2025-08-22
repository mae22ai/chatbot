# chatbot_app/school_morph.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import json

# ─────────────────────────────────────────────────────────
# Bareun POS → 학교 문법 표기 맵핑(세분)
# ─────────────────────────────────────────────────────────
def _tag_to_school(tag: str) -> Tuple[str, str, str]:
    t = (tag or "").upper()

    # 명사 계열
    if t == "NNG":  return ("명사",       "실질", "자립")
    if t == "NNP":  return ("고유명사",   "실질", "자립")
    if t == "NNB":  return ("의존명사",   "실질", "의존")
    if t == "NP":   return ("대명사",     "실질", "자립")
    if t == "NR":   return ("수사",       "실질", "자립")

    # 수식언
    if t == "MM":   return ("관형사",     "실질", "자립")
    if t in ("MAG","MAJ"): return ("부사","실질", "자립")

    # 감탄사
    if t == "IC":   return ("감탄사",     "실질", "자립")

    # 용언 어간
    if t == "VV":   return ("동사 어간",   "실질", "의존")
    if t == "VA":   return ("형용사 어간", "실질", "의존")
    if t == "VX":   return ("보조용언 어간","실질","의존")
    if t == "VCP":  return ("계사 어간",   "실질", "의존")
    if t == "VCN":  return ("부정지사 어간","실질","의존")

    # 어미
    if t == "EP":   return ("선어말 어미", "형식", "의존")
    if t == "EF":   return ("어말어미(종결)","형식","의존")
    if t == "EC":   return ("연결 어미",   "형식", "의존")
    if t == "ETM":  return ("전성어미(관형)","형식","의존")
    if t == "ETN":  return ("전성어미(명사)","형식","의존")

    # 조사(세분)
    if t == "JKS":  return ("주격 조사",   "형식", "의존")
    if t == "JKC":  return ("보격 조사",   "형식", "의존")
    if t == "JKG":  return ("관형격 조사", "형식", "의존")
    if t == "JKO":  return ("목적격 조사", "형식", "의존")
    if t == "JKB":  return ("부사격 조사", "형식", "의존")
    if t == "JKV":  return ("호격 조사",   "형식", "의존")
    if t == "JKQ":  return ("인용격 조사", "형식", "의존")
    if t == "JX":   return ("보조사",     "형식", "의존")
    if t == "JC":   return ("접속 조사",   "형식", "의존")
    if t.startswith("J"): return ("조사",  "형식", "의존")

    # 접사/어근
    if t == "XPN":  return ("접두사",      "형식", "의존")
    if t == "XSN":  return ("접미사(명사)","형식", "의존")
    if t == "XSV":  return ("접미사(동사)","형식", "의존")
    if t == "XSA":  return ("접미사(형용사)","형식","의존")
    if t == "XR":   return ("어근",        "실질", "의존")

    # 기타
    if t in ("SN","SL","SH","SY"): return ("기타", "형식", "의존")
    return ("기타", "형식", "의존")


def _best_effort_parse(bareun_output: Any) -> List[Dict[str,str]]:
    """Bareun JSON/str을 [{'morph':..., 'tag':...}, ...]로 평탄화."""
    doc = bareun_output
    if isinstance(doc, str):
        try: doc = json.loads(doc)
        except Exception: return []

    # 구조1
    try:
        out = []
        for s in doc.get("sentences", []):
            for tok in s.get("tokens", []):
                for m in tok.get("morphemes", []):
                    lemma = m.get("lemma") or m.get("text") or m.get("morph") or ""
                    tag   = m.get("tag")   or m.get("pos")  or ""
                    if lemma: out.append({"morph": lemma, "tag": tag})
        if out: return out
    except Exception:
        pass

    # 구조2
    try:
        out = []
        for m in doc.get("morphs", []):
            lemma = m.get("lemma") or m.get("text") or m.get("morph") or ""
            tag   = m.get("tag")   or m.get("pos")  or ""
            if lemma: out.append({"morph": lemma, "tag": tag})
        if out: return out
    except Exception:
        pass

    return []


def parse_for_hybrid(bareun_output: Any) -> Tuple[List[Dict[str,str]], List[str], str]:
    """
    반환:
      morph_list: [{'morph','tag'}...]
      candidates: 합성/파생 의심 후보 목록(주로 NNG 길이 ≥ 2)
      posline: '나/NP 는/JX 어제/MAG ...' 형태의 한 줄 요약
    """
    morphs = _best_effort_parse(bareun_output)
    posline = " ".join(f"{m['morph']}/{m.get('tag','')}" for m in morphs)

    candidates = []
    for m in morphs:
        tag = (m.get("tag") or "").upper()
        if tag in ("NNG","NNB","NNP") and len(m["morph"]) >= 2:
            candidates.append(m["morph"])
    # 중복 제거(순서 보존)
    seen = set(); cand_uniq = []
    for w in candidates:
        if w not in seen:
            seen.add(w); cand_uniq.append(w)
    return morphs, cand_uniq, posline


def build_sections_from_bareun(bareun_output: Any) -> Tuple[str, bool]:
    """
    Bareun 결과로
      ① 단어/형태소 요약
      ② 학교 문법 표
      ③ 실질/형식, 자립/의존 목록
    을 **마크다운**으로 생성.
    """
    morph_list = _best_effort_parse(bareun_output)
    if not morph_list:
        return ("", False)

    # 표 행
    rows = []
    for it in morph_list:
        pos, subst, dep = _tag_to_school(it.get("tag",""))
        rows.append((it["morph"], pos, subst, dep))

    # 토큰 대략 복원(보기에만)
    tokens = []
    buf = []
    for morph, pos, subst, dep in rows:
        if subst == "실질" and dep == "자립":
            if buf: tokens.append("".join(buf)); buf = []
            tokens.append(morph)
        else:
            buf.append(morph)
    if buf: tokens.append("".join(buf))

    # 목록
    real = [m for (m,_,s,_) in rows if s=="실질"]
    form = [m for (m,_,s,_) in rows if s=="형식"]
    free = [m for (m,_,_,d) in rows if d=="자립"]
    bound= [m for (m,_,_,d) in rows if d=="의존"]

    # MD
    L = []
    L.append("### 🔍 단어 분석 및 형태소 분석\n")
    L.append(f"* **단어**: {', '.join(tokens) if tokens else '—'}")
    L.append(f"* **형태소**: {', '.join(m for (m,_,_,_) in rows)}\n")

    L.append("### 🔍 학교 문법 기준 형태소 분석 표\n")
    L.append("| 형태소 | 품사(세부) | 실질/형식 | 자립/의존 |")
    L.append("| --- | --- | --- | --- |")
    for morph, pos, subst, dep in rows:
        L.append(f"| {morph} | {pos} | {subst} | {dep} |")

    L.append("\n### 🔍 실질형태소와 형식형태소\n")
    L.append(f"* **실질형태소**: {', '.join(real) if real else '—'}")
    L.append(f"* **형식형태소**: {', '.join(form) if form else '—'}")

    L.append("\n### 🔍 자립형태소와 의존형태소\n")
    L.append(f"* **자립형태소**: {', '.join(free) if free else '—'}")
    L.append(f"* **의존형태소**: {', '.join(bound) if bound else '—'}")

    return ("\n".join(L), True)
