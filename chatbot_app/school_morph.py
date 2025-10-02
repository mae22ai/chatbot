# chatbot_app/school_morph.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
from .pos_heuristics import (
    classify_determiner,
    classify_interjection,
    classify_particle_by_heuristic,
    classify_ep_by_heuristic,
    classify_nominal_ending,
    classify_adnominal_ending,
    classify_all_ec,
    classify_deriv_suffix
)

def _best_effort_parse(bareun_output: Any) -> List[Dict[str,str]]:
    """Bareun JSON/str을 [{'morph':..., 'tag':...}, ...]로 평탄화."""
    doc = bareun_output
    if isinstance(doc, str):
        try: doc = json.loads(doc)
        except Exception: return []

    # Bareun v3.2.0 이상 구조
    try:
        out = []
        for s in doc.get("sentences", []):
            for tok in s.get("tokens", []):
                for m in tok.get("morphemes", []):
                    # 중첩된 딕셔너리 구조 처리
                    m_obj = m.get("lemma") or m.get("text") or m.get("morph") or ""
                    if isinstance(m_obj, dict):
                        lemma = m_obj.get("content", "")
                    else:
                        lemma = str(m_obj)

                    tag   = m.get("tag")   or m.get("pos")  or ""

                    # 문장 부호(태그가 'S'로 시작)는 분석에서 제외
                    if not tag.upper().startswith('S'):
                        if lemma: out.append({"morph": lemma, "tag": tag})
        if out: return out
    except Exception:
        pass

    # Bareun 구버전 또는 일반적인 형태소 분석기 구조
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

def parse_for_llm(bareun_output: Any) -> Tuple[List[Dict[str,str]], str]:
    """
    LLM 프롬프트에 사용될 초기 분석 결과와 휴리스틱 주석을 생성
    반환:
      morph_list: [{'morph','tag'}...]
      pos_line: '나/NP 는/JX 어제/MAG ...' 형태의 한 줄 요약
    """
    morph_list = _best_effort_parse(bareun_output)
    pos_line = " ".join(f"{m['morph']}/{m.get('tag','')}" for m in morph_list)
    return morph_list, pos_line

def generate_heuristic_annotations(morph_list: List[Dict[str, str]]) -> str:
    """형태소 분석 결과에 휴리스틱 규칙을 적용하여 LLM에게 제공할 주석 문자열을 생성"""
    annotations = []
    for i, item in enumerate(morph_list):
        morph = item.get("morph", "")
        tag = (item.get("tag") or "").upper()
        
        if not morph:
            continue

        heuristic_results = []

        # 문맥 기반 중의성 해소 ('이/그/저')
        if morph in ["이", "그", "저"]:
            if i + 1 < len(morph_list):
                next_tag = (morph_list[i + 1].get("tag") or "").upper()
                if next_tag.startswith("NN"):
                    heuristic_results.append("'지시 관형사'일 가능성 높음 (다음에 명사 위치)")
            if tag == "NP":
                 heuristic_results.append("'대명사'일 가능성 (bareunpy가 대명사로 분석)")

        # 태그 기반 휴리스틱
        if tag == 'IC':
            res = classify_interjection(morph)
            if res != 'Unknown': heuristic_results.append(f"감탄사 상세: {res}")
        if tag.startswith('J'):
            res = classify_particle_by_heuristic(morph)
            if res != 'Unknown': heuristic_results.append(f"조사 상세: {res}")
        if tag == 'EP':
            res = classify_ep_by_heuristic(morph)
            if res != 'Unknown': heuristic_results.append(f"선어말어미 상세: {res}")
        if tag == 'ETN':
            res = classify_nominal_ending(morph)
            if res != 'Unknown': heuristic_results.append(f"전성어미 상세: {res}")
        if tag == 'ETM':
            res = classify_adnominal_ending(morph)
            if res != 'Unknown': heuristic_results.append(f"전성어미 상세: {res}")
        if tag == 'EC':
            res = classify_all_ec(morph)
            if res != 'Unknown': heuristic_results.append(f"연결어미 상세: {res}")

        # 단어 기반 휴리스틱 (중의성 단어 제외)
        if morph not in ["이", "그", "저"]:
            res = classify_determiner(morph)
            if res != "Unknown":
                 heuristic_results.append(f"사전 기반 분석: {res}")

        # 파생어 휴리스틱
        res = classify_deriv_suffix(morph)
        if res != "Unknown":
            heuristic_results.append(f"파생어 분석: {res}")

        if heuristic_results:
            annotations.append(f"- {morph}: {', '.join(heuristic_results)}")

    return "\n".join(annotations) if annotations else "없음"