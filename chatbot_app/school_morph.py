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

HYPHEN_TOKENS = {"-", "‐", "‑"}


def _best_effort_parse(bareun_output: Any) -> List[Dict[str, Any]]:
    """
    Bareun JSON/str을 어절 단위로 구조화하여 반환.
    반환 형식: [{'text': '어절', 'morphemes': [{'morph': '..', 'tag': '..'}, ...]}, ...]
    """
    doc = bareun_output
    if isinstance(doc, str):
        try: doc = json.loads(doc)
        except Exception: return []

    out = []
    
    # Bareun v3.2.0 이상 구조
    try:
        for s in doc.get("sentences", []):
            for tok in s.get("tokens", []):
                ejeol_text = tok.get("text", "")
                morphemes = []
                for m in tok.get("morphemes", []):
                    # 중첩된 딕셔너리 구조 처리
                    m_obj = m.get("lemma") or m.get("text") or m.get("morph") or ""
                    if isinstance(m_obj, dict):
                        lemma = m_obj.get("content", "")
                    else:
                        lemma = str(m_obj)

                    tag = m.get("tag") or m.get("pos") or ""
                    
                    # 모든 형태소 포함 (문장부호 포함)
                    if lemma:
                        morphemes.append({"morph": lemma, "tag": tag})
                
                if morphemes:
                    out.append({"text": ejeol_text, "morphemes": morphemes})
        if out: return out
    except Exception:
        pass

    # Bareun 구버전 또는 일반적인 형태소 분석기 구조 (Flat structure)
    # 이 경우 어절 정보를 복원하기 어려우므로 Flat list를 가짜 어절로 포장
    try:
        flat_morphs = []
        for m in doc.get("morphs", []):
            lemma = m.get("lemma") or m.get("text") or m.get("morph") or ""
            tag   = m.get("tag")   or m.get("pos")  or ""
            if lemma: flat_morphs.append({"morph": lemma, "tag": tag})
        
        if flat_morphs:
            # 어절 정보가 없으므로 전체를 하나의 그룹으로 묶거나 개별 처리
            # 여기서는 호환성을 위해 단순 리스트 반환 (호출부에서 처리 필요할 수 있음)
            # 하지만 _best_effort_parse의 시그니처를 맞추기 위해 
            # 각 형태소를 하나의 어절로 취급하는 fallback
            return [{"text": m['morph'], "morphemes": [m]} for m in flat_morphs]
    except Exception:
        pass

    return []

def parse_for_llm(bareun_output: Any) -> Tuple[List[Dict[str,str]], str]:
    """
    LLM 프롬프트에 사용될 초기 분석 결과와 휴리스틱 주석을 생성
    반환:
      morph_list: [{'morph','tag'}...] (Flat list for heuristics & issue loading)
      pos_line: '나/NP 는/JX  학교/NNG ...' (Ejeol-aware string for LLM)
    """
    ejeol_list = _best_effort_parse(bareun_output)
    
    # 1. Flat list 생성 (기존 로직 호환성 및 휴리스틱용)
    morph_list = []
    for ejeol in ejeol_list:
        morph_list.extend(ejeol['morphemes'])

    # 2. Ejeol-aware pos_line 생성 (띄어쓰기 보존)
    # 예: "나/NP+는/JX  학교/NNG+에/JKB  간다/VV+ㄴ다/EF"
    pos_segments = []
    for ejeol in ejeol_list:
        morphs_str = "+".join(f"{m['morph']}/{m.get('tag','')}" for m in ejeol['morphemes'])
        pos_segments.append(morphs_str)
    
    pos_line = "  ".join(pos_segments) # 어절 사이는 공백 2개로 구분하여 시각적 명확성 확보

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
