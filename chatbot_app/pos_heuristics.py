# chatbot_app/pos_heuristics.py
from typing import Tuple, Dict, List
import re


def _matches_variant(token: str, candidate: str) -> bool:
    """
    조사 사전에는 '이/가' 형태로 표현된 엔트리가 많으므로
    슬래시로 분리된 모든 변형을 비교한다.
    """
    token = token.strip()
    variants = [part.strip() for part in candidate.split("/") if part.strip()]
    if not variants:
        return False
    return token in variants


# ──────────────────────────────────────────────
# 관형사(Determiner) 휴리스틱
# ──────────────────────────────────────────────
import json
import os
from django.conf import settings

# Load heuristics from JSON file
HEURISTICS_PATH = os.path.join(settings.BASE_DIR, 'chatbot_app', 'data', 'heuristics.json')

try:
    with open(HEURISTICS_PATH, 'r', encoding='utf-8') as f:
        HEURISTICS_DATA = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load heuristics.json: {e}")
    HEURISTICS_DATA = {}

adn_determiners = HEURISTICS_DATA.get("adn_determiners", {})
particle_categories = HEURISTICS_DATA.get("particle_categories", {"case": {}, "접속조사": [], "보조사": []})
ep_categories = HEURISTICS_DATA.get("ep_categories", {})
nominal_ending_list = HEURISTICS_DATA.get("nominal_ending_list", [])
adnominal_endings = HEURISTICS_DATA.get("adnominal_endings", [])
auxiliary_ec_list = HEURISTICS_DATA.get("auxiliary_ec_list", [])
coordinative_ec_categories = HEURISTICS_DATA.get("coordinative_ec_categories", {})
subordinate_ec_list = HEURISTICS_DATA.get("subordinate_ec_list", [])
interjection_categories = HEURISTICS_DATA.get("interjection_categories", {})
noun_deriv_suffixes = HEURISTICS_DATA.get("noun_deriv_suffixes", [])
verb_deriv_suffixes = HEURISTICS_DATA.get("verb_deriv_suffixes", [])
adj_deriv_suffixes = HEURISTICS_DATA.get("adj_deriv_suffixes", [])


def classify_determiner(word: str) -> str:
    """입력된 단어가 관형사 목록에 있으면 분류 반환"""
    word = word.strip()
    for cat, word_list in adn_determiners.items():
        if word in word_list:
            return f"관형사 ({cat})"
    return "Unknown"


def classify_particle_by_heuristic(particle: str) -> str:
    """입력된 조사 문자열을 격조사/접속조사/보조사로 분류"""
    particle = particle.strip()
    
    # Case markers
    for case_type, case_list in particle_categories.get("case", {}).items():
        for entry in case_list:
            if _matches_variant(particle, entry):
                return f"격조사 ({case_type})"
    
    # Conjunction particles
    for entry in particle_categories.get("접속조사", []):
        if _matches_variant(particle, entry):
            return "접속조사"
            
    # Auxiliary particles
    for entry in particle_categories.get("보조사", []):
        if _matches_variant(particle, entry):
            return "보조사"
            
    return "Unknown"


def classify_ep_by_heuristic(ep: str) -> str:
    """입력된 선어말어미 문자열을 카테고리별로 분류"""
    ep = ep.strip()
    for cat, ep_list in ep_categories.items():
        if ep in ep_list:
            return f"선어말어미 ({cat})"
    return "Unknown"


def classify_nominal_ending(etn: str) -> str:
    """입력된 명사형 전성어미(etn)를 분류"""
    etn = etn.strip()
    if etn in nominal_ending_list:
        return "전성어미 (명사형)"
    return "Unknown"


def classify_adnominal_ending(ending: str) -> str:
    """입력된 관형사형 어미를 분류"""
    ending = ending.strip()
    if ending in adnominal_endings:
        return "전성어미 (관형사형)"
    return "Unknown"


def classify_all_ec(ec: str) -> str:
    """연결어미(ec)를 세부적으로 분류"""
    ec = ec.strip()
    if ec in auxiliary_ec_list:
        return "연결어미 (보조적)"
    for cat, lst in coordinative_ec_categories.items():
        if ec in lst:
            return f"연결어미 (대등적-{cat})"
    if ec in subordinate_ec_list:
        return "연결어미 (종속적)"
    return "Unknown"


def classify_interjection(word: str) -> str:
    """입력된 단어가 감탄사 목록에 있으면 분류 반환"""
    word = word.strip()
    for cat, word_list in interjection_categories.items():
        if word in word_list:
            return f"감탄사 ({cat})"
    return "Unknown"


def classify_deriv_suffix(word: str) -> str:
    """입력된 단어가 파생접미사로 끝나는지 판별하여 품사 추정"""
    word = word.strip()
    for suf in noun_deriv_suffixes:
        if word.endswith(suf):
            stem = word[:-len(suf)]
            if stem: return f"명사 (파생: {stem}+{suf})"
    for suf in verb_deriv_suffixes:
        if word.endswith(suf):
            stem = word[:-len(suf)]
            if stem: return f"동사 (파생: {stem}+{suf})"
    for suf in adj_deriv_suffixes:
        if word.endswith(suf):
            stem = word[:-len(suf)]
            if stem: return f"형용사 (파생: {stem}+{suf})"
    return "Unknown"
