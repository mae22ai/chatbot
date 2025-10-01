# chatbot_app/pos_heuristics.py
from typing import Tuple, Dict, List
import re

# ──────────────────────────────────────────────
# 관형사(Determiner) 휴리스틱
# ──────────────────────────────────────────────
adn_determiners = {
    "지시 관형사": [
        "이", "그", "저", "요", "어느", "여느", "여늬", "웬", "무슨", "어떤", "어인", "이런",
        "그런", "저런", "요런", "이딴", "그딴", "저딴", "이까짓", "그까짓", "저까짓",
        "이깟", "그깟", "저깟", "요만", "이만", "그만", "저만"
    ],
    "수 관형사": [
        "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열",
        "스무", "수십", "수백", "수천", "수만", "수십만", "억만", "억조", "억천만",
        "여남", "서너", "두서너", "두어", "여럿", "몇", "몇몇", "첫", "둘째", "셋째",
        "일이", "삼사", "사오", "육칠", "칠팔", "팔구"
    ],
    "성상 관형사": [
        "온갖", "갖은", "각", "모든", "모모", "애먼", "허튼", "헌", "오랜", "옛",
        "별", "별별", "별의별", "빌어먹을", "염병할", "젠장맞을", "젠장칠", "난장맞을",
        "난장칠", "넨장맞을", "넨장칠", "제미붙을", "제밀할", "떡을할", "한다는", "한다하는",
        "몹쓸", "귀한", "바른", "아무런", "아무아무", "요런조런", "그런저런", "이런저런"
    ]
}

def classify_determiner(word: str) -> str:
    """입력된 단어가 관형사 목록에 있으면 분류 반환"""
    word = word.strip()
    for cat, word_list in adn_determiners.items():
        if word in word_list:
            return f"관형사 ({cat})"
    return "Unknown"

# ──────────────────────────────────────────────
# 조사 분류 휴리스틱
# ──────────────────────────────────────────────
particle_categories = {
    "case": {
        "주격조사": ["이/가", "께서", "에서", "서"],
        "목적격": ["을/를", "께"],
        "보격조사": ["이/가"],
        "호격조사": ["아/야", "시여", "이시여"],
        "관형격조사": ["의"],
        "부사격조사": ["에", "서", "에서", "에게", "에게서", "보다", "로", "로서", "로써", "으로", "으로서", "으로써", "와", "과", "고", "라고", "하고", "한테", "한테서"],
        "서술격조사": ["이다"]
    },
    "접속조사": ["와", "과", "이랑", "랑", "하고", "이며", "며", "에", "에다"],
    "보조사": ["은", "는", "도", "만", "까지", "부터", "이나", "나", "밖에", "마다", "란", "이란", "뿐", "야", "이야", "대로",
              "조차", "마저", "야말로", "이야말로", "다가", "나마", "이나마", "커녕", "치고", "따라", "든지",
              "이든지", "이거나", "이건", "이라도", "깨나", "만큼", "을랑", "일랑", "마는", "든가", "이든가",
              "꺼정", "이사", "서껀", "말고", "라는", "이라는", "더러", "따라", "보고", "요", "니", "이니", "라야", "이라야", "인들", "처럼"]
}

def classify_particle_by_heuristic(particle: str) -> str:
    """입력된 조사 문자열을 격조사/접속조사/보조사로 분류"""
    particle = particle.strip()
    for case_type, case_list in particle_categories["case"].items():
        if particle in case_list:
            return f"격조사 ({case_type})"
    if particle in particle_categories["접속조사"]:
        return "접속조사"
    if particle in particle_categories["보조사"]:
        return "보조사"
    return "Unknown"

# ──────────────────────────────────────────────
# 선어말어미(EP) 휴리스틱
# ──────────────────────────────────────────────
ep_categories = {
    "높임": ["시", "으시"],
    "시제/상": ["었", "았", "겠", "았었", "었었", "더"],
}

def classify_ep_by_heuristic(ep: str) -> str:
    """입력된 선어말어미 문자열을 카테고리별로 분류"""
    ep = ep.strip()
    for cat, ep_list in ep_categories.items():
        if ep in ep_list:
            return f"선어말어미 ({cat})"
    return "Unknown"

# ──────────────────────────────────────────────
# 명사형 전성어미(ETN) 휴리스틱
# ──────────────────────────────────────────────
nominal_ending_list = ["음", "기", "ㅁ"]

def classify_nominal_ending(etn: str) -> str:
    """입력된 명사형 전성어미(etn)를 분류"""
    etn = etn.strip()
    if etn in nominal_ending_list:
        return "전성어미 (명사형)"
    return "Unknown"

# ──────────────────────────────────────────────
# 관형사형 어미(ADN: Adnominal) 휴리스틱
# ──────────────────────────────────────────────
adnominal_endings = ["을", "는", "은", "던", "ㄹ"]

def classify_adnominal_ending(ending: str) -> str:
    """입력된 관형사형 어미를 분류"""
    ending = ending.strip()
    if ending in adnominal_endings:
        return "전성어미 (관형사형)"
    return "Unknown"

# ──────────────────────────────────────────────
# 통합 연결어미(EC) 휴리스틱
# ──────────────────────────────────────────────
auxiliary_ec_list = ["아", "어", "게", "지", "고"]
coordinative_ec_categories = {
    "나열": ["고", "며"],
    "선택": ["거나", "든지", "나"],
    "대조": ["나", "지만"]
}
subordinate_ec_list = [
    "거든", "-고도", "-고서", "-고자", "-길래", "-느니", "-(으)니만큼", "-느라고",
    "-는다면", "-다가", "-다시피", "-더니", "-더라도", "-던데", "-던지", "-도록",
    "-든지", "-듯이", "-라", "-아다가", "-어다가", "-아/어도", "-아/어서", "-아/어야",
    "-아/어야지", "-았/었더라면", "-았/었으면", "-(으)니까", "-(으)되", "-(으)라고",
    "-(으)러", "-(으)려고", "-(으)려다가", "-(으)려면", "-(으)면", "-(으)면서",
    "-(으)므로", "-은/는데", "-은/는지", "-은들", "-을수록", "-을지", "-을지라도",
    "-음에도", "-자", "-자마자", "-자면"
]

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

# ──────────────────────────────────────────────
# 감탄사(Interjection) 휴리스틱
# ──────────────────────────────────────────────
interjection_categories = {
    "감정": ["와", "어머나", "아차", "아이고", "헉"],
    "응답": ["응", "네", "아니", "그래", "그럼"],
    "부름": ["야", "얘", "여보", "여보세요"],
    "습관적": ["음", "어", "자", "뭐", "허허", "흠"]
}

def classify_interjection(word: str) -> str:
    """입력된 단어가 감탄사 목록에 있으면 분류 반환"""
    word = word.strip()
    for cat, word_list in interjection_categories.items():
        if word in word_list:
            return f"감탄사 ({cat})"
    return "Unknown"

# ──────────────────────────────────────────────
# 파생접미사(Derivational Suffix) 휴리스틱
# ──────────────────────────────────────────────
noun_deriv_suffixes = ["기", "음", "ㅁ"]
verb_deriv_suffixes = ["하다", "되다", "이다", "히다", "시키다", "거리다", "추다", "받다", "당하다", "대다", "애다"]
adj_deriv_suffixes = ["답다", "스럽다", "롭다", "궂다", "나다", "쩍다", "되다", "다랗다", "맞다", "하다", "음직스럽다", "음직하다", "지다"]

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