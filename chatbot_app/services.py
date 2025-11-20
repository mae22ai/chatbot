import os
import logging
from django.conf import settings
from .llm_client import analyze_school_grammar
from .utils import bareun_client
from .school_morph import parse_for_llm, generate_heuristic_annotations

logger = logging.getLogger(__name__)

def process_chatbot_request(
    text: str, 
    std_kr_entry: str = "",
    use_bareun: bool = None,
    use_heuristics: bool = None,
    use_llm: bool = True
) -> dict:
    """
    챗봇 요청을 처리하는 메인 서비스 함수.
    Bareun 형태소 분석 -> 쟁점 탐지 -> LLM 분석 과정을 조율합니다.
    테스트를 위해 use_bareun, use_heuristics, use_llm 플래그로 동작을 제어할 수 있습니다.
    """
    
    # 설정값 결정 (인자가 None이면 settings 값 사용)
    effective_use_bareun = use_bareun if use_bareun is not None else settings.USE_BAREUN_ANALYZER
    effective_use_heuristics = use_heuristics if use_heuristics is not None else settings.USE_HEURISTICS

    # 1. Bareun 초기 분석
    pos_line, heuristic_info = "", ""
    used_bareun, bareun_error = False, None
    morph_list = []

    if effective_use_bareun:
        try:
            # Singleton 클라이언트 사용
            tagger = bareun_client.get_tagger()
            bareun_json = tagger.tags([text]).as_json()
            morph_list, pos_line = parse_for_llm(bareun_json)
            used_bareun = True
            
            if effective_use_heuristics:
                heuristic_info = generate_heuristic_annotations(morph_list)

        except Exception as e:
            bareun_error = str(e)
            logger.error(f"Bareun analysis failed: {e}")
    else:
        pos_line = "(Bareun 분석기 비활성화됨)"

    # 2. 쟁점 사전 탐지 및 컨텍스트 로딩
    # Smart Filtering: 흔한 단음절 형태소는 특정 품사일 때만 쟁점 문서를 로딩합니다.
    issue_context = ""
    try:
        if morph_list:
            issues_dir = os.path.join(settings.BASE_DIR, 'data', 'issues')
            if os.path.exists(issues_dir):
                issue_files = {os.path.splitext(f)[0] for f in os.listdir(issues_dir) if f.endswith('.md')}
                
                # 필터링 규칙: {형태소: [허용할 태그 접두사 목록]}
                # 예: '이'는 조사(J)나 서술격조사(VCP)일 때만 로딩
                SMART_FILTERS = {
                    "이": ["J", "VCP"],
                    "가": ["J"],
                    "을": ["J"],
                    "를": ["J"],
                    "의": ["J"],
                    "에": ["J"],
                    "로": ["J"],
                    "와": ["J"],
                    "과": ["J"],
                }

                preliminary_issues = set()
                for morph in morph_list:
                    if not isinstance(morph, dict):
                        continue
                    
                    morph_text = morph.get('morph')
                    morph_tag = (morph.get('tag') or "").upper()

                    if morph_text in issue_files:
                        # 필터링 규칙 적용
                        if morph_text in SMART_FILTERS:
                            allowed_prefixes = SMART_FILTERS[morph_text]
                            if not any(morph_tag.startswith(prefix) for prefix in allowed_prefixes):
                                continue # 태그가 일치하지 않으면 스킵
                        
                        preliminary_issues.add(morph_text)

                # '이다' 특수 처리 (Bareun이 '이'+'다'로 분석하는 경우 등)
                if '이다' in issue_files:
                     if any((m.get('morph') == '이' and m.get('tag') == 'VCP') or (m.get('morph') == '이다') for m in morph_list if isinstance(m, dict)):
                        preliminary_issues.add('이다')

                if preliminary_issues:
                    context_parts = []
                    for issue in preliminary_issues:
                        file_path = os.path.join(issues_dir, f"{issue}.md")
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                context_parts.append(f.read())
                    issue_context = "\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Issue context loading failed: {e}", exc_info=True)
        # 쟁점 로딩 실패는 치명적이지 않으므로 계속 진행

    # 3. 메인 LLM 통합 분석 (단어 분해 포함)
    if not use_llm:
        return {
            'ok': True,
            'markdown': "LLM analysis skipped.",
            'used_bareun': used_bareun,
            'bareun_error': None if used_bareun else bareun_error,
            'issues_found': [],
            'debug_info': {
                'pos_line': pos_line,
                'heuristic_info': heuristic_info,
                'issue_context_length': len(issue_context)
            }
        }

    try:
        main_md, md_sino, issues_found = analyze_school_grammar(
            sentence=text, 
            std_kr_entry=std_kr_entry,
            pretokenized=pos_line if used_bareun else "",
            issue_context=issue_context,
            heuristic_info=heuristic_info
        )
        result_md = f"{main_md}\n\n{md_sino}".strip()
        
        return {
            'ok': True,
            'markdown': result_md,
            'used_bareun': used_bareun,
            'bareun_error': None if used_bareun else bareun_error,
            'issues_found': issues_found,
        }

    except Exception as e:
        logger.critical(f"Main LLM analysis failed: {e}", exc_info=True)
        # 사용자에게 친절한 에러 메시지 제공
        error_msg = f"분석 중 오류가 발생했습니다. (상세: {str(e)})"
        if used_bareun:
             error_msg += " ※ 형태소 분석기는 정상 동작했습니다."
        
        return {
            'ok': False,
            'error': error_msg,
            'used_bareun': used_bareun,
            'bareun_error': bareun_error
        }
