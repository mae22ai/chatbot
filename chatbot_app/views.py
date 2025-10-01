# views.py 맨 위
from bleach.sanitizer import ALLOWED_TAGS as DEFAULT_TAGS, ALLOWED_ATTRIBUTES as DEFAULT_ATTRS
import os
import re
from django.conf import settings
import logging
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .llm_client import analyze_school_grammar, detect_grammatical_issue, analyze_sino_korean, decompose_words
from .tagger import Tagger
from .school_morph import parse_for_llm, generate_heuristic_annotations
import traceback
from markdown import markdown
import bleach

logger = logging.getLogger(__name__)

ALLOWED_TAGS = list(DEFAULT_TAGS) + [
    "p", "span", "table", "thead", "tbody", "tr", "th", "td",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "strong", "em", "code", "pre", "br"
]
ALLOWED_ATTRS = {**DEFAULT_ATTRS, "span": ["class"], "td": ["colspan", "rowspan"]}

@api_view(['POST'])
def chatbot_view(request):
    try:
        text = (request.data.get('text') or "").strip()
        if not text:
            return JsonResponse({'ok': False, 'error': '분석할 문장을 입력하세요.'}, status=400)

        std_kr_entry = (request.data.get('std_kr_entry') or "").strip()

        # 1) Bareun 초기 분석 (설정값에 따라 분기)
        pos_line, heuristic_info = "", ""
        used_bareun, bareun_error = False, None
        morph_list = []

        if settings.USE_BAREUN_ANALYZER:
            try:
                tagger = Tagger(settings.BAREUN_API_KEY, 'api.bareun.ai', 443)
                bareun_json = tagger.tags([text]).as_json()
                morph_list, pos_line = parse_for_llm(bareun_json)
                used_bareun = True
                
                if settings.USE_HEURISTICS:
                    heuristic_info = generate_heuristic_annotations(morph_list)

            except Exception as e:
                bareun_error = str(e)
        else:
            pos_line = "(Bareun 분석기 비활성화됨)"

        # 2) 1단계: 단어 분해 LLM 호출
        decomposition_info = ""
        try:
            decomposition_result = decompose_words(sentence=text)
            if decomposition_result and decomposition_result != "없음":
                decomposition_info = decomposition_result
        except Exception as e:
            logger.warning(f"단어 분해 LLM 실패: {e}")

        # 3) 쟁점 사전 탐지 및 컨텍스트 로딩
        issue_context = ""
        try:
            if 'morph_list' in locals() and morph_list:
                processed_dir = os.path.join(settings.BASE_DIR, 'data', 'processed')
                issue_files = {os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith('.md')}
                
                # 더 안전한 코드로 수정 및 디버깅 로그 추가
                preliminary_issues = set()
                for morph in morph_list:
                    if not isinstance(morph, dict):
                        logger.warning(f"morph_list에 dict가 아닌 항목 발견: {morph}")
                        continue
                    
                    morph_text = morph.get('morph')
                    if not isinstance(morph_text, str):
                        logger.warning(f"morph 딕셔너리에 문자열이 아닌 'morph' 값 발견: {morph_text}")
                        continue

                    if morph_text in issue_files:
                        preliminary_issues.add(morph_text)

                if any(morph.get('morph') == '이' and morph.get('tag') == 'VCP' for morph in morph_list if isinstance(morph, dict)) and '이다' in issue_files:
                    preliminary_issues.add('이다')

                if preliminary_issues:
                    context_parts = []
                    for issue in preliminary_issues:
                        file_path = os.path.join(processed_dir, f"{issue}.md")
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                context_parts.append(f.read())
                    issue_context = "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"쟁점 컨텍스트 로딩 중 결정적 오류 발생: {e}", exc_info=True)


        # 4) 2단계: 메인 LLM 통합 분석 (최적화)
        try:
            main_md, md_sino, issues_found = analyze_school_grammar(
                sentence=text, std_kr_entry=std_kr_entry,
                pretokenized=pos_line if used_bareun else "",
                issue_context=issue_context,
                decomposition_info=decomposition_info,
                heuristic_info=heuristic_info
            )
            result_md = f"{main_md}\n\n{md_sino}".strip()

        except Exception as e:
            logger.critical(f"메인 LLM 분석 실패: {e}", exc_info=True)
            return JsonResponse({
                'ok': False, 'error': f'LLM 오류: {e}', 'mode': 'error',
                'used_bareun': used_bareun, 'bareun_error': bareun_error
            }, status=500)

        # 5) 최종 결과 반환 (한자어 분석 및 쟁점 탐지는 LLM이 동시 수행)
        logger.info(f"--- 최종 챗봇 답변 (Markdown) ---\n{result_md}\n------------------------------------")
        html_unsafe = markdown(result_md, extensions=["tables", "fenced_code"])
        html = bleach.clean(html_unsafe, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        
        return JsonResponse({
            'ok': True, 'html': html, 'markdown': result_md, 'mode': 'full-llm',
            'used_bareun': used_bareun, 'bareun_error': None if used_bareun else bareun_error,
            'issues_found': issues_found,
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)

def index(request):
    return render(request, 'index.html')

def get_issue_document(request, issue_name):
    processed_dir = os.path.join(settings.BASE_DIR, 'data', 'processed')
    
    try:
        all_files = os.listdir(processed_dir)
        md_files = [f for f in all_files if f.endswith('.md')]
        file_map = {os.path.splitext(f)[0]: f for f in md_files}
    except FileNotFoundError:
        return JsonResponse({'ok': False, 'error': '문서 디렉토리를 찾을 수 없습니다.'}, status=500)
    
    filename = file_map.get(issue_name)
    if not filename:
        return JsonResponse({'ok': False, 'error': '해당 이슈 문서를 찾을 수 없습니다.'}, status=404)

    file_path = os.path.join(processed_dir, filename)
    
    if not os.path.exists(file_path):
        return JsonResponse({'ok': False, 'error': '해당 이슈 문서를 찾을 수 없습니다.'}, status=404)
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html_unsafe = markdown(content, extensions=["tables", "fenced_code"])
        html = bleach.clean(html_unsafe, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        
        return JsonResponse({'ok': True, 'markdown': content, 'html': html})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'문서 처리 중 오류 발생: {e}'}, status=500)
