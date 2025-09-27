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
from .tagger import analyze_text
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

        # 1) Bareun 초기 분석
        pretokenized, used_bareun, bareun_error = [], False, None
        try:
            pretokenized = analyze_text(text)
            used_bareun = True
        except Exception as e:
            bareun_error = str(e)

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
            processed_dir = os.path.join(settings.BASE_DIR, 'data', 'processed')
            issue_files = {os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith('.md')}
            
            preliminary_issues = {morph[0] for morph in pretokenized if morph[0] in issue_files}
            
            if any(morph[0] == '이' and morph[1] == 'VCP' for morph in pretokenized) and '이다' in issue_files:
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
            logger.warning(f"쟁점 컨텍스트 로딩 중 오류 발생: {e}")

        # 4) 2단계: 메인 LLM 분석 (단어 분해 및 쟁점 컨텍스트 주입)
        try:
            main_md = analyze_school_grammar(
                sentence=text, std_kr_entry=std_kr_entry,
                pretokenized=pretokenized if used_bareun else "",
                issue_context=issue_context,
                decomposition_info=decomposition_info
            )
        except Exception as e:
            logger.critical(f"메인 LLM 분석 실패: {e}", exc_info=True)
            return JsonResponse({
                'ok': False, 'error': f'LLM 오류: {e}', 'mode': 'error',
                'used_bareun': used_bareun, 'bareun_error': bareun_error
            }, status=500)

        # 5) 한자어 분석 (메인 분석 결과에서 명사 추출)
        nouns = []
        try:
            noun_pattern = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(?:명사|대명사)", re.MULTILINE | re.UNICODE)
            nouns = noun_pattern.findall(main_md)
            nouns = sorted(list(set([n.replace('-', '').strip() for n in nouns])))
        except Exception as e:
            logger.warning(f"명사 추출 중 오류 발생: {e}")

        md_sino = analyze_sino_korean(nouns=nouns)
        result_md = f"{main_md}\n\n{md_sino}".strip()

        # 6) 최종 쟁점 탐지 (버튼 생성용)
        issues_found = []
        try:
            processed_dir = os.path.join(settings.BASE_DIR, 'data', 'processed')
            issue_files = [os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith('.md')]
            
            detected_issues_str = detect_grammatical_issue(
                analysis_markdown=main_md,
                issue_list=issue_files
            )
            if detected_issues_str and detected_issues_str != "없음":
                issues_found = [issue.strip() for issue in detected_issues_str.split(',')]
            
            logger.info(f"쟁점 탐지 LLM 결과: '{detected_issues_str}' -> 파싱 후: {issues_found}")

        except Exception as e:
            logger.warning(f"지능형 쟁점 탐지 중 오류 발생: {e}")

        # 7) 최종 결과 반환
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