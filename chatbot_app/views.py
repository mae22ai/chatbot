# views.py 맨 위
from bleach.sanitizer import ALLOWED_TAGS as DEFAULT_TAGS, ALLOWED_ATTRIBUTES as DEFAULT_ATTRS
import os
import re
from django.conf import settings
import logging
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .services import process_chatbot_request
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

        # 서비스 레이어 호출
        result = process_chatbot_request(text, std_kr_entry)

        if not result['ok']:
            return JsonResponse(result, status=500)

        # HTML 변환 (Presentation Logic)
        result_md = result['markdown']
        logger.info(f"--- 최종 챗봇 답변 (Markdown) ---\n{result_md}\n------------------------------------")
        
        html_unsafe = markdown(result_md, extensions=["tables", "fenced_code"])
        html = bleach.clean(html_unsafe, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        
        return JsonResponse({
            'ok': True, 
            'html': html, 
            'markdown': result_md, 
            'mode': 'full-llm',
            'used_bareun': result.get('used_bareun'), 
            'bareun_error': result.get('bareun_error'),
            'issues_found': result.get('issues_found'),
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)

def index(request):
    return render(request, 'index.html')

def get_issue_document(request, issue_name):
    issues_dir = os.path.join(settings.BASE_DIR, 'data', 'issues')
    
    try:
        all_files = os.listdir(issues_dir)
        md_files = [f for f in all_files if f.endswith('.md')]
        file_map = {os.path.splitext(f)[0]: f for f in md_files}
    except FileNotFoundError:
        return JsonResponse({'ok': False, 'error': '문서 디렉토리를 찾을 수 없습니다.'}, status=500)
    
    filename = file_map.get(issue_name)
    if not filename:
        return JsonResponse({'ok': False, 'error': '해당 이슈 문서를 찾을 수 없습니다.'}, status=404)

    file_path = os.path.join(issues_dir, filename)
    
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
