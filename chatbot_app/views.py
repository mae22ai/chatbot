# chatbot_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view

from .llm_client import analyze_school_grammar, analyze_compound_block
from .tagger import analyze_text
from .school_morph import build_sections_from_bareun, parse_for_hybrid  # ★ 추가

from markdown import markdown
import bleach

# (생략) ALLOWED_TAGS/ALLOWED_ATTRS 그대로

@api_view(['POST'])
def chatbot_view(request):
    text = (request.data.get('text') or "").strip()
    if not text:
        return JsonResponse({'ok': False, 'error': '분석할 문장을 입력하세요.'}, status=400)

    korea_u_form = (request.data.get('korea_u_form') or "").strip()
    std_kr_entry = (request.data.get('std_kr_entry') or "").strip()

    # 1) Bareun
    pretokenized, used_bareun, bareun_error = "", False, None
    try:
        pretokenized = analyze_text(text)   # JSON/문자열
        used_bareun = True
    except Exception as e:
        bareun_error = str(e)

    # 2) 하이브리드
    if used_bareun:
        md_core, ok = build_sections_from_bareun(pretokenized)
        morphs, candidates, posline = parse_for_hybrid(pretokenized)
        if ok and md_core:
            try:
                md_comp = analyze_compound_block(
                    sentence=text,
                    korea_u_form=korea_u_form,
                    std_kr_entry=std_kr_entry,
                    pretokenized=pretokenized,
                    candidates=candidates,   # ★ 후보 강제
                    posline=posline          # ★ POS 힌트
                )
                result_md = md_core + "\n\n" + md_comp
                html_unsafe = markdown(result_md, extensions=["tables", "fenced_code"])
                html = bleach.clean(html_unsafe, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
                return JsonResponse({
                    'ok': True,
                    'html': html,
                    'markdown': result_md,
                    'mode': 'hybrid',
                    'used_bareun': True,
                    'bareun_error': None,
                })
            except Exception:
                pass  # 아래 폴백 진행

    # 3) 폴백: 전체 LLM
    try:
        result_md = analyze_school_grammar(
            sentence=text,
            korea_u_form=korea_u_form,
            std_kr_entry=std_kr_entry,
            pretokenized=pretokenized if used_bareun else ""
        )
        html_unsafe = markdown(result_md, extensions=["tables", "fenced_code"])
        html = bleach.clean(html_unsafe, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        return JsonResponse({
            'ok': True,
            'html': html,
            'markdown': result_md,
            'mode': 'full-llm',
            'used_bareun': used_bareun,
            'bareun_error': None if used_bareun else bareun_error,
        })
    except Exception as e:
        return JsonResponse({
            'ok': False,
            'error': f'LLM 오류: {e}',
            'mode': 'error',
            'used_bareun': used_bareun,
            'bareun_error': bareun_error
        }, status=500)

def index(request):
    return render(request, 'index.html')
