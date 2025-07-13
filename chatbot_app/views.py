from django.shortcuts import render
from .openai_client import summarize_analysis
from .tagger import analyze_text
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.http import HttpResponse


@api_view(['POST'])
def chatbot_view(request):
    data=request.data
    text=data.get('text','')
    if not text:
        return JsonResponse({'error': '분석할 문장을 입력하세요'}, status=400)
    
    #baruen으로  형태소 분석
    analysis_result=analyze_text(text)
    #OpenAI로 응답 생성
    response=summarize_analysis(analysis_result)

    # return HttpResponse(response, content_type='text/plain; charset=utf-8')
    response = response.replace('\n', '<br>')
    return JsonResponse({'response': response})


def index(request):
    return render(request, 'index.html')