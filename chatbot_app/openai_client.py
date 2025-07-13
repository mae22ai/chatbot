from config.settings import OPENAI_API_KEY
from openai import OpenAI, OpenAIError, APIConnectionError, RateLimitError, APIStatusError
from django.http import JsonResponse
from rest_framework.decorators import api_view

# Create your views here.
client=OpenAI(api_key=OPENAI_API_KEY)

def summarize_analysis(analysis_result):
    prompt = f"""다음은 한국어 형태소 분석 결과입니다:
    {analysis_result}

    이 결과를 사람이 읽기 좋게, 간결한 형태로 정리해주세요.
    
    '안녕하세요' 출력 예시: 

    '안녕하' : VA
    '시' : EP
    '어요' : EF

    """
    
    try:
        response=client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "너는 형태소 분석을 돕는 유능하고 친절한 한국어 챗봇이야."},
                {"role": "user", "content": prompt}
            ]   
        )
        completion=response.choices[0].message.content
        return completion
    except(APIStatusError, RateLimitError, APIStatusError) as e:
        print(f"API ERROR: {e.message}")
        raise
    except OpenAIError as e:
        print(f"OpenAI Error: {e.message}")
        raise
