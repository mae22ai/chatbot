# from settings import API_KEY, HOST, PORT    #상대경로
import sys
from config.settings import BAREUN_API_KEY, HOST, BAREUN_PORT   #절대경로
import google.protobuf.text_format as tf
from bareunpy import Tagger

#
# you can API-KEY from https://bareun.ai/
# 아래에 "https://bareun.ai/"에서 이메일 인증 후 발급받은 API KEY("koba-...")를 입력해주세요. "로그인-내정보 확인"
#tagger = Tagger(BAREUN_API_KEY, HOST,BAREUN_PORT)
tagger = Tagger(BAREUN_API_KEY, 'api.bareun.ai', 443)


# # print results.
# res = tagger.tags(["안녕하세요.", "반가워요!"])

# # get protobuf message.
# # 전체 분석 결과를 뽑아냄
# m = print(res.as_json_str())
# print(m)


def analyze_text(self, sentence: str):
    if not self.tagger:
        raise RuntimeError("Tagger가 초기화되지 않았습니다.")
    
    # 형태소 분석 실행
    result = self.tagger.pos(sentence)
    
    # 문장 부호(품사 태그가 'S'로 시작하는 경우) 제외
    filtered_result = [morph for morph in result if not morph[1].startswith('S')]
    
    print(filtered_result)
    return filtered_result
