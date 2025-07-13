# from settings import API_KEY, HOST, PORT    #상대경로
import sys
from config.settings import BAREUN_API_KEY, HOST, BAREUN_PORT   #절대경로
import google.protobuf.text_format as tf
from bareunpy import Tagger

#
# you can API-KEY from https://bareun.ai/
# 아래에 "https://bareun.ai/"에서 이메일 인증 후 발급받은 API KEY("koba-...")를 입력해주세요. "로그인-내정보 확인"
tagger = Tagger(BAREUN_API_KEY, HOST,BAREUN_PORT)


# # print results.
# res = tagger.tags(["안녕하세요.", "반가워요!"])

# # get protobuf message.
# # 전체 분석 결과를 뽑아냄
# m = print(res.as_json_str())
# print(m)


def analyze_text(text):
    res=tagger.tags([text])
    # m = res.as_json_str()  #전체 분석 결과
    # return m
    pa=res.pos()
    print(pa)
    return pa
