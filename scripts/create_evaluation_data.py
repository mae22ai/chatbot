# scripts/create_evaluation_data.py

import os
import sys
import json
import csv
import time
import django
from tqdm import tqdm

# --------------------------------------------------------------------------
# Django 환경 설정
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

try:
    django.setup()
except Exception as e:
    print(f"❌ Django 설정 중 오류 발생: {e}")
    sys.exit(1)

# Django 환경 설정 후 임포트
try:
    from django.conf import settings
    from chatbot_app.tagger import Tagger
    from chatbot_app.llm_client import analyze_school_grammar, decompose_words
    from chatbot_app.school_morph import parse_for_llm, generate_heuristic_annotations
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
# 상수 정의
# --------------------------------------------------------------------------
INPUT_JSON_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_sentences_demo.json')
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'evaluation_results_demo2.csv')
API_CALL_DELAY = 5 # 초

# --------------------------------------------------------------------------
# 분석 함수
# --------------------------------------------------------------------------
def run_analysis(sentence, pretokenized="", heuristic_info="", decomposition_info=""):
    """LLM 분석을 실행하고 예외를 처리하는 래퍼 함수"""
    try:
        main_md, md_sino, _ = analyze_school_grammar(
            sentence=sentence,
            pretokenized=pretokenized,
            heuristic_info=heuristic_info,
            decomposition_info=decomposition_info
        )
        return f"{main_md}\n\n{md_sino}".strip()
    except Exception as e:
        return f"Analysis Error: {e}"
    finally:
        time.sleep(API_CALL_DELAY)

# --------------------------------------------------------------------------
# 메인 실행 함수
# --------------------------------------------------------------------------
def main():
    print(f"--- 입력 데이터 로드 --- ")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            sentences_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 입력 파일({INPUT_JSON_PATH})을 찾을 수 없습니다.")
        return
    print(f"✅ 총 {len(sentences_data)}개의 문장 로드 완료.")

    print(f"--- 분석 및 평가 데이터 생성 시작 ---")
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = [
            "ID", "Category", "Sentence", 
            "1_LLM_Baseline", "2_LLM_with_Bareun", 
            "3_LLM_with_Heuristics", "4_LLM_Full"
        ]
        writer.writerow(header)

        tagger = Tagger(settings.BAREUN_API_KEY, 'api.bareun.ai', 443)

        for item in tqdm(sentences_data, desc="전체 분석 중"):
            sentence = item.get('sentence', '')
            
            # --- 공통 준비 단계 ---
            # 단어 분해
            decomposition_info = decompose_words(sentence=sentence)
            time.sleep(API_CALL_DELAY)

            # Bareun 분석 및 휴리스틱 생성
            try:
                bareun_json = tagger.tags([sentence]).as_json()
                morph_list, pos_line = parse_for_llm(bareun_json)
                heuristic_info = generate_heuristic_annotations(morph_list)
            except Exception as e:
                morph_list, pos_line, heuristic_info = [], "", f"Bareun/Heuristic Error: {e}"

            # --- 시나리오별 분석 실행 ---
            # 1. LLM 단독
            scen1 = run_analysis(sentence)

            # 2. LLM + Bareun
            scen2 = run_analysis(sentence, pretokenized=pos_line)

            # 3. LLM + Bareun + Heuristics
            scen3 = run_analysis(sentence, pretokenized=pos_line, heuristic_info=heuristic_info)

            # 4. LLM + Bareun + Heuristics + Decompose (Full)
            scen4 = run_analysis(sentence, pretokenized=pos_line, heuristic_info=heuristic_info, decomposition_info=decomposition_info)

            writer.writerow([
                item.get('id', ''), item.get('category', ''), sentence,
                scen1, scen2, scen3, scen4
            ])
            tqdm.write(f"{item.get('id')}: {sentence[:20]}... 완료")

    print("\n✅ 모든 작업이 완료되었습니다!")
    print(f"최종 결과가 {OUTPUT_CSV_PATH} 파일에 저장되었습니다.")

if __name__ == '__main__':
    main()
