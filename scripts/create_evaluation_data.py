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
    from chatbot_app.tagger import tagger # analyze_text 대신 tagger 객체를 직접 임포트
    from chatbot_app.llm_client import analyze_school_grammar
    from chatbot_app.school_morph import parse_for_hybrid
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
# 상수 정의
# --------------------------------------------------------------------------
INPUT_JSON_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_sentences.json')
BAREUN_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'bareun_cache.json')
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'evaluation_results.csv')


# --------------------------------------------------------------------------
# 메인 실행 함수
# --------------------------------------------------------------------------
def run_evaluation():
    # --- 1단계: Bareun 분석 및 캐시 생성 ---
    if not os.path.exists(BAREUN_CACHE_PATH):
        print(f"--- 1단계: Bareun 분석 및 캐시 생성 시작 ---")
        print(f"입력: {INPUT_JSON_PATH}")
        try:
            with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                sentences_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ 입력 파일({INPUT_JSON_PATH})을 찾을 수 없습니다.")
            return

        cached_data = []
        for item in tqdm(sentences_data, desc="1단계: Bareun 분석 중"):
            sentence = item.get('sentence', '')
            output1, posline_for_llm = "", ""
            try:
                # tagger.pos()를 직접 호출하도록 수정
                bareun_result_raw = tagger.pos(sentence)
                output1 = str(bareun_result_raw)
                _, _, posline_for_llm = parse_for_hybrid(bareun_result_raw)
            except Exception as e:
                output1 = f"Bareun Error: {e}"
            
            item['output1_bareun'] = output1
            item['posline_for_llm'] = posline_for_llm
            cached_data.append(item)
        
        with open(BAREUN_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 1단계 완료. Bareun 분석 결과가 {BAREUN_CACHE_PATH}에 저장되었습니다.")
    else:
        print(f"--- 1단계: 이미 생성된 Bareun 캐시({BAREUN_CACHE_PATH})를 사용합니다. ---")

    # --- 2단계: LLM 분석 ---
    print(f"--- 2단계: LLM 분석 시작 ---")
    try:
        with open(BAREUN_CACHE_PATH, 'r', encoding='utf-8') as f:
            llm_input_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 캐시 파일({BAREUN_CACHE_PATH})을 찾을 수 없습니다. 1단계를 먼저 실행해야 합니다.")
        return

    print(f"✍️ 최종 출력 파일: {OUTPUT_CSV_PATH}")
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = [
            "ID", "Category", "Sentence", "Output1_Bareun", "Output2_LLM_Baseline",
            "Output3_LLM_with_Bareun", "Output4_LLM_with_Heuristics", "Output5_LLM_Full"
        ]
        writer.writerow(header)

        for item in tqdm(llm_input_data, desc="2단계: LLM 분석 중"):
            sentence_id = item.get('id', '')
            category = item.get('category', '')
            sentence = item.get('sentence', '')
            output1 = item.get('output1_bareun', '')
            posline_for_llm = item.get('posline_for_llm', '')

            outputs = {"2": "", "3": "", "4": "", "5": ""}
            analysis_scenarios = {
                "2": {"use_system_prompt": False},
                "3": {"pretokenized": posline_for_llm, "use_system_prompt": False},
                "4": {"use_system_prompt": True},
                "5": {"pretokenized": posline_for_llm, "use_system_prompt": True}
            }

            for i, params in analysis_scenarios.items():
                try:
                    # llm_client에 재시도 로직이 내장되어 있으므로 직접 호출합니다.
                    outputs[i] = analyze_school_grammar(sentence=sentence, **params)
                except Exception as e:
                    outputs[i] = f"LLM Error: {e}"
                time.sleep(5) # 개별 시나리오 간 5초 지연

            tqdm.write("\n" + "-" * 20)
            tqdm.write(f"ID: {sentence_id} | Category: {category}")
            tqdm.write(f"Sentence: {sentence}")
            tqdm.write("-" * 20)
            tqdm.write(f"[1] Bareun: {output1}")
            tqdm.write(f"[2] LLM Baseline: {(outputs['2'] or '')[:100]}...")
            tqdm.write(f"[3] LLM + Bareun: {(outputs['3'] or '')[:100]}...")
            tqdm.write(f"[4] LLM + Heuristics: {(outputs['4'] or '')[:100]}...")
            tqdm.write(f"[5] LLM Full: {(outputs['5'] or '')[:100]}...")
            tqdm.write("-" * 20)

            writer.writerow([
                sentence_id, category, sentence, output1, outputs["2"],
                outputs["3"], outputs["4"], outputs["5"]
            ])

    print("✅ 모든 작업이 완료되었습니다!")
    print(f"최종 결과가 {OUTPUT_CSV_PATH} 파일에 저장되었습니다.")

if __name__ == '__main__':
    run_evaluation()
