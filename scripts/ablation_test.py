# scripts/create_evaluation_data.py

import os
import sys
import json
import csv
import time
import argparse
from glob import glob
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
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test')
DEFAULT_INPUT_JSON = os.path.join(TEST_DATA_DIR, 'test_sentences.json')
DEFAULT_OUTPUT_CSV = os.path.join(TEST_DATA_DIR, 'evaluation_results.csv')
INPUT_PATTERN = os.path.join(TEST_DATA_DIR, 'test_sentences_*.json')
API_CALL_DELAY = float(os.getenv("LLM_CALL_DELAY", "5")) # 초
CSV_COLUMNS = [
    "ID", "Category", "Sentence",
    "1_Bareun",
    "2_LLM_Baseline",
    "3_LLM_with_Bareun",
    "4_LLM_with_Heuristics",
    "5_LLM_Full"
]
SCENARIO_COLUMNS = CSV_COLUMNS[3:]

def _is_error_value(value: str) -> bool:
    return isinstance(value, str) and "Error" in value

def _columns_to_process(row_data: dict, repair_only: bool) -> list:
    if not repair_only:
        return list(SCENARIO_COLUMNS)
    cols = []
    for col in SCENARIO_COLUMNS:
        val = row_data.get(col, "")
        if not val or _is_error_value(val):
            cols.append(col)
    return cols

# --------------------------------------------------------------------------
# 분석 함수
# --------------------------------------------------------------------------
def run_analysis(sentence, pretokenized="", heuristic_info="", decomposition_info="", system_prompt_override=None):
    """LLM 분석을 실행하고 예외를 처리하는 래퍼 함수"""
    try:
        main_md, md_sino, _ = analyze_school_grammar(
            sentence=sentence,
            pretokenized=pretokenized,
            heuristic_info=heuristic_info,
            decomposition_info=decomposition_info,
            system_prompt_override=system_prompt_override
        )
        return f"{main_md}\n\n{md_sino}".strip()
    except Exception as e:
        return f"Analysis Error: {e}"
    finally:
        time.sleep(API_CALL_DELAY)

def _load_sentences(input_path):
    print(f"--- 입력 데이터 로드: {input_path} ---")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            sentences_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 입력 파일({input_path})을 찾을 수 없습니다.")
        return None
    print(f"✅ 총 {len(sentences_data)}개의 문장 로드 완료.")
    return sentences_data

def _determine_output_path(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]
    prefix = "test_sentences_"
    if base.startswith(prefix):
        suffix = base[len(prefix):] or "default"
        filename = f"evaluation_results_{suffix}.csv"
    else:
        filename = os.path.basename(DEFAULT_OUTPUT_CSV)
    return os.path.join(TEST_DATA_DIR, filename)

def _collect_input_files(target_suffixes=None):
    if target_suffixes:
        lower_targets = [s.lower() for s in target_suffixes]
        if any(t in ("all", "*") for t in lower_targets):
            target_suffixes = None  # fallback to default behavior
        else:
            files = []
            seen = set()
            for suffix in target_suffixes:
                key = suffix.lower()
                if key in seen:
                    continue
                seen.add(key)
                if key in ("default", "base", ""):
                    candidate = DEFAULT_INPUT_JSON
                else:
                    candidate = os.path.join(TEST_DATA_DIR, f"test_sentences_{suffix}.json")
                if os.path.exists(candidate):
                    files.append(candidate)
                else:
                    print(f"⚠️ 지정한 파일을 찾을 수 없습니다: {candidate}")
            return files

    matched_files = sorted(glob(INPUT_PATTERN))
    if matched_files:
        return matched_files
    if os.path.exists(DEFAULT_INPUT_JSON):
        return [DEFAULT_INPUT_JSON]
    return []

def _load_existing_csv(csv_path):
    rows = []
    lookup = {}
    if not os.path.exists(csv_path):
        return rows, lookup
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {col: row.get(col, "") for col in CSV_COLUMNS}
            rows.append(normalized)
            row_id = normalized.get("ID", "")
            if row_id:
                lookup[row_id] = normalized
    return rows, lookup

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation CSVs from test_sentences JSON files.")
    parser.add_argument(
        "--target",
        help="처리할 접미사 (예: N 또는 N,JOSA). ALL 입력 시 전체 처리.",
        default=None,
    )
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="기존 CSV에서 오류난 셀만 다시 호출하여 채웁니다.",
    )
    return parser.parse_args()

def _resolve_targets(args):
    if args.target:
        return [part.strip() for part in args.target.split(",") if part.strip()]
    try:
        user_input = input("처리할 test_sentences 접미사를 입력하세요 (예: N 또는 N,JOSA, ALL=전체): ").strip()
    except EOFError:
        user_input = ""
    if not user_input:
        return None
    return [part.strip() for part in user_input.split(",") if part.strip()]

def process_dataset(input_path, output_path, tagger, repair_only=False):
    sentences_data = _load_sentences(input_path)
    if sentences_data is None:
        return

    if repair_only and not os.path.exists(output_path):
        print(f"❌ 기존 CSV({output_path})를 찾을 수 없습니다. --repair-only 모드에서는 필수입니다.")
        return

    # <<<--- 수정된 부분: 테스트용 프롬프트 로드 ---
    table_only_prompt_path = os.path.join(PROJECT_ROOT, 'chatbot_app', 'prompts', 'school_morph_table_only.md')
    try:
        with open(table_only_prompt_path, 'r', encoding='utf-8') as f:
            table_only_prompt = f.read()
        print("✅ 테스트용 '표 전용' 프롬프트 로드 완료.")
    except FileNotFoundError:
        print(f"❌ 테스트용 프롬프트({table_only_prompt_path})를 찾을 수 없습니다.")
        return
    # --- 수정된 부분 끝 ---/>

    print(f"--- 분석 및 평가 데이터 생성 시작: {output_path} ---")

    failures = []
    llm_retry_queue = []

    if repair_only:
        rows, row_lookup = _load_existing_csv(output_path)
    else:
        rows, row_lookup = [], {}

    for item in tqdm(sentences_data, desc="전체 분석 중"):
        sentence = item.get('sentence', '')
        item_id = item.get('id', '')
        row_data = row_lookup.get(item_id)
        if not row_data:
            row_data = {col: "" for col in CSV_COLUMNS}
            rows.append(row_data)
            row_lookup[item_id] = row_data

        row_data["ID"] = item_id
        row_data["Category"] = item.get('category', '')
        row_data["Sentence"] = sentence

        columns_to_process = _columns_to_process(row_data, repair_only)
        if repair_only and not columns_to_process:
            continue

        # --- 공통 준비 단계 ---
        decomposition_info = decompose_words(sentence=sentence)
        time.sleep(API_CALL_DELAY)

        try:
            bareun_json = tagger.tags([sentence]).as_json()
            morph_list, pos_line = parse_for_llm(bareun_json)
            heuristic_info = generate_heuristic_annotations(morph_list)
            bareun_result = pos_line or json.dumps(bareun_json, ensure_ascii=False)
        except Exception as e:
            morph_list, pos_line, heuristic_info = [], "", f"Bareun/Heuristic Error: {e}"
            bareun_result = f"Bareun Error: {e}"
            failures.append((item_id, "Bareun", str(e)))

        if "1_Bareun" in columns_to_process:
            row_data["1_Bareun"] = bareun_result

        scenario_configs = [
            ("2_LLM_Baseline", "LLM Baseline", {}),
            ("3_LLM_with_Bareun", "LLM with Bareun", {"pretokenized": pos_line}),
            ("4_LLM_with_Heuristics", "LLM with Heuristics", {"pretokenized": pos_line, "heuristic_info": heuristic_info}),
            ("5_LLM_Full", "LLM Full", {"pretokenized": pos_line, "heuristic_info": heuristic_info, "decomposition_info": decomposition_info}),
        ]

        for col_key, scen_name, kwargs in scenario_configs:
            if col_key not in columns_to_process:
                continue
            # <<<--- 수정된 부분: 테스트용 프롬프트를 인자로 전달 ---
            result = run_analysis(sentence, **kwargs, system_prompt_override=table_only_prompt)
            # --- 수정된 부분 끝 ---/>
            row_data[col_key] = result
            if result.startswith("Analysis Error"):
                if "503" in result:
                    llm_retry_queue.append({
                        "id": item_id,
                        "scenario": scen_name,
                        "col_key": col_key,
                        "sentence": sentence,
                        "kwargs": kwargs,
                        "row_ref": row_data
                    })
                else:
                    failures.append((item_id, scen_name, result))

        tqdm.write(f"{item_id}: {sentence[:20]}... 처리 완료")

    if llm_retry_queue:
        print("\n🔁 503 오류가 발생한 문장을 다시 시도합니다...")
        for retry in tqdm(llm_retry_queue, desc="503 재시도 중"):
            # <<<--- 수정된 부분: 재시도 시에도 테스트용 프롬프트 전달 ---
            result = run_analysis(retry["sentence"], **retry["kwargs"], system_prompt_override=table_only_prompt)
            # --- 수정된 부분 끝 ---/>
            retry["row_ref"][retry["col_key"]] = result
            if result.startswith("Analysis Error"):
                failures.append((retry["id"], retry["scenario"], result))

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row_data in rows:
            writer.writerow([row_data.get(col, "") for col in CSV_COLUMNS])

    print("\n✅ 작업 완료!")
    print(f"최종 결과가 {output_path} 파일에 저장되었습니다.")
    if failures:
        print("\n⚠️ 실패한 항목 요약:")
        for sent_id, scenario, message in failures:
            print(f"- {sent_id} [{scenario}]: {message}")
    else:
        print("\n🎉 모든 시나리오가 성공했습니다.")

# --------------------------------------------------------------------------
# 메인 실행 함수
# --------------------------------------------------------------------------
def main():
    args = _parse_args()
    target_suffixes = _resolve_targets(args)
    input_files = _collect_input_files(target_suffixes)
    if not input_files:
        print("❌ data/test 디렉터리에 처리할 test_sentences JSON 파일이 없습니다.")
        return

    tagger = Tagger(settings.BAREUN_API_KEY, 'api.bareun.ai', 443)
    for input_path in input_files:
        output_path = _determine_output_path(input_path)
        print(f"\n=== {os.path.basename(input_path)} -> {os.path.basename(output_path)} ===")
        process_dataset(input_path, output_path, tagger, repair_only=args.repair_only)

if __name__ == '__main__':
    main()
