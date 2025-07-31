import os
import fitz
from pdf2image import convert_from_path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, ImageOps
import numpy as np
import re
import json
from datetime import datetime
import pytesseract
import argparse

# -----------------------------
# 환경 변수 & API 설정
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdf")
JSON_DIR = os.path.join(BASE_DIR, "json")
TXT_DIR = os.path.join(BASE_DIR, "txt")
REF_TXT_DIR = os.path.join(BASE_DIR, "txt-ref")

USD_TO_KRW = 1390

PRICING = {
    "gemini-1.5-flash": {"input": 0.000018, "output": 0.000036},
    "gemini-1.5-pro": {"input": 0.000125, "output": 0.000375}
}

total_cost_usd = 0.0

# -----------------------------
# 프롬프트
# -----------------------------
BASE_PROMPT = """
다음 텍스트를 전처리해 주세요.

요구사항:
1. 띄어쓰기와 줄바꿈을 문맥에 맞게 수정
2. 문장부호를 한국어 맞춤법에 맞게 수정
3. 한자로만 써 있는 모든 한자어는 '한국어(漢字)' 형태로 변환. 예: '漢字가 있었다.' → '한자(漢字)가 있었다.'
4. 학술 논문 기호, 특수문자는 원형 보존
5. 불필요한 공백 제거
6. 논문 텍스트를 임의로 다듬지 말고 최대한 원문 보존
7. 논문의 목차 구조를 고려하고 문단 구성을 유지
8. 절대로 텍스트를 요약하거나 생략하지 말 것
9. 입력된 모든 페이지를, 각 페이지별 [PAGE X] 태그를 유지한 채 전부 변환하여 출력할 것
10. 변환된 결과 텍스트만 출력하고, 설명·평가·부연 문구를 절대 포함하지 말 것

[각주/미주 처리 규칙]
- 각주/미주 번호는 본문에 그대로 유지 (예: "...하였다¹.")
- 각주/미주 내용은 본문에서는 제거하고, 아래 형식으로 별도 저장:
    [FOOTNOTE 1] 각주 내용
    [ENDNOTE 1] 미주 내용
- 단, 해당 각주/미주가 본문 이해에 필수적인 경우에는 해당 위치에 괄호 안에 병합:
    예: "...하였다(다만, ~에 대해서는 추가 논의가 필요하다)."
- 본문과 각주/미주 내용은 분리하여 반환

출력 형식(형식을 반드시 지켜주세요):
1) 전처리된 본문
<본문 내용>

2) 각주/미주 목록
<각주/미주 내용>
"""

REF_PROMPT = """
다음 텍스트에서 '서론', '선행연구', '관련연구', '이론적 배경', '문헌연구', '연구동향', '연구사 정리' 등
기존 연구를 설명하는 부분만 추출해 주세요.
특히 다른 연구자들의 연구 내용을 요약하거나 언급한 부분이 있다면 모두 포함하여 추출해 주세요.
예: '홍길동(2020)은 ...라고 하였다.'와 같은 문장이 포함된 문단은 모두 추출.

조건:
- 페이지 구분 태그([PAGE X])를 절대 삭제하지 말고 유지
- 여러 페이지에 걸친 경우 페이지 범위를 한 줄로 표시 가능
- 연구 방법, 실험 결과, 결론은 제외
- 추출 텍스트도 띄어쓰기·맞춤법·한자 변환 적용
- 절대로 텍스트를 요약하거나 생략하지 말 것
- 추출된 결과 텍스트만 출력하고, 설명·평가·부연 문구를 절대 포함하지 말 것

출력 형식(형식을 반드시 지켜주세요):
<본문 내용>
"""

# -----------------------------
# 토큰 / 비용 계산
# -----------------------------
def count_tokens(model_name, text):
    try:
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text).total_tokens
    except Exception:
        return int(len(re.findall(r"\S+", text)) * 1.3)

def estimate_cost(model_name, input_tokens, output_tokens):
    price_in = PRICING[model_name]["input"] * (input_tokens / 1000)
    price_out = PRICING[model_name]["output"] * (output_tokens / 1000)
    return price_in + price_out

# -----------------------------
# 이미지 전처리 + OCR
# -----------------------------
def preprocess_image_for_ocr(image):
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)
    binary = (arr > 180) * 255
    return Image.fromarray(np.uint8(binary))

def ocr_with_page_tags(images):
    text_with_tags = []
    for i, img in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(img, lang='kor+eng').strip()
        except Exception:
            text = ""
        text_with_tags.append(f"[PAGE {i}]\n{text}")
    return "\n\n".join(text_with_tags)

# -----------------------------
# PDF 텍스트 추출
# -----------------------------
def extract_text_with_page_tags(pdf_path, max_pages):
    doc = fitz.open(pdf_path)
    if max_pages is None:
        max_pages = len(doc)

    texts = []
    for i in range(min(max_pages, len(doc))):
        page_text = doc[i].get_text()
        texts.append(f"[PAGE {i+1}]\n{page_text.strip()}")
    return "\n\n".join(texts)

# -----------------------------
# Gemini 호출
# -----------------------------
def gemini_process(text, extract_ref=False, vision_images=None):
    global total_cost_usd
    model_name = "gemini-1.5-pro" if vision_images is None else "gemini-1.5-flash"

    prompt = (REF_PROMPT if extract_ref else BASE_PROMPT) + "\n\n텍스트:\n" + text

    input_tokens = count_tokens(model_name, prompt)
    try:
        model = genai.GenerativeModel(model_name)
        if vision_images:
            response = model.generate_content([prompt] + vision_images)
        else:
            response = model.generate_content(prompt)
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "", model_name, 0, 0, 0.0

    output_text = (response.text or "").strip()
    output_tokens = count_tokens(model_name, output_text)
    cost = estimate_cost(model_name, input_tokens, output_tokens)
    total_cost_usd += cost

    return output_text, model_name, input_tokens, output_tokens, cost

# -----------------------------
# 페이지 범위 추출
# -----------------------------
def parse_page_ranges_from_ref_text(ref_text):
    pages_found = sorted({int(p) for p in re.findall(r"\[PAGE (\d+)\]", ref_text)})
    if not pages_found:
        return []
    ranges, start, prev = [], pages_found[0], pages_found[0]
    for p in pages_found[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p
    ranges.append((start, prev))
    return [f"{s}-{e}" if s != e else f"{s}" for s, e in ranges]

# -----------------------------
# PDF 유형 판별
# -----------------------------
def is_text_based_pdf(pdf_path, min_text_len=100):
    try:
        doc = fitz.open(pdf_path)
        total_text_len = 0
        for page in doc:
            total_text_len += len(page.get_text().strip())
            if total_text_len >= min_text_len:
                return True
        return False
    except Exception:
        return False

# -----------------------------
# PDF 처리
# -----------------------------
def process_pdf(pdf_path, max_pages=20, force=False):
    rel_path = os.path.relpath(pdf_path, PDF_DIR)
    rel_no_ext = os.path.splitext(rel_path)[0]

    json_path = os.path.join(JSON_DIR, rel_no_ext + ".json")
    txt_clean_path = os.path.join(TXT_DIR, rel_no_ext + ".txt")
    txt_ref_path = os.path.join(REF_TXT_DIR, rel_no_ext + ".txt")

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_clean_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_ref_path), exist_ok=True)

    if not force and os.path.exists(json_path):
        print(f"⚠️ {pdf_path} 이미 처리됨. 스킵.")
        return

    try:
        print(f"📄 PDF 분석 시작: {pdf_path}")
        text_based = is_text_based_pdf(pdf_path)
        print(f"   → PDF 유형: {'텍스트 기반' if text_based else '이미지 기반'}")

        if text_based:
            text_with_tags = extract_text_with_page_tags(pdf_path, max_pages)
            print(f"   → 추출 텍스트 샘플:\n{text_with_tags[:300]}...\n")
            cleaned, m1, in1, out1, c1 = gemini_process(text_with_tags, extract_ref=False)
            print(f"   [본문 처리] 모델: {m1}, 입력토큰: {in1}, 출력토큰: {out1}, 비용: ${c1:.6f}")
            ref, m2, in2, out2, c2 = gemini_process(text_with_tags, extract_ref=True)
            print(f"   [선행연구 추출] 모델: {m2}, 입력토큰: {in2}, 출력토큰: {out2}, 비용: ${c2:.6f}")
        else:
            images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=max_pages)
            processed_images = [preprocess_image_for_ocr(img) for img in images]
            ocr_text = ocr_with_page_tags(processed_images)
            print(f"   → OCR 결과 샘플:\n{ocr_text[:300]}...\n")
            cleaned, m1, in1, out1, c1 = gemini_process(ocr_text, extract_ref=False, vision_images=processed_images)
            print(f"   [본문 처리] 모델: {m1}, 입력토큰: {in1}, 출력토큰: {out1}, 비용: ${c1:.6f}")
            ref, m2, in2, out2, c2 = gemini_process(ocr_text, extract_ref=True, vision_images=processed_images)
            print(f"   [선행연구 추출] 모델: {m2}, 입력토큰: {in2}, 출력토큰: {out2}, 비용: ${c2:.6f}")
    except Exception as e:
        print(f"❌ PDF 처리 실패: {pdf_path} | {e}")
        return

    ref_page_ranges = parse_page_ranges_from_ref_text(ref)
    result_json = {
        "metadata": {
            "file_name": os.path.basename(pdf_path),
            "relative_path": rel_path,
            "is_text_based": text_based,
            "pages_processed": max_pages,
            "ref_page_ranges": ref_page_ranges,
            "timestamp": datetime.now().isoformat()
        },
        "cost_info": {
            "total_usd": round(c1 + c2, 6),
            "total_krw": round((c1 + c2) * USD_TO_KRW, 0),
            "details": {
                "cleaned_text": {"model": m1, "input_tokens": in1, "output_tokens": out1, "cost_usd": c1},
                "ref_text": {"model": m2, "input_tokens": in2, "output_tokens": out2, "cost_usd": c2}
            }
        },
        "cleaned_text": cleaned,
        "ref_text": ref
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    with open(txt_clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned or "")
    with open(txt_ref_path, "w", encoding="utf-8") as f:
        f.write(ref or "")

    print(f"✅ 저장 완료: {json_path}")
    print(f"✅ 저장 완료: {txt_clean_path}")
    print(f"✅ 저장 완료: {txt_ref_path}")
    print(f"💰 총 PDF 비용: ${(c1+c2):.6f} (~₩{(c1+c2)*USD_TO_KRW:,.0f})")

# -----------------------------
# 실행
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="전체 페이지 처리")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    args = parser.parse_args()

    max_pages = None if args.all else 20
    for root, _, files in os.walk(PDF_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"\n=== Processing: {pdf_path} ===")
                process_pdf(pdf_path, max_pages=max_pages, force=args.force)

    print(f"\n총 예상 비용: ${total_cost_usd:.6f} (~₩{total_cost_usd * USD_TO_KRW:,.0f})")

if __name__ == "__main__":
    main()
