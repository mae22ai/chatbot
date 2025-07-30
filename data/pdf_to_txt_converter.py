import os
import pdfplumber

def convert_pdf_to_txt(source_dir, target_dir):
    """
    지정된 폴더(source_dir)와 그 하위 폴더의 모든 PDF 파일을
    대상 폴더(target_dir)에 동일한 구조를 유지하며 .txt 파일로 변환합니다.

    :param source_dir: PDF 파일들이 있는 원본 기본 폴더 (예: 'pdf')
    :param target_dir: 텍스트 파일을 저장할 대상 기본 폴더 (예: 'txt')
    """
    print(f"'{source_dir}' 폴더의 PDF를 '{target_dir}' 폴더로 변환 시작...")

    # 원본 폴더의 모든 하위 폴더를 탐색합니다.
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # PDF 파일만 대상으로 합니다.
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, filename)

                # 결과물이 저장될 경로를 생성합니다.
                relative_path = os.path.relpath(root, source_dir)
                txt_output_folder = os.path.join(target_dir, relative_path)

                # 폴더가 존재하지 않으면 생성합니다.
                os.makedirs(txt_output_folder, exist_ok=True)

                # .pdf 확장자를 .txt로 변경하여 저장할 파일 경로를 만듭니다.
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(txt_output_folder, txt_filename)

                print(f"변환 중: {pdf_path}  ->  {txt_path}")

                try:
                    extracted_text = ""
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + "\n\n"

                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(extracted_text)

                except Exception as e:
                    print(f" [오류] {pdf_path} 파일 변환 실패: {e}")

    print("모든 PDF 파일 변환 완료!")

if __name__ == '__main__':
    # 'data' 폴더 내부에서 실행하므로 기준 경로는 'pdf'와 'txt'가 됩니다.
    pdf_base_folder = 'pdf'
    txt_base_folder = 'txt'

    # 함수 실행
    convert_pdf_to_txt(pdf_base_folder, txt_base_folder)