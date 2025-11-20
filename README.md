
# 🚀 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/mae22ai/chatbot.git
cd chatbot
```
### 2. 가상환경 생성 및 활성화
```
# 가상환경 생성
python3.11 -m venv venv

# 가상환경 활성화(윈도우)
source ./venv/Scripts/activate
# 가상환경 활성화(맥)
source venv/bin/activate
```

### 3. 의존성 설치
```
pip install -r requirements.txt
```

### 4. ```.env``` 파일 생성 후 환경변수 설정

### 5. 서버 실행
```
python manage.py runserver
```
브라우저에서 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 접속



## 🛠️ 주요 변경 사항 (2024.11 Refactoring)

### 1. 아키텍처 개선
*   **Service Layer 도입**: 비대해진 `views.py`의 로직을 `services.py`로 분리하여 유지보수성을 높였습니다.
*   **Singleton 패턴 적용**: `BareunClient`를 도입하여 형태소 분석기 연결을 재사용, 응답 속도를 획기적으로 개선했습니다.
*   **에러 핸들링 강화**: 외부 API 장애 시에도 서버가 중단되지 않고 사용자에게 안내 메시지를 제공합니다.

### 2. 분석 정확도 및 기능 향상
*   **한자어 의미 기반 분해**: `학교` -> `학`+`교` 처럼 의미가 투명한 한자어는 분해하고, `모순` 처럼 관용적인 단어는 유지하는 스마트한 분석을 도입했습니다.
*   **어절 구조 보존**: LLM에게 띄어쓰기 정보(`나/NP+는/JX`)를 정확히 전달하여 '만큼', '대로', '뿐' 등의 문맥 의존적 품사를 정확히 분석합니다.
*   **스마트 쟁점 로딩**: 불필요한 쟁점 문서 로딩을 방지하여 분석 속도와 정확도를 동시에 높였습니다.

### 3. 개발자 도구 (Developer Tools)
터미널에서 분석 파이프라인을 단계별로 테스트할 수 있는 CLI 도구가 추가되었습니다.

```bash
# 사용법: python manage.py test_analyzer "분석할 문장" [옵션]

# 1. 바른 형태소 분석기 결과만 확인
python manage.py test_analyzer "나는 학교에 간다" --bareun

# 2. LLM 분석만 확인 (토큰 절약)
python manage.py test_analyzer "나는 학교에 간다" --llm

# 3. 전체 파이프라인 실행 (바른 + 휴리스틱 + LLM)
python manage.py test_analyzer "나는 학교에 간다" --all
```
