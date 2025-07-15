# SEC Edgar API RAG 시스템

이 프로젝트는 SEC Edgar API를 사용하여 기업의 제출 서류를 수집하고, RAG(Retrieval-Augmented Generation) 시스템을 통해 질의응답을 수행하는 시스템입니다.

## 주요 기능

- **SEC Edgar API 연동**: 공식 SEC Edgar API를 통한 기업 정보 수집
- **제출 서류 처리**: 10-K, 10-Q 등의 제출 서류 텍스트 추출 및 전처리
- **RAG 시스템**: LangChain을 활용한 문서 기반 질의응답 시스템
- **기업 분석**: 여러 기업의 재무 정보 및 사업 현황 비교 분석

## 파일 구조

- `sec_edgar_api_rag.py`: 완전한 RAG 시스템 구현
- `SEC_Edgar_API_Example.py`: 기본 API 사용 예시
- `README_SEC_Edgar.md`: 이 문서

## 설치 및 설정

### 1. 필요 패키지 설치

```bash
pip install requests pandas langchain langchain-community langchain-huggingface
pip install faiss-cpu python-dotenv
```

### 2. 환경 변수 설정

`.env` 파일에 다음 설정 추가:

```env
# 임베딩 모델 경로
multilingual-e5-small-ko=intfloat/multilingual-e5-small

# LLM 모델 경로 (선택사항)
EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf=/path/to/your/model.gguf
```

## 사용 방법

### 1. 기본 API 사용

```python
from SEC_Edgar_API_Example import SimpleSecApi

# API 클라이언트 초기화
api = SimpleSecApi(user_agent="Your App Name your@email.com")

# 기업 정보 조회
company_info = api.get_company_by_ticker("AAPL")
print(f"회사명: {company_info['title']}")

# 제출 서류 목록 조회
submissions = api.get_company_submissions(company_info['cik_str'])
print(f"총 제출 서류 수: {len(submissions['filings']['recent']['form'])}")
```

### 2. RAG 시스템 사용

```python
from sec_edgar_api_rag import analyze_company_with_rag

# 환경 변수 로드
from dotenv import dotenv_values
config = dotenv_values(".env")

# 질문 목록 준비
questions = [
    "Apple의 주요 사업 영역은 무엇인가?",
    "최근 재무 성과는 어떻게 되나?",
    "주요 리스크 요인은 무엇인가?"
]

# RAG 시스템으로 기업 분석
analyze_company_with_rag("AAPL", questions, config)
```

### 3. 스크립트 실행

```bash
# 기본 예시 실행
python SEC_Edgar_API_Example.py

# 전체 RAG 시스템 실행
python sec_edgar_api_rag.py
```

## SEC Edgar API 주요 엔드포인트

### 1. 기업 티커 정보
- **URL**: `https://data.sec.gov/files/company_tickers.json`
- **설명**: 모든 상장 기업의 티커와 CIK 정보

### 2. 제출 서류 목록
- **URL**: `https://data.sec.gov/submissions/CIK{cik}.json`
- **설명**: 특정 기업의 모든 제출 서류 목록

### 3. 재무 팩트 데이터
- **URL**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
- **설명**: 기업의 XBRL 재무 데이터

### 4. 특정 재무 개념
- **URL**: `https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json`
- **설명**: 특정 재무 개념의 시계열 데이터

## 주요 제출 서류 유형

| 서류 유형 | 설명 | 주기 |
|-----------|------|------|
| 10-K | 연간 보고서 (종합적인 기업 정보) | 연간 |
| 10-Q | 분기 보고서 (분기별 재무 정보) | 분기 |
| 8-K | 중요 사건 보고서 (주요 공시 사항) | 수시 |
| DEF 14A | 주주총회 위임장 (경영진 보상 정보) | 연간 |
| S-1 | 신규 증권 등록서 (IPO 관련) | 수시 |

## 분석 가능한 질문 예시

### 재무 성과 관련
- "매출 증가율은 어떻게 되나?"
- "순이익 변화 추이는?"
- "현금 흐름 상황은?"

### 사업 전략 관련
- "주요 사업 영역은 무엇인가?"
- "신규 사업 계획은?"
- "경쟁 우위는 무엇인가?"

### 리스크 요인 관련
- "주요 리스크 요인은?"
- "규제 환경 변화 영향은?"
- "경쟁 상황은 어떤가?"

### 연구개발 관련
- "R&D 투자 현황은?"
- "특허 보유 현황은?"
- "기술 혁신 전략은?"

## 주의사항

### 1. API 사용 제한
- **요청 빈도**: 초당 10회로 제한
- **User-Agent**: 반드시 적절한 User-Agent 헤더 설정 필요
- **403 오류**: User-Agent 미설정 시 발생

### 2. 데이터 처리
- **메모리 사용량**: 대용량 문서 처리 시 메모리 사용량 주의
- **처리 시간**: 10-K 서류 같은 대용량 문서는 처리 시간이 오래 걸림
- **텍스트 품질**: HTML 태그 제거 과정에서 일부 포맷 정보 손실 가능

### 3. 모델 설정
- **임베딩 모델**: 한국어 지원 모델 사용 권장
- **LLM 모델**: 로컬 모델 사용 시 충분한 컴퓨팅 자원 필요
- **토큰 제한**: 모델의 컨텍스트 길이 제한 고려

## 확장 가능한 기능

### 1. 멀티 기업 비교
```python
def compare_companies(tickers, metric):
    """여러 기업의 특정 지표 비교"""
    results = {}
    for ticker in tickers:
        # 각 기업의 데이터 수집 및 분석
        pass
    return results
```

### 2. 시계열 분석
```python
def analyze_trends(cik, metric, years):
    """특정 기업의 시계열 데이터 분석"""
    facts = api.get_company_facts(cik)
    # 시계열 데이터 추출 및 분석
    pass
```

### 3. 리스크 분석
```python
def analyze_risks(cik):
    """기업의 리스크 요인 분석"""
    # 10-K 서류의 리스크 섹션 추출
    # 텍스트 분석을 통한 리스크 분류
    pass
```

## 문제 해결

### 1. 403 Forbidden 오류
```python
# User-Agent 헤더 확인
headers = {
    "User-Agent": "Your Company Name your@email.com"
}
```

### 2. 메모리 부족 오류
```python
# 문서 크기 제한
if len(content) > 50000:
    content = content[:50000]
```

### 3. 모델 로딩 오류
```python
# 모델 경로 확인
if not os.path.exists(model_path):
    print(f"모델 파일을 찾을 수 없습니다: {model_path}")
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 연락처

질문이나 제안사항이 있으시면 이슈를 등록해주세요.

---

**참고**: SEC Edgar API는 공식 API이며, 사용 시 SEC의 이용 약관을 준수해야 합니다. 