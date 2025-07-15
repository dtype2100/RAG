#!/usr/bin/env python3
"""
SEC Edgar API RAG 시스템

이 스크립트는 SEC Edgar API에서 기업 정보와 제출 서류를 가져와서 
RAG 시스템에 활용하는 방법을 보여줍니다.

사용 예시:
    python sec_edgar_api_rag.py
"""

import requests
import json
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
import os

# RAG 관련 라이브러리
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatLlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document

from dotenv import dotenv_values
import multiprocessing


class SECEdgarAPI:
    """
    SEC Edgar API 클라이언트 클래스
    - 공식 SEC Edgar API를 사용하여 기업 정보와 제출 서류를 수집
    - SEC의 요청 제한 (10 requests/second) 준수
    """
    
    def __init__(self, user_agent: str = "SEC Edgar API Client admin@example.com"):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        SEC API에 요청을 보내고 응답을 반환
        SEC의 요청 제한 (10 requests/second)을 준수하기 위해 0.1초 대기
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            time.sleep(0.1)  # SEC API 요청 제한 준수
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
            return None
    
    def get_company_tickers(self) -> dict:
        """
        모든 회사 티커 및 CIK 정보 조회
        """
        endpoint = "/files/company_tickers.json"
        return self._make_request(endpoint)
    
    def get_company_submissions(self, cik: str) -> dict:
        """
        특정 회사의 제출 서류 목록 조회
        """
        # CIK를 10자리 문자열로 변환
        cik_padded = str(cik).zfill(10)
        endpoint = f"/submissions/CIK{cik_padded}.json"
        return self._make_request(endpoint)
    
    def get_company_facts(self, cik: str) -> dict:
        """
        특정 회사의 재무 팩트 데이터 조회
        """
        cik_padded = str(cik).zfill(10)
        endpoint = f"/api/xbrl/companyfacts/CIK{cik_padded}.json"
        return self._make_request(endpoint)
    
    def get_company_concept(self, cik: str, taxonomy: str, tag: str) -> dict:
        """
        특정 회사의 특정 재무 개념 데이터 조회
        """
        cik_padded = str(cik).zfill(10)
        endpoint = f"/api/xbrl/companyconcept/CIK{cik_padded}/{taxonomy}/{tag}.json"
        return self._make_request(endpoint)
    
    def search_company_by_ticker(self, ticker: str) -> Optional[dict]:
        """
        티커로 회사 정보 검색
        """
        tickers_data = self.get_company_tickers()
        if not tickers_data:
            return None
            
        ticker_upper = ticker.upper()
        for key, company in tickers_data.items():
            if company.get('ticker') == ticker_upper:
                return company
        return None
    
    def get_filing_documents(self, cik: str, accession_number: str) -> List[dict]:
        """
        특정 제출 서류의 문서 목록 조회
        """
        # Accession number에서 하이픈 제거
        accession_clean = accession_number.replace('-', '')
        cik_padded = str(cik).zfill(10)
        
        # 제출 서류 디렉토리 URL 구성
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/"
        
        try:
            response = self.session.get(filing_url)
            response.raise_for_status()
            
            # HTML에서 문서 링크 추출 (간단한 파싱)
            content = response.text
            document_links = re.findall(r'<a[^>]*href="([^"]*\.(?:htm|html|txt))"[^>]*>([^<]*)</a>', content, re.IGNORECASE)
            
            documents = []
            for link, name in document_links:
                if not link.startswith('http'):
                    link = filing_url + link
                documents.append({
                    'name': name.strip(),
                    'url': link
                })
            
            return documents
        except requests.exceptions.RequestException as e:
            print(f"문서 목록 조회 실패: {e}")
            return []
    
    def get_filing_content(self, filing_url: str) -> str:
        """
        제출 서류의 내용 가져오기
        """
        try:
            response = self.session.get(filing_url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"서류 내용 조회 실패: {e}")
            return ""


class SECFilingProcessor:
    """
    SEC 제출 서류를 처리하고 RAG 시스템에 활용하기 위한 클래스
    """
    
    def __init__(self, sec_api: SECEdgarAPI):
        self.sec_api = sec_api
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def clean_html_content(self, html_content: str) -> str:
        """
        HTML 내용에서 텍스트만 추출하고 정리
        """
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', html_content)
        
        # 특수 문자 정리
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # 연속된 공백 및 줄바꿈 정리
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def process_filing_for_rag(self, cik: str, accession_number: str, form_type: str) -> List[Document]:
        """
        특정 제출 서류를 RAG 시스템용으로 처리
        """
        documents = []
        
        # 제출 서류의 문서 목록 조회
        filing_documents = self.sec_api.get_filing_documents(cik, accession_number)
        
        for doc in filing_documents[:2]:  # 처음 2개 문서만 처리 (메모리 절약)
            print(f"처리 중: {doc['name']}")
            
            # 문서 내용 가져오기
            content = self.sec_api.get_filing_content(doc['url'])
            
            if content:
                # HTML 내용 정리
                clean_content = self.clean_html_content(content)
                
                # 너무 긴 내용은 일부만 사용 (처리 시간 단축)
                if len(clean_content) > 50000:
                    clean_content = clean_content[:50000]
                
                # 텍스트 분할
                text_chunks = self.text_splitter.split_text(clean_content)
                
                # Document 객체 생성
                for i, chunk in enumerate(text_chunks):
                    if len(chunk.strip()) > 100:  # 너무 짧은 청크는 제외
                        doc_metadata = {
                            'source': doc['url'],
                            'document_name': doc['name'],
                            'cik': cik,
                            'accession_number': accession_number,
                            'form_type': form_type,
                            'chunk_id': i
                        }
                        documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def get_company_filing_summary(self, cik: str, form_types: List[str] = None) -> pd.DataFrame:
        """
        회사의 제출 서류 요약 정보를 DataFrame으로 반환
        """
        submissions = self.sec_api.get_company_submissions(cik)
        
        if not submissions:
            return pd.DataFrame()
        
        recent_filings = submissions['filings']['recent']
        
        # DataFrame 생성
        df = pd.DataFrame({
            'form': recent_filings['form'],
            'filing_date': recent_filings['filingDate'],
            'accession_number': recent_filings['accessionNumber'],
            'primary_document': recent_filings['primaryDocument'],
            'size': recent_filings['size'],
            'is_xbrl': recent_filings['isXBRL'],
            'is_inline_xbrl': recent_filings['isInlineXBRL']
        })
        
        # 날짜 형식 변환
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        
        # 폼 타입 필터링
        if form_types:
            df = df[df['form'].isin(form_types)]
        
        return df.sort_values('filing_date', ascending=False)


def analyze_company_with_rag(ticker: str, questions: List[str], config: dict):
    """
    특정 기업의 SEC 제출 서류를 분석하고 RAG 시스템으로 질문에 답변
    """
    print(f"\n=== {ticker} 기업 분석 시작 ===")
    
    # SEC Edgar API 초기화
    sec_api = SECEdgarAPI(user_agent=f"RAG Analysis Bot for {ticker}")
    filing_processor = SECFilingProcessor(sec_api)
    
    # 1. 기업 정보 조회
    print(f"1. {ticker} 기업 정보 조회 중...")
    company_info = sec_api.search_company_by_ticker(ticker)
    if not company_info:
        print(f"{ticker} 기업 정보를 찾을 수 없습니다.")
        return
    
    print(f"   회사명: {company_info['title']}")
    print(f"   CIK: {company_info['cik_str']}")
    
    # 2. 제출 서류 목록 조회
    print("2. 제출 서류 목록 조회 중...")
    filings_df = filing_processor.get_company_filing_summary(
        company_info['cik_str'], 
        form_types=['10-K', '10-Q']
    )
    
    if filings_df.empty:
        print(f"{ticker}의 제출 서류가 없습니다.")
        return
    
    print(f"   조회된 제출 서류 수: {len(filings_df)}")
    
    # 3. 최신 10-K 서류 처리
    print("3. 최신 10-K 서류 처리 중...")
    ten_k_filings = filings_df[filings_df['form'] == '10-K']
    
    if ten_k_filings.empty:
        print("10-K 서류를 찾을 수 없습니다.")
        return
    
    latest_10k = ten_k_filings.iloc[0]
    print(f"   처리할 서류: {latest_10k['form']} - {latest_10k['filing_date'].strftime('%Y-%m-%d')}")
    
    # 문서 처리
    documents = filing_processor.process_filing_for_rag(
        cik=company_info['cik_str'],
        accession_number=latest_10k['accession_number'],
        form_type=latest_10k['form']
    )
    
    if not documents:
        print("처리할 문서가 없습니다.")
        return
    
    print(f"   생성된 문서 청크 수: {len(documents)}")
    
    # 4. RAG 시스템 구성
    print("4. RAG 시스템 구성 중...")
    
    # 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name=config.get("multilingual-e5-small-ko", "intfloat/multilingual-e5-small"),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # LLM 설정
    model_path = config.get("EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf")
    if not model_path or not os.path.exists(model_path):
        print("LLM 모델 파일을 찾을 수 없습니다. 모델 없이 문서 검색만 수행합니다.")
        
        # 간단한 문서 검색
        for question in questions:
            print(f"\n질문: {question}")
            docs = retriever.get_relevant_documents(question)
            if docs:
                print(f"관련 문서 내용: {docs[0].page_content[:300]}...")
            else:
                print("관련 문서를 찾을 수 없습니다.")
        return
    
    # LLM 초기화
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = ChatLlamaCpp(
        temperature=0.1,
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=8,
        n_batch=32,
        max_tokens=512,
        n_threads=multiprocessing.cpu_count() - 1,
        repeat_penalty=1.1,
        top_p=0.9,
        verbose=False,
        callback_manager=callback_manager,
        use_mlock=True,
        use_mmap=True,
    )
    
    # QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    print("   RAG 시스템 구성 완료!")
    
    # 5. 질문 처리
    print("5. 질문 처리 중...")
    
    for question in questions:
        print(f"\n질문: {question}")
        print("="*50)
        
        try:
            result = qa_chain.invoke({"query": question})
            print(f"\n답변: {result['result']}")
            
            # 참조 문서 정보
            if result.get('source_documents'):
                print("\n참조 문서:")
                for i, doc in enumerate(result['source_documents'][:2]):
                    print(f"{i+1}. {doc.metadata['document_name']} - {doc.metadata['form_type']}")
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("="*50)


def test_sec_api():
    """
    SEC Edgar API 기본 기능 테스트
    """
    print("=== SEC Edgar API 기본 기능 테스트 ===")
    
    sec_api = SECEdgarAPI()
    
    # 1. 기업 정보 조회 테스트
    print("\n1. 기업 정보 조회 테스트")
    companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    for ticker in companies:
        company_info = sec_api.search_company_by_ticker(ticker)
        if company_info:
            print(f"   ✓ {ticker}: {company_info['title']} (CIK: {company_info['cik_str']})")
        else:
            print(f"   ✗ {ticker}: 정보 조회 실패")
    
    # 2. Apple 제출 서류 통계
    print("\n2. Apple 제출 서류 통계")
    apple_info = sec_api.search_company_by_ticker("AAPL")
    if apple_info:
        submissions = sec_api.get_company_submissions(apple_info['cik_str'])
        if submissions:
            form_types = submissions['filings']['recent']['form']
            form_counts = {}
            for form in form_types:
                form_counts[form] = form_counts.get(form, 0) + 1
            
            # 상위 10개 제출 서류 유형 출력
            sorted_forms = sorted(form_counts.items(), key=lambda x: x[1], reverse=True)
            for form, count in sorted_forms[:10]:
                print(f"   {form}: {count}개")
    
    print("\n=== 테스트 완료 ===")


def main():
    """
    메인 함수
    """
    print("SEC Edgar API RAG 시스템 시작")
    
    # 환경 변수 로드
    config = dotenv_values(dotenv_path=".env")
    
    # 기본 테스트 실행
    test_sec_api()
    
    # Apple 기업 분석 예시
    apple_questions = [
        "Apple의 주요 사업 영역은 무엇인가?",
        "Apple의 최근 매출 성과는 어떻게 되나?",
        "Apple이 직면한 주요 리스크는 무엇인가?",
        "Apple의 연구개발 투자 현황은?"
    ]
    
    analyze_company_with_rag("AAPL", apple_questions, config)
    
    print("\n=== 사용 가이드 ===")
    print("""
    이 스크립트는 SEC Edgar API를 사용하여 기업 정보를 수집하고 RAG 시스템에 활용합니다.
    
    주요 기능:
    1. SEC Edgar API를 통한 기업 정보 수집
    2. 제출 서류(10-K, 10-Q 등) 텍스트 추출
    3. RAG 시스템 구축 및 질의응답
    
    주요 제출 서류 유형:
    - 10-K: 연간 보고서 (종합적인 기업 정보)
    - 10-Q: 분기 보고서 (분기별 재무 정보)
    - 8-K: 중요 사건 보고서 (주요 공시 사항)
    
    주의사항:
    - SEC API는 10 requests/second 제한이 있습니다
    - User-Agent 헤더를 적절히 설정해야 합니다
    - 대용량 문서 처리 시 메모리 사용량에 주의하세요
    """)


if __name__ == "__main__":
    main() 