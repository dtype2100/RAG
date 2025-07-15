#!/usr/bin/env python3
"""
SEC Edgar API 사용 예시

이 스크립트는 SEC Edgar API를 사용하여 기업 정보를 수집하는 간단한 예시입니다.
"""

import requests
import json
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime
import re

class SimpleSecApi:
    """
    간단한 SEC Edgar API 클라이언트
    """
    
    def __init__(self, user_agent: str = "SEC API Client admin@example.com"):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_company_by_ticker(self, ticker: str) -> Optional[Dict]:
        """
        티커로 기업 정보 조회
        """
        try:
            # 회사 티커 정보 조회
            response = self.session.get(f"{self.base_url}/files/company_tickers.json")
            response.raise_for_status()
            time.sleep(0.1)  # Rate limiting
            
            tickers_data = response.json()
            ticker_upper = ticker.upper()
            
            for key, company in tickers_data.items():
                if company.get('ticker') == ticker_upper:
                    return company
            
            return None
        except Exception as e:
            print(f"기업 정보 조회 실패: {e}")
            return None
    
    def get_company_submissions(self, cik: str) -> Optional[Dict]:
        """
        기업의 제출 서류 목록 조회
        """
        try:
            cik_padded = str(cik).zfill(10)
            response = self.session.get(f"{self.base_url}/submissions/CIK{cik_padded}.json")
            response.raise_for_status()
            time.sleep(0.1)
            
            return response.json()
        except Exception as e:
            print(f"제출 서류 조회 실패: {e}")
            return None
    
    def get_company_facts(self, cik: str) -> Optional[Dict]:
        """
        기업의 재무 팩트 데이터 조회
        """
        try:
            cik_padded = str(cik).zfill(10)
            response = self.session.get(f"{self.base_url}/api/xbrl/companyfacts/CIK{cik_padded}.json")
            response.raise_for_status()
            time.sleep(0.1)
            
            return response.json()
        except Exception as e:
            print(f"재무 팩트 조회 실패: {e}")
            return None

def analyze_company(ticker: str):
    """
    기업 분석 함수
    """
    api = SimpleSecApi()
    
    print(f"\n=== {ticker} 기업 분석 ===")
    
    # 1. 기업 기본 정보
    company_info = api.get_company_by_ticker(ticker)
    if not company_info:
        print(f"{ticker} 기업 정보를 찾을 수 없습니다.")
        return
    
    print(f"회사명: {company_info['title']}")
    print(f"CIK: {company_info['cik_str']}")
    print(f"티커: {company_info['ticker']}")
    
    # 2. 제출 서류 정보
    submissions = api.get_company_submissions(company_info['cik_str'])
    if submissions:
        recent_filings = submissions['filings']['recent']
        print(f"총 제출 서류 수: {len(recent_filings['form'])}")
        
        # 제출 서류 유형별 통계
        form_counts = {}
        for form in recent_filings['form']:
            form_counts[form] = form_counts.get(form, 0) + 1
        
        print("\n제출 서류 유형별 통계 (상위 10개):")
        sorted_forms = sorted(form_counts.items(), key=lambda x: x[1], reverse=True)
        for form, count in sorted_forms[:10]:
            print(f"  {form}: {count}개")
        
        # 최근 10-K, 10-Q 서류
        print("\n최근 10-K/10-Q 서류:")
        for i, (form, date, doc) in enumerate(zip(
            recent_filings['form'], 
            recent_filings['filingDate'], 
            recent_filings['primaryDocument']
        )):
            if form in ['10-K', '10-Q'] and i < 5:
                print(f"  {form} - {date} - {doc}")
    
    # 3. 재무 팩트 정보 (요약)
    facts = api.get_company_facts(company_info['cik_str'])
    if facts:
        print(f"\n재무 팩트 정보:")
        print(f"회사명: {facts['entityName']}")
        
        # 주요 재무 지표들
        if 'us-gaap' in facts['facts']:
            us_gaap = facts['facts']['us-gaap']
            key_metrics = [
                'Revenues', 'Revenue', 'SalesRevenueNet',
                'NetIncomeLoss', 'NetIncomeAttributableToParent',
                'Assets', 'AssetsCurrent', 'Liabilities',
                'StockholdersEquity', 'CommonStockSharesOutstanding'
            ]
            
            print("\n주요 재무 지표:")
            for metric in key_metrics:
                if metric in us_gaap:
                    print(f"  {metric}: 데이터 있음")
                    # 최근 데이터 하나만 출력
                    if 'USD' in us_gaap[metric]['units']:
                        recent_data = us_gaap[metric]['units']['USD'][-1]
                        print(f"    최근 값: {recent_data.get('val', 'N/A')} ({recent_data.get('end', 'N/A')})")
                    break

def get_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    """
    제출 서류 URL 생성
    """
    accession_clean = accession_number.replace('-', '')
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_document}"

def demonstrate_filing_access():
    """
    제출 서류 접근 데모
    """
    print("\n=== 제출 서류 접근 데모 ===")
    
    api = SimpleSecApi()
    
    # Apple 정보 조회
    apple_info = api.get_company_by_ticker("AAPL")
    if not apple_info:
        print("Apple 정보를 찾을 수 없습니다.")
        return
    
    # 제출 서류 목록 조회
    submissions = api.get_company_submissions(apple_info['cik_str'])
    if not submissions:
        print("제출 서류를 찾을 수 없습니다.")
        return
    
    # 최근 10-K 서류 찾기
    recent_filings = submissions['filings']['recent']
    for i, form in enumerate(recent_filings['form']):
        if form == '10-K':
            filing_date = recent_filings['filingDate'][i]
            accession_number = recent_filings['accessionNumber'][i]
            primary_document = recent_filings['primaryDocument'][i]
            
            print(f"최근 10-K 서류:")
            print(f"  날짜: {filing_date}")
            print(f"  Accession Number: {accession_number}")
            print(f"  문서: {primary_document}")
            
            # 서류 URL 생성
            filing_url = get_filing_url(
                apple_info['cik_str'], 
                accession_number, 
                primary_document
            )
            print(f"  URL: {filing_url}")
            
            # 서류 내용 일부 가져오기 (선택사항)
            try:
                response = api.session.get(filing_url)
                response.raise_for_status()
                content = response.text
                
                # HTML 태그 제거 후 처음 500자만 출력
                clean_content = re.sub(r'<[^>]+>', '', content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                
                print(f"\n서류 내용 미리보기:")
                print(f"  {clean_content[:500]}...")
                
            except Exception as e:
                print(f"서류 내용 조회 실패: {e}")
            
            break

def main():
    """
    메인 함수
    """
    print("SEC Edgar API 사용 예시")
    print("=" * 40)
    
    # 여러 기업 분석
    companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    for ticker in companies:
        analyze_company(ticker)
        print("-" * 40)
    
    # 제출 서류 접근 데모
    demonstrate_filing_access()
    
    print("\n=== 사용 가이드 ===")
    print("""
    SEC Edgar API 주요 엔드포인트:
    
    1. 기업 티커 정보: /files/company_tickers.json
    2. 제출 서류 목록: /submissions/CIK{cik}.json
    3. 재무 팩트 데이터: /api/xbrl/companyfacts/CIK{cik}.json
    4. 특정 재무 개념: /api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json
    
    주요 제출 서류 유형:
    - 10-K: 연간 보고서
    - 10-Q: 분기 보고서
    - 8-K: 중요 사건 보고서
    - DEF 14A: 주주총회 위임장
    - S-1: 신규 증권 등록서
    
    주의사항:
    - SEC API는 10 requests/second 제한이 있습니다
    - User-Agent 헤더를 반드시 설정해야 합니다
    - 403 오류 발생 시 User-Agent를 확인하세요
    """)

if __name__ == "__main__":
    main() 