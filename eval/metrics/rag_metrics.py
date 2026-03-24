"""RAGAS 기반 RAG 평가 메트릭 설정.

지원 메트릭:
  - LLMContextPrecisionWithReference: 검색된 문서가 reference 기준으로 얼마나 정밀한지
  - LLMContextRecall: reference 답변을 생성하는 데 필요한 내용이 검색되었는지
  - Faithfulness: LLM 응답이 검색된 문서에 근거하고 있는지
  - ResponseRelevancy: LLM 응답이 질문과 얼마나 관련 있는지
"""

import os

import instructor
from dotenv import load_dotenv
from google import genai
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

# ragas 내부가 agenerate()를 사용하므로 AsyncInstructor를 쓰도록 패치
from instructor.providers.genai import client as _genai_provider

_orig_from_genai = _genai_provider.from_genai


def _from_genai_async(client, mode=instructor.Mode.GENAI_TOOLS, use_async=True, **kwargs):
    return _orig_from_genai(client, mode=mode, use_async=use_async, **kwargs)


instructor.from_genai = _from_genai_async


def get_ragas_llm():
    """RAGAS 평가용 Gemini LLM을 반환한다. .env의 GOOGLE_API_KEY 필요."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    client = genai.Client(api_key=api_key)
    return llm_factory(
        "gemini-2.5-flash",
        provider="google",
        client=client,
        adapter="auto",
    )


def get_rag_metrics(llm=None) -> list:
    """RAG 평가에 사용할 RAGAS 메트릭 목록을 반환한다."""
    if llm is None:
        llm = get_ragas_llm()
    return [
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm),
    ]
