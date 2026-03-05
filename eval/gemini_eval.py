"""
RAG 평가 스크립트: Context Precision 등 ragas 메트릭으로 검증.

- Google Gen AI (google-genai SDK) + ragas instructor 어댑터 사용.
- score() 호출 시 ragas 내부가 ascore() → agenerate()를 사용하므로,
  instructor.from_genai(use_async=True)로 AsyncInstructor를 쓰도록 패치함.
- 필요 시: pip install jsonref (instructor GENAI_TOOLS 모드에서 사용).
"""

import os
import instructor
from dotenv import load_dotenv
from google import genai
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

load_dotenv()

# ragas가 instructor.from_genai() 호출 시 use_async=True로 AsyncInstructor를 쓰도록 패치.
# (ragas는 기본으로 동기 Instructor만 만들어 agenerate() 사용 시 에러가 남)
from instructor.providers.genai import client as _genai_provider

_orig_from_genai = _genai_provider.from_genai


def _from_genai_async(client, mode=instructor.Mode.GENAI_TOOLS, use_async=True, **kwargs):
    """Async GenAI 클라이언트를 쓰는 from_genai 래퍼 (ragas agenerate 호환)."""
    return _orig_from_genai(client, mode=mode, use_async=use_async, **kwargs)


instructor.from_genai = _from_genai_async

# --- 클라이언트 및 메트릭 생성 ---
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
llm = llm_factory(
    "gemini-2.5-flash",
    provider="google",
    client=client,
    adapter="auto",
)
scorer = ContextPrecision(llm=llm)


def main():
    """Context Precision 단일 샘플 평가를 실행하고 점수를 출력한다."""
    result = scorer.score(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Eiffel Tower is located in Paris.",
            "The Brandenburg Gate is located in Berlin.",
        ],
    )
    print(f"Context Precision Score: {result.value}")


if __name__ == "__main__":
    main()
