from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import List, Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph


Intent = Literal[
    "greeting",
    "small_talk",
    "faq",
    "product_inquiry",
    "order_status",
    "refund",
    "complaint",
    "unknown",
]


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    reason: str


class IntentGraphState(TypedDict, total=False):
    """LangGraph 상태 스키마.

    기본 동작은 '의도 분류'만 수행하며, 벡터서치는 `ENABLE_RETRIEVAL=1`일 때만 시도합니다.
    """

    user_text: str
    intent: Intent
    confidence: float
    reason: str
    retrieved_docs: List[Document]
    retrieval_attempts: int


def rule_based_intent(text: str) -> Intent | None:
    """고정 패턴(키워드)으로 빠르게 의도를 분류한다. 매칭 실패 시 None 반환."""
    t = text.lower().strip()

    rules = [
        (r"(환불|반품|취소)", "refund"),
        (r"(배송|언제 와|도착|운송장|주문 상태)", "order_status"),
        (r"(불만|화나|문제|오류|안 돼|최악)", "complaint"),
        (r"(가격|할인|재고|스펙|사양|기능)", "product_inquiry"),
        (r"(안녕|hello|hi|반가워)", "greeting"),
    ]

    for pattern, label in rules:
        if re.search(pattern, t):
            return label  # type: ignore[return-value]
    return None


def llm_intent(text: str, model: str = "gemma3-270m:latest") -> IntentResult:
    """규칙으로 못 잡은 문장을 소형 LLM으로 JSON 강제 분류한다."""
    system = """너는 의도분류기다.
출력은 반드시 JSON 한 개만 반환:
{{"intent":"...", "confidence":0.0, "reason":"..."}}
intent 허용값:
[greeting, small_talk, faq, product_inquiry, order_status, refund, complaint, unknown]
규칙:
- 문장 의미만 보고 하나만 선택
- confidence는 0~1 숫자
- 불명확하면 unknown
- JSON 외 텍스트 금지
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{user_text}")]
    )

    llm = ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        temperature=0.0,
        # Ollama JSON 모드(모델에 따라 안정성 차이 있음)
        format="json",
    )

    raw = (prompt | llm).invoke({"user_text": text}).content
    # 일부 모델은 json 모드에서도 코드펜스를 섞어 반환할 수 있어 제거한다.
    raw = str(raw).strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    # 방어적 파싱
    try:
        obj = json.loads(raw)
        intent = obj.get("intent", "unknown")
        confidence = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason", ""))

        if intent not in {
            "greeting",
            "small_talk",
            "faq",
            "product_inquiry",
            "order_status",
            "refund",
            "complaint",
            "unknown",
        }:
            intent = "unknown"

        confidence = max(0.0, min(1.0, confidence))
        return IntentResult(intent=intent, confidence=confidence, reason=reason)
    except Exception:
        return IntentResult(intent="unknown", confidence=0.0, reason="invalid_json")


def detect_intent(text: str) -> IntentResult:
    """1차 규칙 + 2차 LLM fallback 방식으로 안정적으로 의도를 판별한다."""
    rule = rule_based_intent(text)
    if rule is not None:
        return IntentResult(intent=rule, confidence=0.95, reason="matched_rule")
    return llm_intent(text)


def intent_classifier_node(state: IntentGraphState) -> IntentGraphState:
    """의도 분류 노드: 입력 텍스트 -> intent/confidence/reason."""
    user_text = state["user_text"]
    result = detect_intent(user_text)
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "reason": result.reason,
    }


def should_retrieve(state: IntentGraphState) -> Literal["retrieve", "end"]:
    """의도 및 환경설정에 따라 검색 단계 수행 여부 결정."""
    enable_retrieval = os.getenv("ENABLE_RETRIEVAL", "0") == "1"
    intent = state.get("intent", "unknown")

    # 인사 류는 기본적으로 검색을 생략
    if not enable_retrieval:
        return "end"
    if intent == "greeting":
        return "end"
    return "retrieve"


def retriever_node(state: IntentGraphState) -> IntentGraphState:
    """(옵션) 벡터서치 노드.

    실제 Chroma/TEI 구성이 준비되지 않았을 수 있으므로, 실패 시 빈 리스트를 반환한다.
    """
    user_text = state["user_text"]

    if os.getenv("ENABLE_RETRIEVAL", "0") != "1":
        return {"retrieved_docs": []}

    try:
        from scripts.embedding import load_vectorstore

        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(user_text)
        return {"retrieved_docs": docs}
    except Exception:
        return {"retrieved_docs": []}


if __name__ == "__main__":
    graph = StateGraph(IntentGraphState)
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("retriever", retriever_node)

    graph.add_edge(START, "intent_classifier")
    graph.add_conditional_edges(
        "intent_classifier",
        should_retrieve,
        {"retrieve": "retriever", "end": END},
    )
    graph.add_edge("retriever", END)

    graph = graph.compile()

    # 데모: 기본은 intent만 출력(ENABLE_RETRIEVAL=1일 때만 retrieved_docs가 채워짐)
    samples = [
        "배송 언제 오나요?",
        "환불 받고 싶어요",
        "안녕하세요",
        "이 제품 배터리 용량이 어떻게 돼요?",
        "그냥 궁금해서 물어봐요",
    ]
    for s in samples:
        state = graph.invoke({"user_text": s, "retrieval_attempts": 0})
        print(s, "->", state.get("intent"), state.get("confidence"), state.get("reason"))