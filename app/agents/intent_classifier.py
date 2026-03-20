import operator
from typing import Annotated, List

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from typing_extensions import TypedDict

from app.config import settings


class IntentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    intent: str
    retrieved_docs: List[Document]
    retrieval_attempts: int


def _get_message_content(msg) -> str:
    if msg is None:
        return ""
    return msg.get("content", "") if isinstance(msg, dict) else msg.content


def get_model():
    return init_chat_model(
        settings.INTENT_MODEL,
        model_provider="ollama",
        base_url=settings.INTENT_BASE_URL,
        temperature=settings.INTENT_TEMPERATURE,
        max_tokens=settings.INTENT_MAX_TOKENS,
        model_kwargs={"num_predict": 50},
    )


def make_intent_classifier(model):
    def intent_classifier(state: IntentState):
        user_message = _get_message_content(state["messages"][-1] if state["messages"] else None)

        response = model.invoke([
            SystemMessage(content=(
                "당신은 의도 분류 전문가입니다. 주어진 문장의 의도를 다음 중 하나로 분류하세요:\n"
                "- 정보요청: 특정 주제에 대한 정보나 설명을 요청\n"
                "- 개념정의: 용어나 개념의 정의를 요청\n"
                "- 비교분석: 두 가지 이상의 개념을 비교하거나 관계를 파악\n"
                "- 기술질문: 작동 방식이나 방법론에 대한 질문\n"
                "- 기타: 위에 해당하지 않는 경우\n\n"
                '의도만 답변하세요. 예: "정보요청"'
            )),
            HumanMessage(content=f"문장: {user_message}"),
        ])

        return {
            "messages": [response],
            "intent": response.content.strip(),
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    return intent_classifier


def make_retriever_call(vectorstore: InMemoryVectorStore):
    def retriever_call(state: IntentState):
        attempts = state.get("retrieval_attempts", 0)
        k = min(settings.RETRIEVAL_K + attempts * 3, 10)
        user_message = _get_message_content(state["messages"][0] if state["messages"] else None)

        docs_with_scores = vectorstore.similarity_search_with_score(user_message, k=k)

        return {
            "messages": [],
            "retrieved_docs": [doc for doc, _ in docs_with_scores],
            "retrieved_scores": [score for _, score in docs_with_scores],
            "user_query": user_message,
        }

    return retriever_call


def make_verify_retrieval(model):
    def verify_retrieval(state: IntentState):
        user_query = state.get("user_query", "")
        intent = state.get("intent", "")
        retrieved_docs = state.get("retrieved_docs", [])
        retrieved_scores = state.get("retrieved_scores", [])
        attempts = state.get("retrieval_attempts", 0)

        fail = {"messages": [], "retrieval_valid": False, "filtered_docs": [], "retrieval_attempts": attempts + 1}

        if not retrieved_docs:
            return fail

        if retrieved_scores and max(retrieved_scores) < settings.SIMILARITY_THRESHOLD:
            return fail

        top_docs = retrieved_docs[:3]
        doc_summaries = "\n".join(
            [f"{i+1}. {doc.page_content[:100]}..." for i, doc in enumerate(top_docs)]
        )

        verification = model.invoke([
            SystemMessage(content=(
                "검색된 문서들이 사용자 질문과 의도에 관련이 있는지 판단하세요.\n"
                "관련성이 있는 문서 번호만 쉼표로 구분하여 나열하세요. 없으면 '없음'이라고 답변하세요. 예: 1,3 또는 없음"
            )),
            HumanMessage(content=(
                f"사용자 질문: {user_query}\n의도: {intent}\n\n검색된 문서:\n{doc_summaries}\n\n"
                "관련성이 있는 문서 번호만 쉼표로 구분하여 나열하세요. 없으면 '없음'이라고 답변하세요."
            )),
        ])

        text = verification.content.strip()
        if "없음" in text or not text:
            return fail

        try:
            indices = [int(x.strip()) - 1 for x in text.split(",") if x.strip().isdigit()]
            filtered_docs = [top_docs[i] for i in indices if 0 <= i < len(top_docs)]
        except Exception:
            filtered_docs = []

        if not filtered_docs:
            return fail

        return {
            "messages": [SystemMessage(content=f"참고 문서:\n{chr(10).join([d.page_content for d in filtered_docs])}")],
            "retrieval_valid": True,
            "filtered_docs": filtered_docs,
            "retrieval_attempts": attempts,
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    return verify_retrieval


def should_retry_retrieval(state: IntentState) -> str:
    if state.get("retrieval_valid", False):
        return "proceed_to_llm"
    if state.get("retrieval_attempts", 0) < settings.MAX_RETRIEVAL_ATTEMPTS:
        return "retry_retrieval"
    return "proceed_without_docs"


def make_llm_call(model):
    _prompts = {
        "정보요청": "사용자가 요청한 정보를 참고 문서를 바탕으로 자세히 설명해주세요.",
        "개념정의": "참고 문서를 바탕으로 요청한 개념의 정의를 명확하게 설명해주세요.",
        "비교분석": "참고 문서를 바탕으로 요청한 개념들의 차이점과 관계를 비교하여 설명해주세요.",
        "기술질문": "참고 문서를 바탕으로 작동 방식이나 방법론을 설명해주세요.",
        "기타": "참고 문서를 바탕으로 사용자의 질문에 답변해주세요.",
    }

    def llm_call(state: IntentState):
        intent = state.get("intent", "기타")
        prompt = _prompts.get(intent, _prompts["기타"])
        response = model.invoke(state["messages"] + [SystemMessage(content=prompt)])
        return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

    return llm_call


def make_llm_call_without_docs(model):
    def llm_call_without_docs(state: IntentState):
        response = model.invoke(state["messages"] + [
            SystemMessage(content="관련 문서를 찾을 수 없어 일반적인 지식으로 답변합니다. 사용자의 질문에 최선을 다해 답변해주세요.")
        ])
        return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

    return llm_call_without_docs
