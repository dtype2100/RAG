from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from app.agents.intent_classifier import (
    IntentState,
    get_model,
    make_intent_classifier,
    make_llm_call,
    make_llm_call_without_docs,
    make_retriever_call,
    make_verify_retrieval,
    should_retry_retrieval,
)


def build_intent_graph(vectorstore: InMemoryVectorStore):
    model = get_model()

    graph = StateGraph(IntentState)
    graph.add_node("intent_classifier", make_intent_classifier(model))
    graph.add_node("retriever_call", make_retriever_call(vectorstore))
    graph.add_node("verify_retrieval", make_verify_retrieval(model))
    graph.add_node("llm_call", make_llm_call(model))
    graph.add_node("llm_call_without_docs", make_llm_call_without_docs(model))

    graph.add_edge(START, "intent_classifier")
    graph.add_edge("intent_classifier", "retriever_call")
    graph.add_edge("retriever_call", "verify_retrieval")
    graph.add_conditional_edges(
        "verify_retrieval",
        should_retry_retrieval,
        {
            "proceed_to_llm": "llm_call",
            "retry_retrieval": "retriever_call",
            "proceed_without_docs": "llm_call_without_docs",
        },
    )
    graph.add_edge("llm_call", END)
    graph.add_edge("llm_call_without_docs", END)

    return graph.compile()
