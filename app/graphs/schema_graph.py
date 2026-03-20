from langgraph.graph import END, StateGraph

from app.agents.schema_mapper import MappingState, retriever_node, slm_reasoning_node


def build_schema_graph():
    workflow = StateGraph(MappingState)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoner", slm_reasoning_node)
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "reasoner")
    workflow.add_edge("reasoner", END)
    return workflow.compile()
