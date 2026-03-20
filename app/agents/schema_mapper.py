import json
from typing import List

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import TypedDict

from app.config import settings

_embed_model: SentenceTransformer = None
_target_embeddings: np.ndarray = None

MAPPING_PROMPT = PromptTemplate(
    template="""You are a Data Engineer. Select the best Target Column from Candidates.
[Input]
- Source: "{source_col}"
- Sample: "{sample_value}"
- Candidates: {candidates}

Analyze and select ONE target column.
Return JSON ONLY.

Example:
{{ "selected_column": "col_name", "reason": "why" }}""",
    input_variables=["source_col", "sample_value", "candidates"],
)


class MappingState(TypedDict):
    source_col: str
    sample_value: str
    candidates: List[str]
    final_mapping: str
    reasoning: str


def _get_embed_model():
    global _embed_model, _target_embeddings
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.EMBED_MODEL_NAME)
        _target_embeddings = _embed_model.encode(settings.TARGET_SCHEMA)
    return _embed_model, _target_embeddings


def get_llm() -> ChatOllama:
    return ChatOllama(model=settings.SCHEMA_MODEL, temperature=0, format="json")


def retriever_node(state: MappingState) -> dict:
    embed_model, target_embeddings = _get_embed_model()
    source_vec = embed_model.encode([state["source_col"]])
    similarities = cosine_similarity(source_vec, target_embeddings)[0]
    top_indices = np.argsort(similarities)[-settings.TOP_K_CANDIDATES:][::-1]
    candidates = [settings.TARGET_SCHEMA[i] for i in top_indices]
    return {"candidates": candidates}


def slm_reasoning_node(state: MappingState) -> dict:
    chain = MAPPING_PROMPT | get_llm()
    try:
        response = chain.invoke({
            "source_col": state["source_col"],
            "sample_value": state["sample_value"],
            "candidates": state["candidates"],
        })
        result = json.loads(response.content)
        return {
            "final_mapping": result.get("selected_column", "Unknown"),
            "reasoning": result.get("reason", "No reason"),
        }
    except Exception as e:
        return {"final_mapping": "Error", "reasoning": str(e)}
