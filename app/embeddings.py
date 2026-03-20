from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from app.config import settings


def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    return HuggingFaceEndpointEmbeddings(client=settings.EMBEDDING_ENDPOINT)
