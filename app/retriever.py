from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.embeddings import get_embeddings


def build_vectorstore(documents: List[str]) -> InMemoryVectorStore:
    docs = [Document(page_content=text) for text in documents]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)
    return InMemoryVectorStore.from_documents(splits, get_embeddings())


def get_retriever(vectorstore: InMemoryVectorStore, k: int = None):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.RETRIEVAL_K},
    )
