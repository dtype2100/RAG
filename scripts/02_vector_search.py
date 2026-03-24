# https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
# https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader

client = QdrantClient(url="http://localhost:6333")
# client = QdrantClient(":memory:")

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings(client="http://localhost:8080")
    
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3},
)

print(retriever.invoke("혜움 연혁 알려줘"))

