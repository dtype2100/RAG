# https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
# https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

# client = QdrantClient(url="http://localhost:6333")
client = QdrantClient(":memory:")



text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    separators=["*", "**", "#", "##", "###", "\n"]
)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


with open("./md_pages") as f:
    md_file = f.read()
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)


md_header_splits = markdown_splitter.split_text(md_file)

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings(client="http://localhost:8080")


vector_size = len(embeddings.embed_query(md_header_splits[0].page_content))
if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )


    
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)


vector_store.add_documents(md_header_splits)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3},
)

retriever.invoke("query")

