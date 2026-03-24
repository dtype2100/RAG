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


markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)


md_dir = (Path(__file__).resolve().parent / "md_pages").resolve()
loader = DirectoryLoader(
    str(md_dir),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()
if not docs:
    raise FileNotFoundError(f"No .md files found under: {md_dir}")

# 모든 md 파일을 "파일 단위"로 헤더 분할 -> 재귀 청커로 추가 분할해서
# Document 결과를 통째로 Qdrant에 넣는다.
all_docs = []
for doc in tqdm(docs, desc="MD 헤더 분할", unit="file"):
    # DirectoryLoader의 metadata['source']는 대개 전체 경로이므로, 파일명만 남긴다.
    base_meta = dict(doc.metadata or {})
    source_path = str(base_meta.get("source", ""))
    base_meta["source"] = Path(source_path).name if source_path else base_meta.get("source", "")

    # 1) 헤더 기준으로 분할
    header_splits = markdown_splitter.split_text(doc.page_content)

    # 2) 헤더 블록을 다시 재귀 청커로 쪼개서 메타데이터 유지
    for header_doc in header_splits:
        header_meta = dict(header_doc.metadata or {})
        merged_meta = {**base_meta, **header_meta}
        all_docs.extend(
            text_splitter.create_documents(
                [header_doc.page_content],
                metadatas=[merged_meta],
            )
        )

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings(client="http://localhost:8080")


if not all_docs:
    raise ValueError("No documents produced from md splitting; check md_pages format.")

vector_size = len(embeddings.embed_query(all_docs[0].page_content))
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


batch_size = 32
for i in tqdm(range(0, len(all_docs), batch_size), desc="Qdrant 업로드", unit="chunk"):
    batch = all_docs[i : i + batch_size]
    vector_store.add_documents(batch)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3},
)

retriever.invoke("혜움 연혁 알려줘")

