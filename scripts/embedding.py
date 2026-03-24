"""
JSONL로 저장된 분할 청크를 로드하고, TEI(Text Embeddings Inference) 엔드포인트로 임베딩한 뒤
Chroma 벡터 DB에 저장하는 스크립트. RAG 검색/리트리버에 사용 가능.

TEI: 기본 http://localhost:8080 (compose.tei.yaml TEI_PORT). 배치·지연 최적화에 유리.
"""
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

# 스크립트 기준 경로
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLITS_PATH = SCRIPT_DIR / "splits_20260314022536.jsonl"
DEFAULT_PERSIST_DIR = SCRIPT_DIR / "chroma_db"

# TEI 기본 설정 (infra/model-serving/embedding/compose.tei.yaml)
DEFAULT_TEI_BASE_URL = os.getenv("TEI_BASE_URL", "http://localhost:8080")


def load_splits_jsonl(file_path: str | Path) -> list[Document]:
    """JSONL 파일에서 Document 리스트를 로드합니다."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj.get("page_content", ""),
                    metadata=obj.get("metadata", {}),
                )
            )
    return docs


def _tei_embeddings(base_url: str):
    """TEI 엔드포인트 URL로 HuggingFaceEndpointEmbeddings 래퍼 생성 (LangChain TEI 연동)."""
    url = base_url.rstrip("/")
    return HuggingFaceEndpointEmbeddings(client=f"{url}/")


def build_vectorstore(
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    tei_base_url: str = DEFAULT_TEI_BASE_URL,
    collection_name: str = "heum_splits",
) -> Chroma:
    """
    JSONL 청크를 로드 → TEI로 임베딩 → Chroma에 저장 후 VectorStore를 반환합니다.

    - splits_path: 분할 청크 JSONL 경로
    - persist_directory: Chroma DB 저장 디렉터리 (재실행 시 로드 가능)
    - tei_base_url: TEI 서버 주소 (기본 http://localhost:8080, compose.tei.yaml TEI_PORT)
    """
    docs = load_splits_jsonl(splits_path)
    if not docs:
        raise ValueError(f"청크가 비어 있습니다: {splits_path}")

    embeddings = _tei_embeddings(tei_base_url)

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )
    return vectorstore


def load_vectorstore(
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    tei_base_url: str = DEFAULT_TEI_BASE_URL,
    collection_name: str = "heum_splits",
) -> Chroma:
    """이미 저장된 Chroma DB를 로드합니다. 검색 시 쿼리 임베딩에 동일 TEI URL 사용."""
    embeddings = _tei_embeddings(tei_base_url)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    print("TEI 임베딩 및 Chroma 저장 시작...")
    print(f"  splits: {DEFAULT_SPLITS_PATH}")
    print(f"  TEI: {DEFAULT_TEI_BASE_URL}")
    print(f"  persist: {DEFAULT_PERSIST_DIR}")

    try:
        vs = build_vectorstore(
            splits_path=DEFAULT_SPLITS_PATH,
            persist_directory=DEFAULT_PERSIST_DIR,
            tei_base_url=DEFAULT_TEI_BASE_URL,
        )
        print("완료. 벡터스토어가 저장되었습니다.")

        # 간단 검색 테스트
        retriever = vs.as_retriever(k=2)
        for doc in retriever.invoke("혜움 채용 절차"):
            print(f"  - {doc.metadata.get('source', '')}: {doc.page_content[:80]}...")
    except Exception as e:
        if "Connection" in str(e) or "ECONNREFUSED" in str(e) or "8080" in str(e):
            print("오류: TEI에 연결할 수 없습니다.")
            print("  - infra/model-serving/embedding: docker compose -f compose.tei.yaml up -d")
            print("  - 환경 변수: TEI_BASE_URL (기본 http://localhost:8080)")
        raise

# ---------- 추가로 고려하면 좋은 것 ----------
# 1. 환경 변수: .env 에 TEI_BASE_URL 설정 후 python-dotenv 로드
# 2. 재실행 시: 이미 chroma_db 가 있으면 load_vectorstore() 로 로드 후 검색만 하면 됨 (임베딩 재계산 불필요)
# 3. RAG 파이프라인: 이 벡터스토어 + Ollama LLM 으로 RetrievalQAChain 또는 create_retrieval_chain 구성
# 4. 리랭킹: 검색 결과가 많을 때 TEI 리랭커 또는 Cross-Encoder로 정확도 향상
# 5. 최신 splits 반영: spliter.py 로 새 JSONL 생성 후 이 스크립트 다시 실행하면 collection 덮어쓰기 또는 새 collection_name 으로 구분
