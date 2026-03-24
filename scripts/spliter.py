"""
md_pages 디렉터리 내 마크다운(.md) 파일을 LangChain으로 로드한 뒤
구조 인식(헤더) 1차 분할 + 크기 초과분만 재귀 2차 분할(하이브리드) 또는
단순 RecursiveCharacterTextSplitter로 분할하고, 결과를 JSONL(권장) 또는 JSON으로 저장하는 스크립트.

JSONL: 한 줄에 하나의 청크(JSON 객체), 스트리밍·추가 저장·메모리 효율에 유리. LangChain JSONLoader(json_lines=True)로 로드 가능.
"""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 기본 md_pages 경로 (스크립트 기준 상대 경로)
DEFAULT_MD_DIR = Path(__file__).resolve().parent / "md_pages"
# 분할 결과 기본 저장 경로 (JSONL 권장: 스트리밍·추가·대용량에 유리)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "splits.jsonl"

# 구조 인식 1차 분할용 마크다운 헤더 (h1, h2, h3)
DEFAULT_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def _output_path_with_timestamp(path: str | Path) -> Path:
    """파일 경로의 stem과 확장자 사이에 현재 시간(YmdHMS)을 넣어 반환합니다. 예: splits.jsonl → splits_20260314120530.jsonl"""
    path = Path(path)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return path.parent / f"{path.stem}_{ts}{path.suffix}"


def _get_length_function(use_tokens: bool = False):
    """
    청크 길이 계산 함수. use_tokens=True이고 tiktoken 있으면 토큰 수, 아니면 문자 수.
    """
    if use_tokens:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")

            def token_len(text: str) -> int:
                return len(enc.encode(text))
            return token_len
        except Exception:
            pass
    return len


def load_md_documents(md_dir: str | Path) -> list:
    """
    디렉터리 내 모든 .md 파일을 LangChain Document로 로드합니다.
    DirectoryLoader + TextLoader 사용 (unstructured 의존성 없음).
    """
    md_dir = Path(md_dir)
    if not md_dir.is_dir():
        raise FileNotFoundError(f"디렉터리가 없습니다: {md_dir}")

    loader = DirectoryLoader(
        str(md_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_by_markdown_headers(
    documents: list,
    headers_to_split_on: list[tuple[str, str]] | None = None,
    strip_headers: bool = True,
) -> list[Document]:
    """
    Document 리스트를 마크다운 헤더(#, ##, ###) 기준으로 1차 분할합니다.
    원본 metadata(source 등)는 각 청크에 유지됩니다.
    """
    headers_to_split_on = headers_to_split_on or DEFAULT_HEADERS_TO_SPLIT_ON
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=strip_headers,
    )
    result = []
    for doc in documents:
        try:
            splits = md_splitter.split_text(doc.page_content)
            for s in splits:
                meta = {**doc.metadata, **(s.metadata or {})}
                result.append(Document(page_content=s.page_content, metadata=meta))
        except Exception:
            result.append(doc)
    return result


def split_oversized_with_recursive(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    length_function=None,
    separators: list[str] | None = None,
) -> list[Document]:
    """
    문서 리스트 중 chunk_size를 초과하는 것만 RecursiveCharacterTextSplitter로 2차 분할합니다.
    메타데이터(header, source 등)는 하위 청크에 그대로 상속합니다.
    """
    length_function = length_function or len
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=separators or ["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    result = []
    for doc in documents:
        if length_function(doc.page_content) <= chunk_size:
            result.append(doc)
            continue
        sub = text_splitter.split_documents([doc])
        for s in sub:
            meta = {**doc.metadata, **(s.metadata or {})}
            result.append(Document(page_content=s.page_content, metadata=meta))
    return result


def split_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
    length_function=None,
) -> list[Document]:
    """
    Document 리스트를 RecursiveCharacterTextSplitter로만 분할합니다.
    (전략: recursive_only 일 때 사용)
    """
    length_function = length_function or len
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=separators or ["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def run(
    md_dir: str | Path | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    output_path: str | Path | None = None,
    output_format: str = "jsonl",
    strategy: str = "structure_hybrid",
    use_token_length: bool = False,
):
    """
    md 디렉터리 로드 → 분할 후 분할된 Document 리스트를 반환합니다.

    output_format: "jsonl"(기본, 권장) 또는 "json"
    strategy:
      - "structure_hybrid": 헤더 기준 1차 분할 후, 크기 초과분만 재귀 2차 분할 (권장)
      - "recursive_only": 기존처럼 RecursiveCharacterTextSplitter만 사용
    """
    md_dir = md_dir or DEFAULT_MD_DIR
    docs = load_md_documents(md_dir)
    if not docs:
        print(f"⚠️ {md_dir} 에서 .md 파일을 찾지 못했습니다.")
        return []

    length_fn = _get_length_function(use_tokens=use_token_length)

    if strategy == "structure_hybrid":
        structure_chunks = split_by_markdown_headers(docs)
        splits = split_oversized_with_recursive(
            structure_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
        )
        print(f"✅ 로드 문서 수: {len(docs)}, 1차(헤더) 청크: {len(structure_chunks)}, 최종 청크 수: {len(splits)}")
    else:
        splits = split_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
        )
        print(f"✅ 로드 문서 수: {len(docs)}, 분할 후 청크 수: {len(splits)}")

    if output_path is not None:
        path_to_save = _output_path_with_timestamp(output_path)
        saved = save_splits(splits, path_to_save, output_format=output_format)
        print(f"📁 분할 결과 저장: {saved}")

    return splits


def save_splits(
    splits: list,
    output_path: str | Path,
    output_format: str = "jsonl",
) -> Path:
    """
    분할된 Document 리스트를 JSONL(기본) 또는 JSON 파일로 저장합니다.

    output_format:
      - "jsonl": 한 줄에 하나의 JSON 객체 ({"page_content": str, "metadata": dict}).
        스트리밍·추가·대용량에 유리. LangChain JSONLoader(file_path, json_lines=True, jq_schema=".", content_key="page_content", metadata_func=lambda x: x.get("metadata", {})) 로 로드 가능.
      - "json": 단일 JSON 배열. 기존 호환용.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in splits
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path

    # jsonl: one JSON object per line (no trailing newline after last line is ok; many tools allow it)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(splits):
            line = json.dumps(
                {"page_content": doc.page_content, "metadata": doc.metadata},
                ensure_ascii=False,
            )
            f.write(line)
            if i < len(splits) - 1:
                f.write("\n")
    return output_path


def load_splits(file_path: str | Path) -> list[Document]:
    """
    save_splits로 저장한 JSONL 또는 JSON 파일에서 Document 리스트를 복원합니다.
    확장자 .jsonl 이면 JSONL, .json 이면 JSON 배열로 판단합니다.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    docs = []
    if file_path.suffix.lower() == ".jsonl":
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
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                docs.append(
                    Document(
                        page_content=item.get("page_content", ""),
                        metadata=item.get("metadata", {}),
                    )
                )
        else:
            raise ValueError("JSON 파일은 Document 배열(list)이어야 합니다.")

    return docs


if __name__ == "__main__":
    splits = run(
        md_dir=r".\md_pages",
        chunk_size=500,
        chunk_overlap=50,
        output_path=DEFAULT_OUTPUT_PATH,
        strategy="structure_hybrid",
    )
    for i, doc in enumerate(splits[:3]):
        print(f"\n--- Chunk {i + 1} (source: {doc.metadata.get('source', '')}) ---")
        print(doc.page_content[:200].strip(), "...")
