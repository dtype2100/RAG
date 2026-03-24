"""RAG 파이프라인 (intent_graph) RAGAS 평가 스크립트.

실행 전 필요 서비스:
  - TEI 임베딩 서버 (기본 http://127.0.0.1:8080/)
      cd infra/model-serving/embedding && docker compose -f compose.tei.yaml up -d
  - Ollama LLM 서버 (기본 http://localhost:11434, smollm2 모델)
      cd infra/model-serving/llm && docker compose -f compose.ollama.yaml up -d
  - .env 파일에 GOOGLE_API_KEY 설정 (RAGAS 평가용)

사용법:
  cd <project_root>
  python -m eval.pipelines.rag_eval [--splits /path/to/splits.jsonl]

평가 결과:
  eval/report/rag_eval_<timestamp>.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (app, rag 패키지 임포트용)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.messages import HumanMessage
from ragas import EvaluationDataset, SingleTurnSample, evaluate

from app.graphs.intent_graph import build_intent_graph
from app.retriever import build_vectorstore
from eval.metrics.rag_metrics import get_rag_metrics

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "rag_dataset.json"
REPORT_DIR = Path(__file__).resolve().parent.parent / "report"

# splits JSONL 기본 경로 (rag/ 또는 db/ 중 존재하는 쪽 사용)
_CANDIDATE_SPLITS = [
    PROJECT_ROOT / "rag" / "splits_20260314022536.jsonl",
    PROJECT_ROOT / "db" / "splits_20260314022536.jsonl",
]
DEFAULT_SPLITS_PATH = next((p for p in _CANDIDATE_SPLITS if p.exists()), _CANDIDATE_SPLITS[0])


def _load_page_contents(path: Path) -> list[str]:
    """JSONL 파일에서 page_content 문자열 리스트를 로드한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"splits 파일이 없습니다: {path}\n"
            "rag/spliter.py를 먼저 실행하거나 --splits 옵션으로 경로를 지정하세요."
        )
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            content = obj.get("page_content", "")
            if content:
                docs.append(content)
    return docs


def run_rag_eval(
    dataset_path: Path = DATASET_PATH,
    splits_path: Path = DEFAULT_SPLITS_PATH,
) -> "pd.DataFrame":
    """intent_graph를 실행하고 RAGAS 메트릭으로 평가한다.

    Args:
        dataset_path: rag_dataset.json 경로
        splits_path:  벡터스토어를 구축할 JSONL splits 파일 경로

    Returns:
        RAGAS 평가 결과 DataFrame (eval/report/ 에도 저장됨)
    """
    # 1. 테스트 데이터셋 로드
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = [c for c in json.load(f) if "user_input" in c]
    print(f"[1/5] 테스트 케이스 {len(test_cases)}개 로드")

    # 2. 벡터스토어 구축 (TEI 서버 필요)
    print(f"[2/5] 벡터스토어 구축 중 ({splits_path.name}) ... TEI 서버가 필요합니다.")
    documents = _load_page_contents(splits_path)
    vectorstore = build_vectorstore(documents)
    print(f"      문서 {len(documents)}개 → 벡터스토어 완료")

    # 3. intent_graph 빌드
    print("[3/5] intent_graph 빌드")
    graph = build_intent_graph(vectorstore)

    # 4. 각 케이스 실행 → SingleTurnSample 수집
    print("[4/5] 그래프 실행 중...")
    samples = []
    for i, case in enumerate(test_cases):
        user_input = case["user_input"]
        reference = case["reference"]
        print(f"      [{i+1}/{len(test_cases)}] {user_input[:60]}")

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "llm_calls": 0,
            "intent": "",
            "retrieved_docs": [],
            "retrieval_attempts": 0,
        }
        result = graph.invoke(initial_state)

        # 최종 LLM 응답 (마지막 메시지)
        response = result["messages"][-1].content if result["messages"] else ""

        # verify_retrieval 이후 필터링된 문서 (없으면 retrieved_docs fallback)
        filtered = result.get("filtered_docs") or result.get("retrieved_docs", [])
        retrieved_contexts = [doc.page_content for doc in filtered]

        samples.append(
            SingleTurnSample(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
        )

    # 5. RAGAS 평가
    print("[5/5] RAGAS 평가 실행 중 ... Google API 호출")
    eval_dataset = EvaluationDataset(samples=samples)
    metrics = get_rag_metrics()
    result_df = evaluate(eval_dataset, metrics=metrics).to_pandas()

    # 결과 저장
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"rag_eval_{timestamp}.csv"
    result_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\n평가 완료 → {report_path}")
    print(result_df.mean(numeric_only=True).to_string())
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 파이프라인 RAGAS 평가")
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS_PATH,
        help="벡터스토어 구축에 사용할 JSONL splits 파일 경로",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="평가 데이터셋 JSON 파일 경로",
    )
    args = parser.parse_args()
    run_rag_eval(dataset_path=args.dataset, splits_path=args.splits)
