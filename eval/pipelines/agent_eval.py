"""스키마 매핑 에이전트 (schema_graph) 정확도 평가 스크립트.

실행 전 필요 서비스:
  - Ollama LLM 서버 (기본 http://localhost:11434, qwen2.5-coder:3b 모델)
      cd infra/model-serving/llm && docker compose -f compose.ollama.yaml up -d

사용법:
  cd <project_root>
  python -m eval.pipelines.agent_eval [--dataset /path/to/schema_dataset.json]

평가 결과:
  eval/report/agent_eval_<timestamp>.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가 (app 패키지 임포트용)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.graphs.schema_graph import build_schema_graph
from eval.metrics.agent_metrics import compute_schema_accuracy

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "schema_dataset.json"
REPORT_DIR = Path(__file__).resolve().parent.parent / "report"


def run_agent_eval(
    dataset_path: Path = DATASET_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """schema_graph를 실행하고 Exact Match 정확도를 평가한다.

    Args:
        dataset_path: schema_dataset.json 경로

    Returns:
        (metrics_dict, detail_list) 튜플 (eval/report/ 에도 저장됨)
    """
    # 1. 테스트 데이터셋 로드
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"[1/3] 테스트 케이스 {len(test_cases)}개 로드")

    # 2. schema_graph 빌드 및 실행
    print("[2/3] schema_graph 빌드 및 실행 중 ... Ollama 서버가 필요합니다.")
    graph = build_schema_graph()

    results = []
    for i, case in enumerate(test_cases):
        source_col = case["source_col"]
        sample_value = case.get("sample_value", "")
        expected = case["expected"]

        state = {
            "source_col": source_col,
            "sample_value": sample_value,
            "candidates": [],
            "final_mapping": "",
            "reasoning": "",
        }
        result = graph.invoke(state)
        predicted = result.get("final_mapping", "Error")

        match = "O" if predicted == expected else "X"
        print(f"      [{i+1}/{len(test_cases)}] {source_col!r:20} → {predicted!r:20} (예상: {expected!r}) {match}")

        results.append({
            "source_col": source_col,
            "sample_value": sample_value,
            "predicted": predicted,
            "expected": expected,
            "reasoning": result.get("reasoning", ""),
        })

    # 3. 정확도 계산 및 저장
    print("[3/3] 정확도 계산")
    metrics = compute_schema_accuracy(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"agent_eval_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, ensure_ascii=False, indent=2)

    print(f"\n정확도: {metrics['accuracy']:.1%}  ({metrics['correct']}/{metrics['total']})")
    if metrics["errors"]:
        print("오답 목록:")
        for e in metrics["errors"]:
            print(f"  {e['source_col']:20} → 예측: {e['predicted']:20} | 정답: {e['expected']}")
    print(f"결과 저장 → {report_path}")
    return metrics, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="스키마 매핑 에이전트 정확도 평가")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="평가 데이터셋 JSON 파일 경로",
    )
    args = parser.parse_args()
    run_agent_eval(dataset_path=args.dataset)
