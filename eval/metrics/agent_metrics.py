"""스키마 매핑 에이전트 정확도 메트릭."""

from typing import Any


def exact_match(predicted: str, expected: str) -> bool:
    return predicted.strip().lower() == expected.strip().lower()


def compute_schema_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """예측된 매핑과 정답을 비교해 정확도를 계산한다.

    Args:
        results: 각 항목이 {"source_col", "predicted", "expected", ...}인 리스트

    Returns:
        accuracy, total, correct, incorrect, errors 포함 딕셔너리
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "correct": 0, "incorrect": 0, "errors": []}

    correct = sum(1 for r in results if exact_match(r["predicted"], r["expected"]))
    errors = [r for r in results if not exact_match(r["predicted"], r["expected"])]

    return {
        "accuracy": round(correct / total, 4),
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "errors": [
            {
                "source_col": e["source_col"],
                "predicted": e["predicted"],
                "expected": e["expected"],
            }
            for e in errors
        ],
    }
