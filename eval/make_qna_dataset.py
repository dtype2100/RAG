import os
from datetime import datetime

from ragas import Dataset

# 스크립트 위치 기준으로 데이터셋 저장 경로 고정 (실행 cwd 무관)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
DEFAULT_CSV_PATH = os.path.join(DATASETS_DIR, "test_dataset.csv")


def load_dataset():
    """테스트용 평가 데이터셋을 로드하거나 생성한다.
    이미 test_dataset.csv가 있으면 타임스탬프 이름으로 새 데이터셋을 만들고,
    없으면 test_dataset으로 저장한다.
    """
    if not os.path.exists(DEFAULT_CSV_PATH):
        name = "test_dataset"
    else:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_test_dataset"

    dataset = Dataset(
        name=name,
        backend="local/csv",
        root_dir=SCRIPT_DIR,
    )

    data_samples = [
        {
            "question": "What is Ragas?",
            "grading_notes": "Ragas is an evaluation framework for LLM applications",
        },
        {
            "question": "How do metrics work?",
            "grading_notes": "Metrics evaluate the quality and performance of LLM responses",
        },
        # Add more test cases here
    ]

    for sample in data_samples:
        dataset.append(sample)

    dataset.save()
    return dataset

if __name__ == "__main__":
    load_dataset()
