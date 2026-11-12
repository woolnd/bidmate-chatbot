import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_search_b_experiments import (  # noqa: E402
    calculate_metrics,
    compact,
    get_agency_matches,
    run_experiment_b,
    summarize,
)


def test_positive_question_metrics():
    result = calculate_metrics({"A", "B"}, ["A", "WRONG"])

    assert result["true_positive_ids"] == ["A"]
    assert result["false_positive_ids"] == ["WRONG"]
    assert result["false_negative_ids"] == ["B"]
    assert result["recall"] == 0.5
    assert result["top1_correct"] is True


def test_negative_question_with_no_result_is_correct():
    result = calculate_metrics(set(), [])

    assert result["recall"] == 0.0
    assert result["top1_correct"] is True
    assert result["false_positive_ids"] == []


def test_negative_question_with_result_is_false_positive():
    result = calculate_metrics(set(), ["WRONG"])

    assert result["recall"] == 0.0
    assert result["top1_correct"] is False
    assert result["false_positive_ids"] == ["WRONG"]


def test_precision_at_k_uses_fixed_k():
    result = calculate_metrics({"A"}, ["A"], k=3)

    assert result["precision_at_k"] == 1 / 3


def test_invalid_k_raises_error():
    try:
        calculate_metrics({"A"}, ["A"], k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("k가 0이면 ValueError가 발생해야 합니다.")


def test_summary_separates_positive_and_negative_questions():
    rows = [
        {
            "evaluation_type": "supervised",
            "ground_truth_status": "OK",
            "expected_ids": ["A"],
            "a_ids": ["A"],
        },
        {
            "evaluation_type": "supervised",
            "ground_truth_status": "OK",
            "expected_ids": [],
            "a_ids": ["WRONG"],
        },
    ]

    result = summarize(rows, "a")

    assert result["evaluated_questions"] == 2
    assert result["positive_questions"] == 1
    assert result["negative_questions"] == 1

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0

    assert result["top1_accuracy"] == 0.5
    assert result["micro_precision"] == 0.5
    assert result["micro_recall"] == 1.0
    assert result["negative_accuracy"] == 0.0
    assert result["negative_false_positive"] == 1
    assert result["mean_recall"] == 1.0


def test_unresolved_and_agency_pool_are_excluded():
    rows = [
        {
            "evaluation_type": "supervised",
            "ground_truth_status": "UNRESOLVED",
            "expected_ids": ["A"],
            "a_ids": [],
        },
        {
            "evaluation_type": "agency_pool",
            "ground_truth_status": "OK",
            "expected_ids": ["B"],
            "a_ids": ["B"],
        },
    ]

    result = summarize(rows, "a")

    assert result["evaluated_questions"] == 0
    assert result["positive_questions"] == 0
    assert result["negative_questions"] == 0
    assert result["micro_precision"] == 0.0
    assert result["micro_recall"] == 0.0


def test_empty_dataset_returns_zero_metrics():
    result = summarize([], "a")

    assert result["evaluated_questions"] == 0
    assert result["negative_accuracy"] == 0.0
    assert result["negative_false_positive"] == 0
    assert result["mean_recall"] == 0.0


def test_compact_removes_legal_entity_suffix():
    assert compact("인천공항운영서비스(주)") == "인천공항운영서비스"
    assert compact("인천공항운영서비스㈜") == "인천공항운영서비스"
    assert compact("한국가스공사 주식회사") == "한국가스공사"


def test_q05_agency_is_detected_despite_legal_suffix():
    from types import SimpleNamespace

    project = SimpleNamespace(
        project_id="인천공항운영서비스(주)_차세대 ERP 구축.hwp",
        agency_name="인천공항운영서비스(주)",
        project_name="인천공항운영서비스㈜ 차세대 ERP시스템 구축 사업",
    )

    agencies = get_agency_matches(
        "인천공항운영서비스 차세대 ERP 시스템 구축",
        [project],
    )

    assert agencies == ["인천공항운영서비스(주)"]


def test_q05_search_returns_correct_agency_project():
    from types import SimpleNamespace

    project = SimpleNamespace(
        project_id="인천공항운영서비스(주)_차세대 ERP 구축.hwp",
        agency_name="인천공항운영서비스(주)",
        project_name="인천공항운영서비스㈜ 차세대 ERP시스템 구축 사업",
    )

    question = "인천공항운영서비스 차세대 ERP 시스템 구축"

    result_ids, strategy = run_experiment_b(
        question,
        [project],
        ["한국가스공사_차세대 ERP 구축.hwp"],
    )

    assert result_ids == [project.project_id]
    assert strategy == "AGENCY_AND_PROJECT"
