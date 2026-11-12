"""두 방식 비교의 입력 검증과 집계 조건을 확인한다."""

import math
import unicodedata

import pytest

from scripts.compare_search_b_models import has_entity, summarize, validate_cases


def test_diagnostic_cases_reject_normalized_duplicate_questions():
    cases = [
        {
            "case_id": "one",
            "question": "가나다기관 예산",
            "expected_ids": ["one"],
            "split": "development",
        },
        {
            "case_id": "two",
            "question": unicodedata.normalize("NFD", "가나다기관 예산"),
            "expected_ids": ["one"],
            "split": "development",
        },
    ]
    with pytest.raises(ValueError, match="중복"):
        validate_cases(cases, {"one"})


def test_diagnostic_runner_does_not_accept_holdout_questions():
    cases = [{"case_id": "one", "question": "기관 예산", "expected_ids": [], "split": "holdout"}]
    with pytest.raises(ValueError, match="개발 질문"):
        validate_cases(cases, set())


@pytest.mark.parametrize(
    "score,end,label",
    [(math.nan, 2, "AGENCY"), (0.9, 20, "AGENCY"), (0.2, 2, "AGENCY"), (0.9, 2, "O")],
)
def test_entity_gate_rejects_invalid_or_unaccepted_spans(score, end, label):
    entities = [{"entity_group": label, "score": score, "start": 0, "end": end}]
    assert not has_entity("기관 예산", entities, 0.3)


def test_entity_gate_accepts_project_span_without_an_agency():
    entities = [{"entity_group": "PROJECT", "score": 0.8, "start": 0, "end": 4}]
    assert has_entity("문서관리 예산", entities, 0.3)


def test_comparison_counts_documents_and_questions_separately():
    rows = [
        {"expected_ids": ["one", "two"], "predicted_ids": ["one", "extra"], "selector_ms": 1.0},
        {"expected_ids": [], "predicted_ids": [], "selector_ms": 2.0},
    ]
    summary = summarize(rows)
    assert summary["exact_set_accuracy"] == 0.5
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 1, 1)
    assert summary["f1"] == 0.5


def test_training_overlap_uses_the_same_question_normalization():
    cases = [
        {
            "case_id": "one",
            "question": "가나다기관   ERP 예산",
            "expected_ids": ["one"],
            "split": "development",
        }
    ]
    with pytest.raises(ValueError, match="학습"):
        validate_cases(cases, {"one"}, training_texts={"가나다기관 ERP 예산"})
