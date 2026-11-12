"""NER 파생 데이터의 기관 분리와 이름 구간 정렬을 확인한다."""

from copy import deepcopy

import pytest

from scripts.build_search_b_ner_training import augment_training
from scripts.train_search_b_ner import align_labels, decode_spans, span_metrics


def test_augmentation_preserves_validation_and_input_rows():
    rows = [
        {
            "text": "가나다 문서관리",
            "spans": [{"start": 4, "end": 8, "label": "PROJECT"}],
            "group": "가나다",
            "split": "train",
        },
        {"text": "라마바 사업", "spans": [], "group": "라마바", "split": "validation"},
    ]
    original = deepcopy(rows)
    result = augment_training(rows, {"문서관리 필수 조건"})
    assert rows == original
    assert [r for r in result if r["split"] == "validation"] == [rows[1]]
    additional = result[len(rows) :]
    assert additional
    assert all(r["split"] == "train" and r["group"] == "가나다" for r in additional)
    assert "문서관리 필수 조건" not in {r["text"] for r in additional}


def test_alignment_does_not_assign_outside_label_to_a_crossing_token():
    offsets = [(0, 0), (0, 2), (2, 4), (0, 0)]
    spans = [{"start": 1, "end": 4, "label": "PROJECT"}]
    assert align_labels(offsets, spans) == [-100, -100, 3, -100]


def test_overlapping_gold_spans_are_rejected():
    spans = [{"start": 0, "end": 3, "label": "AGENCY"}, {"start": 2, "end": 4, "label": "PROJECT"}]
    with pytest.raises(ValueError, match="겹친다"):
        align_labels([(0, 4)], spans)


def test_span_metrics_require_same_start_end_and_label():
    predicted = decode_spans([(0, 2), (2, 4), (5, 7)], [3, 4, 1])
    assert predicted == [(0, 4, "PROJECT"), (5, 7, "AGENCY")]
    metrics = span_metrics([[(0, 4, "AGENCY")]], [predicted])
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 2, 1)


def test_augmentation_blocks_evaluation_text_after_whitespace_normalization():
    rows = [
        {
            "text": "가나다 문서관리",
            "spans": [{"start": 4, "end": 8, "label": "PROJECT"}],
            "group": "가나다",
            "split": "train",
        }
    ]
    result = augment_training(rows, {"문서관리  필수 조건"})
    assert "문서관리 필수 조건" not in {r["text"] for r in result[1:]}
