"""점수 기록(scores.csv) 테스트. 파일 쓰기만 확인하고 OpenAI는 부르지 않는다."""

import csv
import sys
from pathlib import Path

import pytest

# run_evaluation.py는 evaluation 폴더 안의 answer_grader, score_search를 바로 불러온다
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from evaluation import run_evaluation  # noqa: E402

LINE = {
    "시각": "2026-11-16 10:00",
    "질문 수": 50,
    "답변_정확도": 0.74,
    "평균_초": 3.8,
    "가장_느린_초": 9.8,
    "검색_실제흐름": 0.657,
    "검색_정답문서": 0.714,
    "메모": "기준점",
}


def test_save_score_line_쌓기(tmp_path, monkeypatch):
    """새 파일이면 칸 이름을 쓰고, 다음 실행부터는 줄만 붙인다."""
    scores_path = tmp_path / "scores.csv"
    monkeypatch.setattr(run_evaluation, "SCORES_PATH", scores_path)

    run_evaluation.save_score_line(LINE)
    run_evaluation.save_score_line({**LINE, "메모": "두 번째"})

    with open(scores_path, encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert tuple(rows[0]) == run_evaluation.SCORE_COLUMNS
    assert [row["메모"] for row in rows] == ["기준점", "두 번째"]


def test_save_score_line_옛_칸이면_멈춤(tmp_path, monkeypatch):
    """옛 칸으로 된 파일에 붙이면 줄이 어긋나므로 쓰지 않고 멈춘다."""
    scores_path = tmp_path / "scores.csv"
    old_lines = "시각,질문 수,검색_실제흐름\n2026-11-13,50,0.776\n"
    scores_path.write_text(old_lines, encoding="utf-8-sig")
    monkeypatch.setattr(run_evaluation, "SCORES_PATH", scores_path)

    with pytest.raises(ValueError):
        run_evaluation.save_score_line(LINE)
    assert scores_path.read_text(encoding="utf-8-sig").count("\n") == 2
