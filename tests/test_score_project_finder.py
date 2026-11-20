import pytest

from bidmate.data.metadata import load_project_list
from bidmate.schemas import ProjectInfo
from evaluation.score_project_finder import score


@pytest.fixture
def projects() -> list[ProjectInfo]:
    return [
        ProjectInfo("a", "가나다기관", "문서관리시스템 구축", None, None),
        ProjectInfo("b", "가나다기관", "보안관제 운영", None, None),
    ]


def test_score_counts_exact_sets_and_skips_followups(projects):
    questions = [
        ("한 문서", "가나다기관 사업 목록", "문서관리시스템 구축"),
        ("여러 문서", "가나다기관 문서관리 예산", "문서관리시스템 구축|보안관제 운영"),
        (
            "여러 문서",
            "가나다기관 문서관리와 보안관제 각각 예산",
            "보안관제 운영|문서관리시스템 구축",
        ),
        ("없는 내용", "가나다기관 위성 발사 사업 예산", ""),
        ("후속 질문", "그 사업은?", ""),
    ]
    rows = [
        {
            "question_id": str(index),
            "question_type": kind,
            "question": question,
            "expected_project_names": names,
        }
        for index, (kind, question, names) in enumerate(questions)
    ]
    result = score(rows, projects)
    assert (result["correct"], result["counted"], result["skipped_followups"]) == (2, 4, 1)
    assert [row["exact"] for row in result["rows"]] == [False, False, True, True]
    assert result["rows"][0]["extra_ids"] == ["b"]
    assert result["rows"][1]["missing_ids"] == ["b"]


def test_score_rejects_ambiguous_expected_project_name(projects):
    projects.append(ProjectInfo("c", "라마바기관", "보안관제 운영", None, None))
    rows = [
        {
            "question_id": "Q1",
            "question_type": "한 문서",
            "question": "보안관제 예산",
            "expected_project_names": "보안관제 운영",
        }
    ]
    with pytest.raises(ValueError, match="Q1: 정답 사업명"):
        score(rows, projects)


def test_score_keeps_filename_with_significant_spaces(tmp_path):
    input_csv = tmp_path / "metadata.csv"
    input_csv.write_text(
        "파일명,발주 기관,사업명,사업 금액,입찰 참여 마감일\n"
        "  a.hwp ,가나다기관,문서관리시스템 구축,,\n",
        encoding="utf-8",
    )
    projects = load_project_list(input_csv)
    row = {
        "question_id": "Q1",
        "question_type": "한 문서",
        "question": "가나다기관 문서관리",
        "expected_project_names": "문서관리시스템 구축",
    }
    result = score([row], projects)
    assert result["rows"][0]["expected_ids"] == ["  a.hwp "]
    assert result["rows"][0]["predicted_ids"] == ["  a.hwp "]
