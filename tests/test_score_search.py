"""검색 적중률 채점(evaluation/score_search.py) 테스트.

가짜 질문 몇 개 + 지어낸 조각 10개 + 가짜 임베딩으로 채점 규칙과 함수 모양만 확인한다.
OpenAI를 부르지 않는다 (비용 0원). 가짜 임베딩은 뜻을 모르지만 "같은 글자 → 같은 숫자"라서,
조각 본문을 그대로 질문으로 넣으면 그 조각이 1등이 된다.
"""

from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from bidmate.retrieval.vector_store import get_vector_store, save_chunks
from bidmate.schemas import ProjectInfo
from evaluation.score_search import check_search_hit, load_questions, score, split_names
from tests.fake_chunks import ACADEMIC, ELEARNING, HOMEPAGE, make_fake_chunks

FAKE_EMBEDDING_SIZE = 16


def make_projects() -> list[ProjectInfo]:
    """지어낸 사업 3개의 목록표 (예산·마감일은 채점에 안 쓰여서 비운다)."""
    return [
        ProjectInfo(project_id, agency_name, project_name, None, None)
        for project_id, agency_name, project_name in (ELEARNING, ACADEMIC, HOMEPAGE)
    ]


def make_saved_store(folder: Path) -> Chroma:
    """지어낸 조각 10개를 가짜 임베딩으로 저장한 테스트용 저장소."""
    vector_store = get_vector_store(DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE), folder)
    save_chunks(make_fake_chunks(), vector_store)
    return vector_store


def make_question(question_id: str, question: str, expected_project_names: str) -> dict:
    """평가셋 한 줄과 같은 모양의 가짜 질문."""
    return {
        "question_id": question_id,
        "question_type": "한 문서",
        "question": question,
        "previous_question": "",
        "expected_answer": "",
        "expected_project_names": expected_project_names,
    }


def make_found(*project_names: str) -> list[tuple[Document, float]]:
    """search_chunks 결과 모양: 사업명만 다른 조각 목록."""
    return [
        (Document(page_content="내용", metadata={"project_name": name}), 0.1)
        for name in project_names
    ]


def test_split_names() -> None:
    """| 로 나누고 앞뒤 공백을 지우며, 빈 칸은 빈 목록이다."""
    assert split_names("사업A | 사업B") == ["사업A", "사업B"]
    assert split_names("") == []
    assert split_names(None) == []


def test_check_search_hit_rules() -> None:
    """한 문서는 하나라도 있으면 맞고, 여러 문서는 정답 문서마다 조각이 있어야 맞다."""
    assert check_search_hit(make_found("A", "B"), ["A"]) is True
    assert check_search_hit(make_found("B", "C"), ["A"]) is False
    assert check_search_hit(make_found("A", "B"), ["A", "B"]) is True
    assert check_search_hit(make_found("A", "A"), ["A", "B"]) is False
    assert check_search_hit(make_found("A"), []) is False


def test_score_returns_team_contract(tmp_path: Path) -> None:
    """score()는 팀 약속 칸(score, counted, wrong_question_ids)을 돌려주고, 없는 내용은 뺀다."""
    target_chunk = make_fake_chunks()[5]  # 라마대학교 조각
    questions = [
        make_question("Q001", target_chunk.page_content, ACADEMIC[2]),  # 맞음 (같은 글자 → 1등)
        make_question("Q002", "없는 내용 질문", ""),  # 계산에서 뺀다
    ]

    result = score(questions, make_saved_store(tmp_path), make_projects(), k=1)

    assert result["score"] == 1.0
    assert result["counted"] == 1
    assert result["wrong_question_ids"] == []
    assert result["average_search_seconds"] >= 0
    assert [row["question_id"] for row in result["rows"]] == ["Q001"]


def test_score_records_wrong_question(tmp_path: Path) -> None:
    """정답 문서 조각을 못 찾아오면 틀림으로 세고 번호를 남긴다."""
    elearning_chunk = make_fake_chunks()[1]  # 가나다공단 조각을 질문으로 → 가나다공단이 1등
    questions = [make_question("Q003", elearning_chunk.page_content, HOMEPAGE[2])]

    result = score(questions, make_saved_store(tmp_path), make_projects(), k=1)

    assert result["score"] == 0.0
    assert result["wrong_question_ids"] == ["Q003"]


def test_score_with_expected_projects_searches_only_those(tmp_path: Path) -> None:
    """검색만 잴 때는 정답 문서 안에서만 찾아서, 여러 문서 질문도 두 문서 조각이 다 나온다."""
    questions = [make_question("Q016", "요구사항 개수는?", f"{ACADEMIC[2]}|{HOMEPAGE[2]}")]

    result = score(
        questions, make_saved_store(tmp_path), make_projects(), use_expected_projects=True, k=5
    )

    assert result["score"] == 1.0
    assert result["rows"][0]["searched_project_ids"] == f"{ACADEMIC[0]}|{HOMEPAGE[0]}"


def test_score_reports_unknown_project_names(tmp_path: Path) -> None:
    """목록표에 없는 정답 사업명(오타)은 따로 알려 준다."""
    questions = [make_question("Q004", "예산은?", "통합 이러닝 플랫폼 구축사업")]  # 오타

    result = score(questions, make_saved_store(tmp_path), make_projects(), k=5)

    assert result["unknown_project_names"] == ["통합 이러닝 플랫폼 구축사업"]


def test_score_with_no_counted_questions(tmp_path: Path) -> None:
    """계산 대상이 0개면 0으로 나누지 않고 0점을 돌려준다."""
    result = score([make_question("Q005", "없는 내용", "")], make_saved_store(tmp_path), [])

    assert result["score"] == 0.0
    assert result["counted"] == 0


def test_load_questions_reads_csv(tmp_path: Path) -> None:
    """평가셋 csv(BOM 포함)를 칸 이름 그대로 읽는다."""
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "question_id,question_type,question,previous_question,expected_answer,"
        "expected_project_names\nQ001,한 문서,예산은?,,1억,사업A|사업B\n",
        encoding="utf-8-sig",
    )

    questions = load_questions(csv_path)

    assert questions[0]["question_id"] == "Q001"
    assert questions[0]["expected_project_names"] == "사업A|사업B"


@pytest.mark.parametrize("bad_k", [0])
def test_score_rejects_bad_k(tmp_path: Path, bad_k: int) -> None:
    """k가 0이면 search_chunks가 오류를 내서 잘못된 설정을 바로 알 수 있다."""
    questions = [make_question("Q006", "예산은?", ELEARNING[2])]

    with pytest.raises(ValueError):
        score(questions, make_saved_store(tmp_path), make_projects(), k=bad_k)
