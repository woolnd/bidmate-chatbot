"""조각 자르기 테스트. 실제 문서·API 없이 가짜 ProjectInfo, LoadedDocument로 확인한다."""

from bidmate.config import CHUNK_SIZE
from bidmate.data.chunker import make_info_chunk, split_document
from bidmate.schemas import CHUNK_METADATA_KEYS, LoadedDocument, ProjectInfo

PROJECT = ProjectInfo("가나다공단_이러닝.hwp", "가나다공단", "이러닝 구축", 130000000, "2024-05-31")
HEADER = "[가나다공단 / 이러닝 구축]\n"


def test_split_document_cuts_long_text() -> None:
    """긴 본문은 여러 조각이 되고, 조각마다 머리말·이름표·번호가 붙는다."""
    text = "\n".join(f"{line_number}번째 줄 요구사항 내용입니다." for line_number in range(500))
    chunks = split_document(LoadedDocument(PROJECT.project_id, text), PROJECT)

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_id"] == "가나다공단_이러닝.hwp-0001"
    assert chunks[1].metadata["chunk_id"] == "가나다공단_이러닝.hwp-0002"
    for chunk in chunks:
        assert chunk.page_content.startswith(HEADER)
        assert len(chunk.page_content) <= len(HEADER) + CHUNK_SIZE
        assert set(chunk.metadata) == set(CHUNK_METADATA_KEYS)


def test_split_document_empty_text() -> None:
    """본문이 비어 있으면 조각이 없다."""
    assert split_document(LoadedDocument(PROJECT.project_id, ""), PROJECT) == []


def test_make_info_chunk() -> None:
    """기본 정보 조각에 예산·마감일이 들어가고 chunk_id는 -info로 끝난다."""
    chunk = make_info_chunk(PROJECT)
    assert chunk.metadata["chunk_id"] == "가나다공단_이러닝.hwp-info"
    assert chunk.page_content.startswith(HEADER)
    assert "130,000,000원" in chunk.page_content
    assert "2024-05-31" in chunk.page_content


def test_make_info_chunk_without_budget_and_deadline() -> None:
    """예산·마감일이 없으면 '정보 없음'으로 적는다."""
    project = ProjectInfo("a.hwp", "가나다공단", "이러닝 구축", None, None)
    assert make_info_chunk(project).page_content.count("정보 없음") == 2
