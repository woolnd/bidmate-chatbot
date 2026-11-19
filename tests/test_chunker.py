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


TABLE_HEADER = "| 구분 | 수량 | 금액 |\n| --- | --- | --- |"


def _long_table(table_header: str) -> str:
    """본문 한 줄 + 표 헤더 + 데이터 행 100줄. 1000자를 넘어 여러 조각으로 잘린다."""
    rows = "\n".join(f"| 장비{row_number} | {row_number}대 | 1억 |" for row_number in range(100))
    return f"사업 개요\n{table_header}\n{rows}"


def test_split_document_repeats_table_header_in_cut_table() -> None:
    """긴 표가 잘리면 두 번째 조각부터 머리말 바로 아래에 표 헤더가 붙는다."""
    chunks = split_document(LoadedDocument(PROJECT.project_id, _long_table(TABLE_HEADER)), PROJECT)

    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert chunk.page_content.startswith(HEADER + TABLE_HEADER + "\n| 장비")
    # 첫 조각은 헤더가 원래 들어 있으므로 한 번만 나온다
    assert chunks[0].page_content.count(TABLE_HEADER) == 1


def test_split_document_short_table_keeps_one_header() -> None:
    """한 조각에 들어가는 짧은 표는 헤더를 다시 붙이지 않는다."""
    text = f"사업 개요\n{TABLE_HEADER}\n| 서버 | 2대 | 1억 |\n| 회선 | 10개 | 2천만 |"
    chunks = split_document(LoadedDocument(PROJECT.project_id, text), PROJECT)

    assert len(chunks) == 1
    assert chunks[0].page_content == HEADER + text


def test_split_document_skips_empty_table_header() -> None:
    """칸 이름이 없는 빈 헤더(|  |  |)는 잘린 조각에 붙이지 않는다."""
    empty_header = "|  |  |  |\n| --- | --- | --- |"
    chunks = split_document(LoadedDocument(PROJECT.project_id, _long_table(empty_header)), PROJECT)

    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert chunk.page_content.startswith(HEADER + "| 장비")


def test_split_document_does_not_use_header_of_earlier_table() -> None:
    """표가 끝나고 일반 글 뒤에 나온 다른 표에는 앞 표의 헤더를 붙이지 않는다."""
    first_table = f"{TABLE_HEADER}\n| 서버 | 2대 | 1억 |"
    # 헤더 없는 두 번째 표 (구분선 없음)
    second_rows = "\n".join(f"| 항목{row_number} | 값 |" for row_number in range(100))
    text = f"{first_table}\n일반 본문 문장입니다.\n{second_rows}"
    chunks = split_document(LoadedDocument(PROJECT.project_id, text), PROJECT)

    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert TABLE_HEADER not in chunk.page_content


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
