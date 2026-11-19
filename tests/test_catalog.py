"""catalog 길 목록표 만들기 테스트. 진짜 data_list.csv 대신 가짜 csv를 쓴다 (CI에는 파일이 없다).

예시는 지어낸 기관명으로 쓴다 (이 파일은 Git에 올라간다).
"""

import pytest

from bidmate.generation import catalog
from bidmate.schemas import CHUNK_METADATA_KEYS

FAKE_CSV = (
    "공고 번호,사업명,사업 금액,발주 기관,입찰 참여 마감일,사업 요약,텍스트\n"
    '1,작은 사업,100000000.0,가나다공단,2024-05-01 10:00:00,"- 목적: 작음\r\n- 내용: 하나",본문1\n'
    '2,예산 빈칸 사업,,라마대학교,,"- 목적: 빈칸",본문2\n'
    '3,큰 사업,2500000000.0,바사아청,2024-06-30 17:00:00,"- 목적: 큼",본문3\n'
    '4,0원 사업,0.0,자차카시,2024-07-01 09:00:00,"- 목적: 영",본문4\n'
)


@pytest.fixture
def catalog_lines(tmp_path, monkeypatch) -> list[str]:
    """가짜 csv로 목록표 조각을 만들어 줄 목록으로 돌려준다."""
    csv_path = tmp_path / "data_list.csv"
    csv_path.write_text(FAKE_CSV, encoding="utf-8-sig")
    monkeypatch.setattr(catalog, "PROJECT_LIST_PATH", csv_path)
    catalog.build_catalog_document.cache_clear()  # 기억해 둔 진짜 목록표를 지운다
    yield catalog.build_catalog_document().page_content.splitlines()
    catalog.build_catalog_document.cache_clear()


def test_sorted_by_budget_with_missing_values_last(catalog_lines):
    # 제목 줄 2개 뒤에 예산이 큰 순서. 빈칸·0원은 맨 뒤
    assert catalog_lines[0] == "공공기관 RFP 사업 목록 (4개, 예산이 큰 순서)"
    project_order = [line.split(" | ")[1] for line in catalog_lines[2:]]
    assert project_order[:2] == ["큰 사업", "작은 사업"]
    assert set(project_order[2:]) == {"예산 빈칸 사업", "0원 사업"}


def test_line_format_and_missing_values(catalog_lines):
    # 예산은 쉼표+원, 마감일은 날짜만, 요약 줄바꿈은 띄어쓰기로. 빈칸은 MISSING_VALUE_TEXT
    big = catalog_lines[2]
    assert big == "바사아청 | 큰 사업 | 2,500,000,000원 | 2024-06-30 | - 목적: 큼"
    small = catalog_lines[3]
    assert small.endswith("| - 목적: 작음 - 내용: 하나")
    blank = next(line for line in catalog_lines if "예산 빈칸 사업" in line)
    assert blank.count(catalog.MISSING_VALUE_TEXT) == 2  # 예산과 마감일
    zero = next(line for line in catalog_lines if "0원 사업" in line)
    assert catalog.MISSING_VALUE_TEXT in zero
    assert "본문" not in "\n".join(catalog_lines)  # 긴 '텍스트' 칸은 넣지 않는다


def test_document_has_chunk_metadata_keys(catalog_lines):
    document = catalog.build_catalog_document()
    assert tuple(sorted(document.metadata)) == tuple(sorted(CHUNK_METADATA_KEYS))
    assert document.metadata["project_name"] == catalog.CATALOG_PROJECT_NAME
