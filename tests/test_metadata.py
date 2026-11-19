"""목록표 읽기 테스트. 실제 data_list.csv는 커밋하지 않으므로 가짜 csv를 만들어 쓴다."""

from pathlib import Path

from bidmate.data.metadata import load_project_list
from bidmate.schemas import ProjectInfo

# 실제 목록표 칸 중 load_project_list가 쓰는 칸만 넣었다
FAKE_CSV = (
    "사업명,사업 금액,발주 기관,입찰 참여 마감일,파일명\n"
    " 가 사업 ,130000000.0,가 기관,2024-10-15 17:00:00,가.hwp\n"
    "나 사업,0.0,나 기관,,나.pdf\n"
)


def test_load_project_list(tmp_path: Path) -> None:
    """금액·날짜 모양을 맞추고, 금액 0과 빈 마감일은 None으로 바꾸는지 확인한다."""
    csv_path = tmp_path / "data_list.csv"
    csv_path.write_text(FAKE_CSV, encoding="utf-8-sig")

    projects = load_project_list(csv_path)

    assert projects == [
        ProjectInfo("가.hwp", "가 기관", "가 사업", 130000000, "2024-10-15"),
        ProjectInfo("나.pdf", "나 기관", "나 사업", None, None),
    ]
