import pytest

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo


@pytest.fixture
def projects() -> list[ProjectInfo]:
    return [
        ProjectInfo(
            project_id="국민연금공단_2024년 이러닝시스템 운영 용역.hwp",
            agency_name="국민연금공단",
            project_name="2024년 이러닝시스템 운영 용역",
            budget=100000000,
            deadline="2024-05-31",
        ),
        ProjectInfo(
            project_id="한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
            agency_name="한국철도공사 (용역)",
            project_name="예약발매시스템 개량 ISMP 용역",
            budget=200000000,
            deadline="2024-06-30",
        ),
        ProjectInfo(
            project_id="한국철도공사 (용역)_모바일오피스 시스템 고도화 용역(총체 및 1차).hwp",
            agency_name="한국철도공사 (용역)",
            project_name="모바일오피스 시스템 고도화 용역(총체 및 1차)",
            budget=300000000,
            deadline="2024-07-31",
        ),
        ProjectInfo(
            project_id="한국철도공사 (용역)_[재공고][긴급][협상형]운행정보기록 자동분석시스.hwp",
            agency_name="한국철도공사 (용역)",
            project_name="[재공고][긴급][협상형]운행정보기록 자동분석시스템 개량",
            budget=400000000,
            deadline="2024-08-31",
        ),
        ProjectInfo(
            project_id="고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf",
            agency_name="고려대학교",
            project_name="차세대 포털·학사 정보시스템 구축사업 ",
            budget=500000000,
            deadline="2024-09-30",
        ),
        ProjectInfo(
            project_id="광주과학기술원_학사시스템 기능개선 사업.hwp",
            agency_name="광주과학기술원",
            project_name="학사시스템 기능개선 사업",
            budget=600000000,
            deadline="2024-10-31",
        ),
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "국민연금공단 이러닝 사업 기간은?",
            ["국민연금공단_2024년 이러닝시스템 운영 용역.hwp"],
        ),
        (
            "국민연금공단 이러닝 사업 예산은?",
            ["국민연금공단_2024년 이러닝시스템 운영 용역.hwp"],
        ),
        (
            "사업 기간은?",
            [],
        ),
        (
            "예약발매시스템 개량 사업비는?",
            ["한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp"],
        ),
        (
            "모바일오피스 고도화 사업 예산은?",
            ["한국철도공사 (용역)_모바일오피스 시스템 고도화 용역(총체 및 1차).hwp"],
        ),
        (
            "운행정보기록 자동분석시스템 계약 방식은?",
            ["한국철도공사 (용역)_[재공고][긴급][협상형]운행정보기록 자동분석시스.hwp"],
        ),
        (
            "고려대 포털 학사 시스템 예산과 마감일은?",
            ["고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"],
        ),
        (
            "철도 예약발매 시스템 개량 사업",
            [],
        ),
        (
            "삼성전자 스마트폰 개발 사업 예산은?",
            [],
        ),
    ],
)
def test_find_project_ids(
    projects: list[ProjectInfo],
    question: str,
    expected: list[str],
) -> None:
    assert find_project_ids(question, projects) == expected
