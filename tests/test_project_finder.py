import pytest

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo


@pytest.fixture
def projects() -> list[ProjectInfo]:
    return [
        ProjectInfo(
            project_id="pension-elearning.hwp",
            agency_name="국민연금공단",
            project_name="이러닝시스템 운영 지원",
            budget=100000000,
            deadline="2024-05-31",
        ),
        ProjectInfo(
            project_id="rail-reservation.hwp",
            agency_name="한국철도공사 (용역)",
            project_name="예약발매시스템 개량 컨설팅",
            budget=200000000,
            deadline="2024-06-30",
        ),
        ProjectInfo(
            project_id="rail-mobile.hwp",
            agency_name="한국철도공사 (용역)",
            project_name="모바일오피스 시스템 고도화(1차)",
            budget=300000000,
            deadline="2024-07-31",
        ),
        ProjectInfo(
            project_id="rail-operation.hwp",
            agency_name="한국철도공사 (용역)",
            project_name="[재공고][긴급]운행정보기록 자동분석시스템 개선",
            budget=400000000,
            deadline="2024-08-31",
        ),
        ProjectInfo(
            project_id="korea-univ-portal.pdf",
            agency_name="고려대학교",
            project_name="포털·학사 정보시스템 구축 ",
            budget=500000000,
            deadline="2024-09-30",
        ),
        ProjectInfo(
            project_id="gist-academic.hwp",
            agency_name="광주과학기술원",
            project_name="학사시스템 개선",
            budget=600000000,
            deadline="2024-10-31",
        ),
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "국민연금공단 이러닝 사업 기간은?",
            ["pension-elearning.hwp"],
        ),
        (
            "국민연금공단 이러닝 사업 예산은?",
            ["pension-elearning.hwp"],
        ),
        (
            "사업 기간은?",
            [],
        ),
        (
            "예약발매시스템 개량 사업비는?",
            ["rail-reservation.hwp"],
        ),
        (
            "모바일오피스 고도화 사업 예산은?",
            ["rail-mobile.hwp"],
        ),
        (
            "운행정보기록 자동분석시스템 계약 방식은?",
            ["rail-operation.hwp"],
        ),
        (
            "고려대 포털 학사 시스템 예산과 마감일은?",
            ["korea-univ-portal.pdf"],
        ),
        (
            "철도 예약발매 시스템 개량 사업",
            [
                "rail-reservation.hwp",
            ],
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


def test_railway_official_name_matches_classified_agency(projects):
    result = find_project_ids(
        "한국철도공사 입찰 일정은?",
        projects,
    )

    assert result == [
        "rail-reservation.hwp",
        "rail-mobile.hwp",
        "rail-operation.hwp",
    ]


def test_agency_only_question_returns_all_agency_projects(projects):
    result = find_project_ids(
        "국민연금공단 사업 마감일은?",
        projects,
    )

    assert result == [
        "pension-elearning.hwp",
    ]


def test_unknown_agency_does_not_return_unrelated_projects(projects):
    result = find_project_ids(
        "삼성전자 스마트폰 개발 사업 예산은?",
        projects,
    )

    assert result == []


@pytest.mark.parametrize(
    "question",
    [
        "국민연금공단 스마트폰 개발 사업 예산은?",
        "국민연금공단 ERP 구축 마감일은?",
    ],
)
def test_known_agency_with_unmatched_project_terms_returns_no_results(
    projects: list[ProjectInfo],
    question: str,
) -> None:
    assert find_project_ids(question, projects) == []


def test_agency_wide_queries_and_unknown_projects():
    from bidmate.retrieval.project_finder import find_project_ids
    from bidmate.schemas import ProjectInfo

    projects = [
        ProjectInfo("water-a", "한국수자원공사", "정보시스템 운영", None, None),
        ProjectInfo("water-b", "한국수자원공사", "데이터 플랫폼 구축", None, None),
        ProjectInfo("pension-a", "국민연금공단", "이러닝시스템 운영", None, None),
    ]

    for question in (
        "한국수자원공사 사업 알려줘",
        "한국수자원공사 요구사항",
        "한국수자원공사 사업 목록",
    ):
        assert find_project_ids(question, projects) == [
            "water-a",
            "water-b",
        ]

    for question in (
        "국민연금공단 스마트폰 개발",
        "국민연금공단 ERP 구축",
    ):
        assert find_project_ids(question, projects) == []
