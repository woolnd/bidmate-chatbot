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
def test_known_agency_with_unmatched_project_terms_returns_agency_pool(
    projects: list[ProjectInfo],
    question: str,
) -> None:
    expected = [project.project_id for project in projects if project.agency_name == "국민연금공단"]
    assert find_project_ids(question, projects) == expected


def test_agency_wide_queries_and_unmatched_projects_return_agency_pool():
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
        assert find_project_ids(question, projects) == ["pension-a"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 사업 마감일은?", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 사업 알려줘", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 요구사항", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 존재하지않는사업 예산", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 문서관리 예산", ["문서[2].hwp"]),
        ("가나다기관연구소 요구사항", ["연구소.pdf"]),
        ("없는발주처 존재하지않는사업", []),
        ("", []),
    ],
)
def test_agency_pool_policy_keeps_scope_ids_and_specific_matches(question, expected):
    projects = [
        ProjectInfo("공동 (1).pdf", "가나다기관", "공동플랫폼 운영", None, None),
        ProjectInfo("문서[2].hwp", "가나다기관", "문서관리시스템 구축", None, None),
        ProjectInfo("연구소.pdf", "가나다기관연구소", "공동플랫폼 운영", None, None),
    ]
    result = find_project_ids(question, projects + [projects[0]])
    assert result == expected
    assert len(result) == len(set(result))


@pytest.fixture
def scoped_projects():
    return [
        ProjectInfo("records", "가나다기관", "문서관리시스템 구축", None, None),
        ProjectInfo("security", "가나다기관", "보안관제 운영", None, None),
        ProjectInfo("external", "라마바기관", "행정정보 구축", None, None),
    ]


@pytest.mark.parametrize("suffix", ["요구사항", "요약해줘", "제출서류", "담당자", "운영은?"])
def test_exact_title_does_not_expand_to_agency_pool(scoped_projects, suffix):
    question = f"가나다기관 문서관리시스템 구축 {suffix}"
    assert find_project_ids(question, scoped_projects) == ["records"]


def test_exact_title_with_unknown_second_project_keeps_proven_match(scoped_projects):
    question = "가나다기관 문서관리시스템 구축과 존재하지않는사업 예산"
    assert find_project_ids(question, scoped_projects) == ["records"]


def test_exact_title_still_includes_a_second_identified_project(scoped_projects):
    question = "가나다기관 문서관리시스템 구축과 보안관제 예산"
    assert set(find_project_ids(question, scoped_projects)) == {"records", "security"}


def test_excluding_unknown_project_does_not_remove_agency_pool(scoped_projects):
    question = "가나다기관 존재하지않는사업 말고 보안관제"
    assert find_project_ids(question, scoped_projects) == ["security"]


@pytest.mark.parametrize("tail", ["나머지", "다른 사업", "다른사업", "전체"])
def test_exclusion_preserves_remaining_projects_in_agency(scoped_projects, tail):
    question = f"가나다기관 문서관리시스템 구축 말고 {tail}"
    assert find_project_ids(question, scoped_projects) == ["security"]


def test_excluding_an_agency_still_allows_another_agency(scoped_projects):
    question = "가나다기관 말고 라마바기관 사업"
    assert find_project_ids(question, scoped_projects) == ["external"]


def test_exclusion_keeps_agency_context_with_colloquial_tail(scoped_projects):
    question = "가나다기관 문서관리시스템 구축 말고 나머지 요구사항 요약해줘"
    assert find_project_ids(question, scoped_projects) == ["security"]


def test_exclusion_keeps_agency_for_exact_title_and_second_short_project(scoped_projects):
    projects = scoped_projects + [
        ProjectInfo("dashboard", "가나다기관", "대시보드 개선", None, None)
    ]
    question = "가나다기관 문서관리시스템 구축 말고 보안관제 운영과 대시보드 예산"
    assert find_project_ids(question, projects) == ["security", "dashboard"]
