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


# ---------- 두 사업 나눠 찾기 ----------


def test_two_projects_joined_by_wa_are_found_separately(projects):
    """ "A와 B 사업의 기간은 각각" 꼴이면 두 사업을 모두 고른다."""
    question = "예약발매시스템과 운행정보기록 자동분석시스템 사업의 기간은 각각 어떻게 되는가?"

    assert find_project_ids(question, projects) == [
        "rail-reservation.hwp",
        "rail-operation.hwp",
    ]


def test_split_falls_back_when_one_part_is_not_a_project(projects):
    """나눈 한쪽이 사업이 아니면 나눠 찾기를 쓰지 않고 원래 방식으로 찾는다."""
    question = "모바일오피스 예산과 기간은?"

    assert find_project_ids(question, projects) == ["rail-mobile.hwp"]


def test_content_after_project_name_is_cut_off(projects):
    """ "OO 사업에서 …" 꼴이면 사업명 뒤 질문 내용은 떼고 찾는다."""
    question = "운행정보기록 자동분석시스템 사업에서 분석하는 기록 항목은 총 몇 개인가?"

    assert find_project_ids(question, projects) == ["rail-operation.hwp"]


# ---------- 혜린님 코드에서 가져온 규칙 (사업명 고유 단어로 다시 찾기) ----------


@pytest.mark.parametrize("agency", ["", "한결연구원 "])
def test_title_word_followed_by_question_field_is_recovered(agency):
    """사업 하나에만 있는 사업명 단어 뒤에 "계약" 같은 질문 항목이 오면 그 사업을 고른다."""
    projects = [
        ProjectInfo("coast", "한결연구원", "연안관측망 자동기록 운영", None, None),
        ProjectInfo("inland", "한결연구원", "내륙관측망 자동기록 운영", None, None),
    ]
    question = f"{agency}연안관측망 계약 참여 자격 추가 증빙 인정 범위 원본 사본 제출"

    assert find_project_ids(question, projects) == ["coast"]


@pytest.mark.parametrize("agency", ["", "한결연구원 "])
def test_same_opening_word_selects_project_known_limit(agency):
    """알려진 한계: 목록표에 없는 단어는 버리므로, 사업명 단어가 하나 맞으면 그 사업을 고른다.

    목록에 없는 사업을 물은 것인지 그 사업 안의 내용을 물은 것인지 목록표만으로는 구분할 수 없다.
    잘못 골라도 그 문서에 내용이 없어 4단계가 "찾은 문서에 없다"고 답한다.
    """
    projects = [ProjectInfo("coast", "한결연구원", "연안관측망 자동기록 운영", None, None)]
    question = f"{agency}연안관측망 해저 소음 교정 드론 복원 사업 예산"

    assert find_project_ids(question, projects) == ["coast"]


@pytest.mark.parametrize(
    "question",
    [
        "국민연금공단 사업 언제까지야?",
        "국민연금공단 사업 예산이 얼마야?",
        "국민연금공단 사업 마감일이 언제예요?",
        "국민연금공단 사업 예산좀",
        "국민연금공단 사업 얼마임?",
    ],
)
def test_question_with_ending_is_agency_wide(question):
    """ "얼마야", "언제예요"처럼 말끝이 붙은 질문 단어도 걸러서 기관 전체 질문으로 본다."""
    projects = [
        ProjectInfo("a", "국민연금공단", "이러닝시스템 운영 지원", None, None),
        ProjectInfo("b", "국민연금공단", "사회보험료 지원 정보시스템 보완", None, None),
    ]
    assert find_project_ids(question, projects) == ["a", "b"]


def test_particle_rang_is_removed(projects):
    """ "예산이랑"처럼 붙은 조사 "이랑"도 떼고 찾는다."""
    question = "운행정보기록 자동분석시스템 예산이랑 계약 방법"

    assert find_project_ids(question, projects) == ["rail-operation.hwp"]


def test_shared_word_does_not_pull_in_other_project_by_itself():
    """공통 단어가 하나 겹친다고, 질문 단어를 전부 덮는 사업 말고 다른 사업까지 같이 고르지 않는다.

    "클라우드 전환"을 물었는데 "클라우드 관제"까지 같이 골랐던 문제의 재현.
    """
    projects = [
        ProjectInfo("migration", "새봄공단", "클라우드 전환 타당성 검토", None, None),
        ProjectInfo("monitoring", "새봄공단", "클라우드 관제 통합 개선", None, None),
    ]
    assert find_project_ids("새봄공단 클라우드 전환 마감", projects) == ["migration"]
    assert find_project_ids("새봄공단 클라우드 관제 기간", projects) == ["monitoring"]
