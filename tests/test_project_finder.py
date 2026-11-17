import pytest

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo


@pytest.mark.parametrize("agency", ["", "한결연구원 "])
def test_title_evidence_survives_content_details(agency):
    projects = [
        ProjectInfo("coast", "한결연구원", "연안관측망 자동기록 운영", None, None),
        ProjectInfo("inland", "한결연구원", "내륙관측망 자동기록 운영", None, None),
    ]
    question = f"{agency}연안관측망 계약 참여 자격 추가 증빙 인정 범위 원본 사본 제출"
    assert find_project_ids(question, projects) == ["coast"]


@pytest.mark.parametrize("agency", ["", "한결연구원 "])
def test_same_opening_word_does_not_select_unlisted_business(agency):
    projects = [ProjectInfo("coast", "한결연구원", "연안관측망 자동기록 운영", None, None)]
    question = f"{agency}연안관측망 해저 소음 교정 드론 복원 사업 예산"
    assert find_project_ids(question, projects) == []


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
def test_known_agency_with_unmatched_project_terms_returns_empty(
    projects: list[ProjectInfo],
    question: str,
) -> None:
    assert find_project_ids(question, projects) == []


def test_agency_wide_queries_return_agency_pool():
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
        "한국수자원공사 사업 좀 알려줘",
        "한국수자원공사 사업 알려 주세요",
    ):
        assert find_project_ids(question, projects) == [
            "water-a",
            "water-b",
        ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 사업 마감일은?", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 사업 알려줘", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 요구사항", ["공동 (1).pdf", "문서[2].hwp"]),
        ("가나다기관 존재하지않는사업 예산", []),
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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("코레일 예약발매 예산", ["rail-reservation.hwp"]),
        ("코레일의 예약발매 예산", ["rail-reservation.hwp"]),
        ("축평원 사업 알려줘", ["livestock"]),
        ("가짜코레일 사업 알려줘", []),
    ],
)
def test_agency_aliases(projects, question, expected):
    projects = projects + [
        ProjectInfo("livestock", "축산물품질평가원", "통합포털 운영", None, None),
    ]
    assert find_project_ids(question, projects) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("문서관리랑 보안관제 각각 예산 알려줘", {"records", "security"}),
        ("가나다기관 문서관리랑 라마바기관 행정정보 각각 예산 알려줘", {"records", "external"}),
        ("가나다기관 문서관리와 보안관제의 예산은 각각?", {"records", "security"}),
        ("가나다기관 문서관리와 존재하지않는사업의 예산은 각각?", {"records"}),
        ("가나다기관 기록물과 개인정보 관리 예산", {"privacy"}),
    ],
)
def test_coordinated_project_queries(scoped_projects, question, expected):
    projects = scoped_projects + [
        ProjectInfo("privacy", "가나다기관", "기록물과 개인정보 관리", None, None),
        ProjectInfo("other-admin", "가나다기관", "행정정보 운영", None, None),
    ]
    assert set(find_project_ids(question, projects)) == expected


def test_details_after_project_mention_do_not_dilute_match(scoped_projects):
    question = (
        "문서관리시스템 사업에서 외부 연계 대상과 상주 인력의 최소 근무 기간은 몇 개월인가요?"
    )
    assert find_project_ids(question, scoped_projects) == ["records"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 문서관리와 행정정보 각각 예산", {"records", "external"}),
        ("가나다기관 보안관제와 라마바기관 문서관리 각각 예산", {"security"}),
    ],
)
def test_separate_project_without_repeated_agency(scoped_projects, question, expected):
    assert set(find_project_ids(question, scoped_projects)) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("문서관리시스템 사업에 참여하는 업체의 기술 인력은 몇 명인가요?", ["records"]),
        ("보안관제 사용 시 외부 계정 등록 절차와 승인 권한이 어떻게 되나요?", ["security"]),
        ("위성발사 사업에 참여하는 업체의 기술 인력은 몇 명인가요?", []),
        ("보안관제 사업 예산과 마감일은?", ["security"]),
    ],
)
def test_business_name_before_content_question(scoped_projects, question, expected):
    assert find_project_ids(question, scoped_projects) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 보안관제 사업에 문서관리시스템 구축이 필요한가요?", ["security"]),
        ("가나다기관 위성발사 사업에 문서관리시스템 구축이 필요한가요?", []),
        ("가나다기관 사업에 문서관리시스템 구축이 포함되나요?", ["records"]),
        ("문서관리시스템 구축과 보안관제 사업에 참여하는 업체는?", ["records", "security"]),
        ("문서관리시스템 구축의 예산과 보안관제 운영의 마감일은 각각?", ["records", "security"]),
    ],
)
def test_content_reference_does_not_replace_subject(scoped_projects, question, expected):
    assert find_project_ids(question, scoped_projects) == expected


@pytest.mark.parametrize("connector", ["과", "이랑", "하고"])
def test_multiple_agencies_without_specific_project(connector, projects):
    other = ProjectInfo("gas", "한국가스공사", "가스망 안전관리 개선", None, None)
    question = f"코레일{connector} 가스공사 사업 목록"
    assert set(find_project_ids(question, projects + [other])) == {
        "rail-reservation.hwp",
        "rail-mobile.hwp",
        "rail-operation.hwp",
        "gas",
    }


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "가나다기관 보안관제 사업에 문서관리시스템 구축과 계정관리 개선이 필요한가요?",
            ["security"],
        ),
        ("위성발사 사업에 문서관리시스템 구축이 필요한가요?", []),
        (
            "보안관제 운영의 요구사항으로 문서관리시스템 구축이 필요한가요?",
            ["security"],
        ),
        ("보안관제/행정정보 각각 예산", ["security", "external"]),
        ("보안관제·행정정보 예산 비교", ["security", "external"]),
        ("보안관제 & 행정정보 각각 예산", ["security", "external"]),
        ("문서관리시스템 구축의 예산과 보안관제 운영의 마감일은?", ["records", "security"]),
        (
            "가나다기관 보안관제 사업에 문서관리시스템 구축과 "
            "계정관리 개선의 요구사항을 각각 반영하나요?",
            ["security"],
        ),
        (
            "가나다기관 위성발사 사업에 문서관리시스템 구축과 "
            "계정관리 개선의 요구사항을 각각 반영하나요?",
            [],
        ),
    ],
)
def test_distinguishes_project_comparison_from_content_list(scoped_projects, question, expected):
    account = ProjectInfo("account", "가나다기관", "계정관리 개선", None, None)
    assert find_project_ids(question, scoped_projects + [account]) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 2024년 문서관리 예산", ["doc24"]),
        ("가나다기관 문서관리 2025 마감일", ["doc25"]),
        ("가나다기관 2026년 문서관리 예산", []),
        ("가나다기관 문서관리 2025년도 예산", ["doc25"]),
        ("가나다기관 문서관리 2025년의 예산", ["doc25"]),
        ("가나다기관 통합백업센터 2차 예산", ["backup2"]),
        ("가나다기관 통합백업센터 3차 예산", []),
        ("가나다기관 통합백업센터 2 차의 예산", ["backup2"]),
        ("가나다기관 통합백업센터2차 예산", ["backup2"]),
        ("가나다기관 2024 문서관리와 2차 통합백업센터 각각 예산", ["doc24", "backup2"]),
        ("가나다기관 2024 문서관리와 2025 문서관리 각각 예산", ["doc24", "doc25"]),
        ("가나다기관 2024년과 2025년 문서관리 예산", ["doc24", "doc25"]),
        ("가나다기관 1차랑 2차 통합백업센터 예산", ["backup1", "backup2"]),
        ("가나다기관 2024년 문서관리 사업의 2025년 운영 예산은?", ["doc24"]),
    ],
)
def test_explicit_year_and_round_scope(question, expected):
    projects = [
        ProjectInfo("doc24", "가나다기관", "2024년 문서관리시스템 운영", None, None),
        ProjectInfo("doc25", "가나다기관", "2025년 문서관리시스템 운영", None, None),
        ProjectInfo("doc23", "가나다기관", "2023년 문서관리시스템 운영", None, None),
        ProjectInfo("backup1", "가나다기관", "통합백업센터 구축 1차", None, None),
        ProjectInfo("backup2", "가나다기관", "통합백업센터 구축 2차", None, None),
    ]
    assert find_project_ids(question, projects) == expected


def test_unique_full_term_match_wins_over_shared_term():
    projects = [
        ProjectInfo("migration", "가나다기관", "클라우드 전환 타당성 검토", None, None),
        ProjectInfo("monitoring", "가나다기관", "클라우드 관제 통합 개선", None, None),
    ]
    assert find_project_ids("가나다기관 클라우드 전환 마감", projects) == ["migration"]
    assert set(
        find_project_ids("가나다기관 클라우드 전환과 클라우드 관제 각각 마감", projects)
    ) == {"migration", "monitoring"}


def test_multiple_exclusions_keep_the_original_agency(scoped_projects):
    projects = [
        *scoped_projects,
        ProjectInfo("portal", "가나다기관", "통합 상담창구 개선", None, None),
    ]
    result = find_project_ids("가나다기관 문서관리 말고 보안관제도 빼고 나머지", projects)
    assert result == ["portal"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 2026년 보조금 정산 플랫폼 예산", []),
        ("가나다기관 2025년 보조금 정산 플랫폼 예산", ["edition25"]),
        ("가나다기관 2024년과 2025년 보조금 정산 플랫폼 예산", ["edition24", "edition25"]),
    ],
)
def test_unversioned_title_does_not_replace_an_edition(question, expected):
    projects = [
        ProjectInfo("generic", "가나다기관", "보조금 정산 플랫폼 개발", None, None),
        ProjectInfo("edition24", "가나다기관", "2024년 보조금 정산 플랫폼 개발", None, None),
        ProjectInfo("edition25", "가나다기관", "2025년 보조금 정산 플랫폼 개발", None, None),
    ]
    assert find_project_ids(question, projects) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 전자문서관리시스템 운영지원센터 사업 예산", []),
        ("가나다기관 통합전자문서관리시스템 운영 사업 예산", []),
        ("가나다기관 전자문서관리시스템 운영사업 예산", ["docs"]),
        ("가나다기관 전자문서관리시스템 운영의 예산은?", ["docs"]),
        ("가나다기관 [전자문서관리시스템운영] 예산", ["docs"]),
    ],
)
def test_complete_title_requires_legal_word_boundaries(question, expected):
    projects = [
        ProjectInfo("docs", "가나다기관", "전자문서관리시스템 운영", None, None),
    ]
    assert find_project_ids(question, projects) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("가나다기관 보안관제 사업의 부가세를 빼고 예산이 얼마인가요?", ["security"]),
        ("가나다기관 보안관제 VAT 빼고 예산 얼마야?", ["security"]),
        ("가나다기관 사업 부가세 빼고 예산 알려줘", ["records", "security"]),
        ("가나다기관 보안관제 사업에서 마감일 말고 계약 방식을 알려줘", ["security"]),
        ("가나다기관 문서관리 사업에서 현장 점검 빼고 제출할 문서는?", ["records"]),
        ("가나다기관 보안관제 사업은 빼고 문서관리 예산 알려줘", ["records"]),
    ],
)
def test_field_exclusion_keeps_the_requested_project(scoped_projects, question, expected):
    assert find_project_ids(question, scoped_projects) == expected
