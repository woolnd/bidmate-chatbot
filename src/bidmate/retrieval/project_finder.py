"""Search-B: 질문에 적합한 사업 문서를 선별한다.

기관명과 사업명을 분리해 비교하고,
전체 사업 목록에서 계산한 단어 중요도를 활용해 후보를 평가한다.

CSV 로딩이나 문서 검색은 담당하지 않으며,
선택한 문서의 원본 project_id만 반환한다.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bidmate.schemas import ProjectInfo


# 질문의 의도나 공고 형식에 가까워 사업 식별에 도움이 적은 단어
STOP_WORDS = {
    "사업",
    "사업비",
    "기간",
    "예산",
    "마감일",
    "계약",
    "방식",
    "시스템",
    "용역",
    "운영",
    "구축",
    "및",
    "등",
    "관련",
    "대한",
    "통한",
    "위한",
    "긴급",
    "재공고",
    "입찰",
    "입찰공고",
    "협상",
    "협상에",
    "의한",
    "제안서",
    "제출",
    "안내",
    "공고",
    "주",
    "사",
}

STOP_WORDS.update({"얼마", "마감", "언제까지", "돈", "개발", "개량", "기능개선"})

STOP_WORDS.update(
    {
        "일정",
        "현황",
        "목록",
        "조회",
        "확인",
        "알려",
        "언제",
        "어디",
        "관련",
        "찾아",
        "검색",
        "입찰",
    }
)


# 의미가 명확한 표현만 별칭으로 연결한다.
ALIASES = {
    "국민연금": "국민연금공단",
    "이러닝": "이러닝시스템",
    "고려대": "고려대학교",
    "포털": "차세대포털",
}

ACTION_WORDS = {"개량", "고도화", "개발", "구축", "운영", "기능개선"}
STOP_WORDS.difference_update(ACTION_WORDS)

# 현재 목록의 기관 표기를 확인한 약칭이다. 여러 기관을 뜻하면 모두 남긴다.
AGENCY_ALIASES = {
    "철도": ("한국철도공사", "국가철도공단"),
    "연금": ("국민연금공단",),
    "가스": ("한국가스공사",),
    "가스공사": ("한국가스공사",),
    "생기원": ("한국생산기술연구원",),
    "연구재단": ("한국연구재단",),
    "수자원공사": ("한국수자원공사",),
    "광주과기원": ("광주과학기술원",),
    "gist": ("광주과학기술원",),
    "서울교육청": ("서울특별시교육청",),
    "여성가족재단": ("서울특별시 여성가족재단",),
    "인천공항": ("인천공항운영서비스",),
    "인천공항운영서비스": ("인천공항운영서비스",),
    "울산": ("울산광역시",),
    "평택": ("경기도 평택시",),
    "인천": ("인천광역시", "인천광역시 동구"),
    "시립대": ("서울시립대학교",),
    "을지대": ("을지대학교",),
    "대전대": ("대전대학교",),
    "조선대": ("조선대학교",),
    "수출입은행": ("한국수출입은행",),
    "어촌어항공단": ("한국어촌어항공단",),
}

# 질문에 붙는 조사
PARTICLES = (
    "에서",
    "으로",
    "에게",
    "부터",
    "까지",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "도",
)


def _normalize(text: str) -> str:
    """유니코드 표현과 대소문자를 통일한다."""
    return unicodedata.normalize("NFKC", text).casefold()


def _tokenize(text: str) -> set[str]:
    """한글, 영문, 숫자 단위로 단어를 추출한다."""

    normalized = _normalize(text)

    # 법인 표기는 기관을 구분하는 검색 단어로 취급하지 않는다.
    normalized = re.sub(
        r"주식회사|유한회사|\(주\)|\(사\)|㈜|㈔",
        " ",
        normalized,
    )

    raw_tokens = re.findall(r"[가-힣a-z0-9]+", normalized)
    tokens: set[str] = set()

    for original_token in raw_tokens:
        token = original_token

        # 연도는 대부분의 사업에서 반복되므로 식별 단어에서 제외한다.
        if re.fullmatch(r"20\d{2}년?", token):
            continue

        if re.fullmatch(
            r"(?:얼마|언제|예산|마감|마감일|사업비|기간|일정|돈)"
            r"(?:은|는|이|가|을|를|까지)?(?:좀|야|임|지|냐|인가|인지|인가요|이야|이에요|예요|요)?",
            token,
        ):
            continue

        # 질문에 붙은 조사 제거
        for particle in PARTICLES:
            if token.endswith(particle) and len(token) > len(particle) + 1:
                token = token[: -len(particle)]
                break

        if not token or token in STOP_WORDS:
            continue

        token = ALIASES.get(token, token)
        tokens.add(token)

    return tokens


def _calculate_idf(
    projects: list[ProjectInfo],
) -> tuple[dict[str, float], int]:
    """사업명에 등장하는 단어의 역문서빈도(IDF)를 계산한다."""

    document_frequency: Counter[str] = Counter()

    for project in projects:
        # 한 사업 안에서 같은 단어가 반복되어도 한 번만 센다.
        document_frequency.update(_tokenize(project.project_name))

    total_documents = len(projects)

    idf = {
        token: math.log((total_documents + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }

    return idf, total_documents


def _weighted_coverage(
    query_tokens: set[str],
    matched_tokens: set[str],
    idf: dict[str, float],
) -> float:
    """질문 단어 중 일치한 단어의 중요도 가중 비율을 계산한다."""

    if not query_tokens:
        return 0.0

    total_weight = sum(idf.get(token, 1.0) for token in query_tokens)
    matched_weight = sum(idf.get(token, 1.0) for token in matched_tokens)

    if total_weight == 0:
        return 0.0

    return matched_weight / total_weight


def _match_project_tokens(
    query_tokens: set[str],
    project_tokens: set[str],
) -> set[str]:
    """사업명 단어의 완전 일치와 안전한 부분 일치를 확인한다.

    부분 일치는 길이가 3자 이상인 질문 단어에만 허용한다.
    예: 예약발매 -> 예약발매시스템
    """
    matched = set()

    for query_token in query_tokens:
        if query_token in project_tokens:
            matched.add(query_token)
            continue

        if re.fullmatch(r"[a-z]+", query_token):
            if any(
                re.match(r"[a-z]+", token) and re.match(r"[a-z]+", token).group() == query_token
                for token in project_tokens
            ):
                matched.add(query_token)
            continue

        if len(query_token) < 2:
            continue

        if any(query_token in project_token for project_token in project_tokens):
            matched.add(query_token)

    return matched


def _compact_agency_name(text: str) -> str:
    """기관명 비교용 정규화: 법인 표기와 분류용 접미사를 제거한다."""
    normalized = _normalize(text)
    normalized = re.sub(
        r"\s*\(\s*용역\s*\)\s*$|\s+용역\s*$",
        "",
        normalized,
    )
    normalized = re.sub(
        r"주식회사|유한회사|\(주\)|\(사\)|㈜|㈔",
        "",
        normalized,
    )
    return re.sub(r"[^가-힣a-z0-9]", "", normalized)


def _get_agency_matches(
    question: str,
    projects: list[ProjectInfo],
) -> list[str]:
    """질문에서 실제 데이터에 존재하는 기관만 안전하게 식별한다."""
    question_key = _compact_agency_name(question)

    agencies = sorted(
        {project.agency_name for project in projects if project.agency_name},
        key=lambda value: len(_compact_agency_name(value)),
        reverse=True,
    )

    matched = []
    occupied: list[tuple[int, int]] = []

    for agency in agencies:
        agency_key = _compact_agency_name(agency)
        if len(agency_key) < 3:
            continue
        positions = [
            (match.start(), match.end())
            for match in re.finditer(re.escape(agency_key), question_key)
        ]
        independent = [
            (start, end)
            for start, end in positions
            if not any(left <= start and end <= right for left, right in occupied)
        ]
        if independent:
            matched.append(agency)
            occupied.extend(independent)

    for alias, canonical in ALIASES.items():
        alias_key = _compact_agency_name(alias)
        canonical_key = _compact_agency_name(canonical)

        if not alias_key or alias_key not in question_key:
            continue

        targets = [agency for agency in agencies if _compact_agency_name(agency) == canonical_key]

        if len(targets) != 1:
            continue

        target = targets[0]

        embedded_in_other_agency = any(
            alias_key in _compact_agency_name(agency) and agency != target for agency in matched
        )

        if not embedded_in_other_agency and target not in matched:
            matched.append(target)

    agency_aliases = dict(AGENCY_ALIASES)
    for agency in agencies:
        key = _compact_agency_name(agency)
        if key.startswith("한국") and len(key[2:]) >= 4:
            short_name = key[2:]
            agency_aliases[short_name] = agency_aliases.get(short_name, ()) + (agency,)
    for alias, canonical_names in agency_aliases.items():
        if not re.search(
            r"(?<![가-힣a-z0-9])" + re.escape(alias) + r"(?=$|[^가-힣a-z0-9])",
            _normalize(question),
        ):
            continue
        if any(alias in _compact_agency_name(agency) for agency in matched):
            continue
        targets = {_compact_agency_name(name) for name in canonical_names}
        for agency in agencies:
            if _compact_agency_name(agency) in targets and agency not in matched:
                matched.append(agency)
    return matched


def find_project_ids(
    question: str,
    projects: list[ProjectInfo],
) -> list[str]:
    """기관 및 사업명 근거를 바탕으로 관련 문서 ID를 반환한다.

    기관명만 식별된 질문은 해당 기관의 전체 후보를 반환한다.
    사업 식별 단어가 포함됐지만 일치하는 사업이 없으면 빈 목록을 반환해
    기관 내 무관한 사업이 검색되는 것을 방지한다.
    """
    question = _normalize(question)
    exclusion = re.split(r"말고|제외하고|빼고", question, maxsplit=1)
    if len(exclusion) == 2:
        before_agencies = _get_agency_matches(exclusion[0], projects)
        after_agencies = _get_agency_matches(exclusion[1], projects)
        if before_agencies and not after_agencies:
            projects = [p for p in projects if p.agency_name in before_agencies]
        excluded_ids = set(find_project_ids(exclusion[0], projects))
        projects = [p for p in projects if p.project_id not in excluded_ids]
        if not _tokenize(exclusion[1]) and before_agencies:
            return list(dict.fromkeys(p.project_id for p in projects if p.project_id.strip()))
        question = exclusion[1].strip()
    projects = [project for project in projects if project.project_id.strip()]

    if not question.strip() or not projects:
        return []

    question_tokens = _tokenize(question)

    if not question_tokens:
        return []

    idf, total_documents = _calculate_idf(projects)

    if total_documents == 0:
        return []

    matched_agencies = _get_agency_matches(question, projects)

    if matched_agencies:
        search_projects = [
            project for project in projects if project.agency_name in matched_agencies
        ]

        agency_tokens = set()

        for agency in matched_agencies:
            agency_tokens.update(_tokenize(agency))
            agency_key = _compact_agency_name(agency)
            if agency_key.startswith("한국") and len(agency_key[2:]) >= 4:
                agency_tokens.update(_tokenize(agency_key[2:]))

            for alias, canonical in ALIASES.items():
                if _compact_agency_name(canonical) == _compact_agency_name(agency):
                    agency_tokens.update(_tokenize(alias))

        for alias, canonical_names in AGENCY_ALIASES.items():
            if any(
                _compact_agency_name(name) == _compact_agency_name(agency)
                for name in canonical_names
                for agency in matched_agencies
            ):
                agency_tokens.update(_tokenize(alias))
        project_query_tokens = question_tokens - agency_tokens

        if not project_query_tokens:
            return list(dict.fromkeys(project.project_id for project in search_projects))
    else:
        search_projects = projects
        project_query_tokens = question_tokens

    # 전체 사업명이 공백 차이만 있는 경우에 한해 원문 근거를 보존한다.
    compact_question = re.sub(r"\s+", "", _normalize(question))
    exact_titles = []
    title_positions = []
    for project in sorted(
        search_projects,
        key=lambda value: len(re.sub(r"\s+", "", _normalize(value.project_name))),
        reverse=True,
    ):
        title = re.sub(r"\s+", "", _normalize(project.project_name))
        if len(title) < 4:
            continue
        positions = [(m.start(), m.end()) for m in re.finditer(re.escape(title), compact_question)]
        independent = [
            (start, end)
            for start, end in positions
            if not any(
                left <= start and end <= right and end - start < right - left
                for left, right in title_positions
            )
        ]
        if independent:
            exact_titles.append(project)
            title_positions.extend(independent)
    if exact_titles:
        remainder = _normalize(question)
        for project in exact_titles:
            title = re.sub(r"\s+", "", _normalize(project.project_name))
            pattern = r"\s*".join(re.escape(char) for char in title)
            remainder = re.sub(pattern, " ", remainder)
        remaining_tokens = _tokenize(remainder)
        if matched_agencies:
            remaining_tokens -= agency_tokens
        extra_ids = (
            find_project_ids(remainder, search_projects) if remaining_tokens - ACTION_WORDS else []
        )
        return list(dict.fromkeys([p.project_id for p in exact_titles] + extra_ids))

    ordered = re.findall(r"[가-힣a-z0-9]+", _normalize(question))
    for left, right in zip(ordered, ordered[1:]):
        compound = left + right
        if left in project_query_tokens and right in project_query_tokens and len(compound) >= 4:
            if any(compound in _normalize(p.project_name) for p in search_projects):
                project_query_tokens -= {left, right}
                project_query_tokens.add(compound)

    if matched_agencies:
        for token in list(project_query_tokens):
            if len(token) < 4 or not re.fullmatch(r"[가-힣]+", token):
                continue
            if any(
                _match_project_tokens({token}, _tokenize(p.project_name)) for p in search_projects
            ):
                continue
            corrections = {
                word
                for p in search_projects
                for word in _tokenize(p.project_name)
                if len(word) >= len(token)
                and word[0] == token[0]
                and sum(a != b for a, b in zip(token, word[: len(token)])) == 1
            }
            if len(corrections) == 1:
                project_query_tokens.remove(token)
                project_query_tokens.update(corrections)

    all_matches = (
        set().union(
            *(
                _match_project_tokens(project_query_tokens, _tokenize(p.project_name))
                for p in search_projects
            )
        )
        if search_projects
        else set()
    )
    multiple_terms_covered = len(project_query_tokens) >= 2 and all_matches == project_query_tokens
    candidates = []

    for project in search_projects:
        project_tokens = _tokenize(project.project_name)
        project_matches = _match_project_tokens(
            project_query_tokens,
            project_tokens,
        )

        if not project_query_tokens or not project_matches:
            continue
        if project_query_tokens - ACTION_WORDS and not project_matches - ACTION_WORDS:
            continue

        project_coverage = _weighted_coverage(
            project_query_tokens,
            project_matches,
            idf,
        )

        distinctive_single_match = any(
            (len(token) >= 4 or bool(matched_agencies) or bool(re.fullmatch(r"[a-z]{2,}", token)))
            and sum(
                bool(_match_project_tokens({token}, _tokenize(item.project_name)))
                for item in search_projects
            )
            <= 2
            for token in project_matches
        )
        enough_project_evidence = len(project_matches) >= 2 or distinctive_single_match
        # 여러 사업명을 함께 물으면 문서마다 한 단어가 맞을 수 있다.
        coverage_floor = 0.0 if matched_agencies and multiple_terms_covered else 0.5
        required_coverage = min(0.5, coverage_floor) if matched_agencies else 0.75
        if not enough_project_evidence or project_coverage < required_coverage:
            continue

        agency_matches = question_tokens & _tokenize(project.agency_name)
        agency_bonus = 0.0

        if agency_matches:
            agency_coverage = len(agency_matches) / max(len(_tokenize(project.agency_name)), 1)
            agency_bonus = min(agency_coverage, 1.0) * 0.15

        score = project_coverage * 0.85 + agency_bonus
        candidates.append((project.project_id, score))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_score = candidates[0][1]

    return list(
        dict.fromkeys(
            project_id
            for project_id, score in candidates
            if score >= best_score - 0.10 or (matched_agencies and multiple_terms_covered)
        )
    )
