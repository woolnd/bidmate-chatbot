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
        "알려줘",
        "각각",
        "비교",
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
    "코레일": ("한국철도공사",),
    "축평원": ("축산물품질평가원",),
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
    "이랑",
    "랑",
    "하고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "도",
    "의",
)


CONTENT_BOUNDARY = re.compile(
    r"(?:(?<=사업)|(?<=용역)|(?<=시스템)|(?<=운영)|(?<=구축)|(?<=개발)|(?<=고도화)|(?<=개선))"
    r"(?:에서|에서는|에|에는|의|은|는)\s+"
    r"|(?<!\S)(?:사용\s*시(?:에|에는)?|사용할\s+때|쓸\s+때)\s+"
)
PROJECT_SEPARATOR = re.compile(
    r"(?<=[가-힣a-z0-9])(?:이랑|하고|랑|과|와)(?=\s)|\s+(?:및|그리고|vs\.?)\s+"
    r"|,\s*|[/·&+]\s*(?=[가-힣a-z])"
)

PROJECT_YEAR = re.compile(
    r"(?<![가-힣a-z0-9])(20\d{2})(?:\s*년(?:도)?)?(?=(?:의|은|는|에)?(?![가-힣a-z0-9]))"
)
PROJECT_ROUND = re.compile(r"(?<![a-z0-9])(\d{1,2})\s*차(?=(?:의|은|는|에)?(?![가-힣a-z0-9]))")


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

    normalized = PROJECT_ROUND.sub(r" \1차", normalized)
    raw_tokens = re.findall(r"[가-힣a-z0-9]+", normalized)
    tokens: set[str] = set()

    for original_token in raw_tokens:
        token = original_token

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

        if not token or token in STOP_WORDS or PROJECT_YEAR.fullmatch(token):
            continue

        token = ALIASES.get(token, token)
        tokens.add(token)

    return tokens


def _calculate_idf(
    projects: list[ProjectInfo],
    tokens_by_title: dict[str, set[str]],
) -> dict[str, float]:
    """미리 나눈 사업명 단어에서 역문서빈도(IDF)를 계산한다."""
    document_frequency: Counter[str] = Counter()
    for project in projects:
        document_frequency.update(tokens_by_title[project.project_name])
    return {
        token: math.log((len(projects) + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }


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
    particle_pattern = "|".join(PARTICLES)
    for alias, canonical_names in agency_aliases.items():
        if not re.search(
            rf"(?<![가-힣a-z0-9]){re.escape(alias)}(?:{particle_pattern})?(?![가-힣a-z0-9])",
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


def _is_agency_wide_query(project_query_tokens: set[str]) -> bool:
    """기관 전체 검색을 허용하는 일반 질문인지 판단한다."""
    generic_terms = {
        "사업",
        "사업들",
        "사업목록",
        "사업정보",
        "전체",
        "모든",
        "목록",
        "리스트",
        "요구사항",
        "관련",
        "알려줘",
        "알려주세요",
        "보여줘",
        "보여주세요",
        "보여",
        "줘",
        "주세요",
        "좀",
        "요약해줘",
        "나머지",
        "다른",
        "다른사업",
    }
    return not project_query_tokens or project_query_tokens <= generic_terms


def _exclude_conflicting_editions(question: str, projects: list[ProjectInfo]) -> list[ProjectInfo]:
    """질문의 연도·차수와 다른 값이 제목에 명시된 문서를 뺀다."""
    for pattern in (PROJECT_YEAR, PROJECT_ROUND):
        requested = set(pattern.findall(question))
        if len(requested) != 1:
            continue
        projects = [
            project
            for project in projects
            if not (editions := set(pattern.findall(_normalize(project.project_name))))
            or not editions.isdisjoint(requested)
        ]
    return projects


def _fill_shared_project_name(parts: list[str], agency_tokens: set[str]) -> None:
    """연도나 차수만 먼저 나열했으면 마지막 구간에 나온 사업명을 이어 붙인다."""
    for index in range(len(parts) - 2, -1, -1):
        first = parts[index]
        second = parts[index + 1].lstrip()
        for pattern in (PROJECT_YEAR, PROJECT_ROUND):
            first_edition = pattern.search(first)
            second_edition = pattern.match(second)
            if first_edition is None or second_edition is None:
                continue
            first_name = first[: first_edition.start()] + first[first_edition.end() :]
            if _tokenize(first_name) - agency_tokens:
                continue
            shared_name = second[second_edition.end() :].strip()
            if not _is_agency_wide_query(_tokenize(shared_name)):
                parts[index] = f"{first} {shared_name}"
                break


def find_project_ids(
    question: str,
    projects: list[ProjectInfo],
) -> list[str]:
    """기관 및 사업명 근거를 바탕으로 관련 문서 ID를 반환한다.

    기관명만 식별된 질문은 해당 기관의 전체 후보를 반환한다.
    특정 사업을 물었는데 일치하는 후보가 없으면 빈 목록을 반환한다.
    """
    return _find_project_ids(question, projects)


def _find_project_ids(
    question: str,
    projects: list[ProjectInfo],
    *,
    allow_agency_pool: bool = True,
    inherited_agencies: list[str] | None = None,
) -> list[str]:
    """문서 ID를 고른다. 추가 사업을 찾을 때는 기관 전체를 고르지 않는다."""
    question = _normalize(question)
    inherited_agencies = inherited_agencies or []
    exclusion = re.split(r"말고|제외하고|빼고", question, maxsplit=1)
    if len(exclusion) == 2:
        before_agencies = _get_agency_matches(exclusion[0], projects) or inherited_agencies
        after_agencies = _get_agency_matches(exclusion[1], projects)
        if before_agencies and not after_agencies:
            projects = [p for p in projects if p.agency_name in before_agencies]
            inherited_agencies = before_agencies
        excluded_ids = set(
            _find_project_ids(
                exclusion[0],
                projects,
                inherited_agencies=before_agencies,
            )
        )
        projects = [p for p in projects if p.project_id not in excluded_ids]
        if _is_agency_wide_query(_tokenize(exclusion[1])) and before_agencies:
            return list(dict.fromkeys(p.project_id for p in projects if p.project_id.strip()))
        question = exclusion[1].strip()
    projects = [project for project in projects if project.project_id.strip()]

    if not question.strip() or not projects:
        return []

    question_tokens = _tokenize(question)

    if not question_tokens:
        return []

    matched_agencies = _get_agency_matches(question, projects) or inherited_agencies
    agency_tokens: set[str] = set()

    if matched_agencies:
        search_projects = [
            project for project in projects if project.agency_name in matched_agencies
        ]

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

        if _is_agency_wide_query(project_query_tokens):
            return (
                list(dict.fromkeys(project.project_id for project in search_projects))
                if allow_agency_pool
                else []
            )
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
    detail = None
    for marker in CONTENT_BOUNDARY.finditer(question):
        offset = len(re.sub(r"\s+", "", question[: marker.start()]))
        if not any(start < offset < end for start, end in title_positions):
            detail = marker
            break

    boundaries = []
    for match in PROJECT_SEPARATOR.finditer(question):
        offset = len(re.sub(r"\s+", "", question[: match.start()]))
        if not any(start < offset < end for start, end in title_positions):
            boundaries.append(match)

    separators = []
    compares_projects = bool(re.search(r"각각|비교", question))
    for index, match in enumerate(boundaries):
        if detail and match.start() > detail.start():
            limit = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(question)
            next_subject = CONTENT_BOUNDARY.search(question, match.end(), limit)
            if next_subject is None:
                continue
            first_is_subject = detail.group().strip() in {"의", "은", "는"}
            second_is_subject = next_subject.group().strip() in {"의", "은", "는"}
            second_names_project = question[match.end() : next_subject.start()].endswith(
                ("사업", "용역")
            )
            if not (
                (first_is_subject and second_is_subject)
                or (compares_projects and second_names_project)
            ):
                continue
        separators.append(match)

    if separators:
        parts = []
        start = 0
        for separator in separators:
            parts.append(question[start : separator.start()])
            start = separator.end()
        parts.append(question[start:])
        _fill_shared_project_name(parts, agency_tokens)

        # 먼저 나온 기관은 기관명이 생략된 다음 사업에만 이어 쓴다.
        part_agencies = inherited_agencies
        coordinated_ids = []
        for part in parts:
            explicit_agencies = _get_agency_matches(part, projects)
            part_agencies = explicit_agencies or part_agencies
            part_ids = _find_project_ids(
                part,
                projects,
                allow_agency_pool=False,
                inherited_agencies=part_agencies,
            )
            if not part_ids and not explicit_agencies and part_agencies and coordinated_ids:
                part_ids = _find_project_ids(part, projects, allow_agency_pool=False)
            if len(part_ids) > 1:
                break
            coordinated_ids.extend(part_ids)
        else:
            unique_ids = list(dict.fromkeys(coordinated_ids))
            if len(unique_ids) >= 2 or (unique_ids and "각각" in question):
                return unique_ids

    if detail:
        mention = question[: detail.start()]
        mention_ids = _find_project_ids(
            mention,
            projects,
            allow_agency_pool=False,
            inherited_agencies=inherited_agencies,
        )
        if len(mention_ids) == 1:
            return mention_ids
        mention_tokens = _tokenize(mention) - ACTION_WORDS
        if matched_agencies:
            mention_tokens -= agency_tokens
        if not mention_ids and not _is_agency_wide_query(mention_tokens):
            return []

    # 본문에 나오는 납기나 지원 연도는 사업 이름의 연도와 따로 본다.
    scoped_projects = _exclude_conflicting_editions(
        question[: detail.start()] if detail else question,
        search_projects,
    )
    if len(scoped_projects) != len(search_projects):
        search_projects = scoped_projects
        exact_titles = [project for project in exact_titles if project in search_projects]
    if not search_projects:
        return []

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
            _find_project_ids(
                remainder,
                search_projects,
                allow_agency_pool=False,
                inherited_agencies=matched_agencies,
            )
            if remaining_tokens - ACTION_WORDS and not _is_agency_wide_query(remaining_tokens)
            else []
        )
        return list(dict.fromkeys([p.project_id for p in exact_titles] + extra_ids))

    ordered = re.findall(r"[가-힣a-z0-9]+", _normalize(question))
    for left, right in zip(ordered, ordered[1:]):
        compound = left + right
        if left in project_query_tokens and right in project_query_tokens and len(compound) >= 4:
            if any(compound in _normalize(p.project_name) for p in search_projects):
                project_query_tokens -= {left, right}
                project_query_tokens.add(compound)

    tokens_by_title = {title: _tokenize(title) for title in {p.project_name for p in projects}}
    search_tokens = [tokens_by_title[p.project_name] for p in search_projects]
    idf = _calculate_idf(projects, tokens_by_title)

    if matched_agencies:
        for token in list(project_query_tokens):
            if len(token) < 4 or not re.fullmatch(r"[가-힣]+", token):
                continue
            if any(_match_project_tokens({token}, terms) for terms in search_tokens):
                continue
            corrections = {
                word
                for terms in search_tokens
                for word in terms
                if len(word) >= len(token)
                and word[0] == token[0]
                and sum(a != b for a, b in zip(token, word[: len(token)])) == 1
            }
            if len(corrections) == 1:
                project_query_tokens.remove(token)
                project_query_tokens.update(corrections)

    matches_by_project = [
        _match_project_tokens(project_query_tokens, terms) for terms in search_tokens
    ]
    match_counts = Counter(token for matches in matches_by_project for token in matches)
    multiple_terms_covered = (
        len(project_query_tokens) >= 2 and set(match_counts) == project_query_tokens
    )
    candidates = []

    for project, project_matches in zip(search_projects, matches_by_project, strict=True):
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
            and match_counts[token] <= 2
            for token in project_matches
        )
        enough_project_evidence = len(project_matches) >= 2 or distinctive_single_match
        # 여러 사업명을 함께 물으면 문서마다 한 단어가 맞을 수 있다.
        coverage_floor = 0.0 if matched_agencies and multiple_terms_covered else 0.5
        required_coverage = min(0.5, coverage_floor) if matched_agencies else 0.75
        if not enough_project_evidence or project_coverage < required_coverage:
            continue

        project_agency_tokens = _tokenize(project.agency_name)
        agency_matches = question_tokens & project_agency_tokens
        agency_bonus = 0.0

        if agency_matches:
            agency_coverage = len(agency_matches) / max(len(project_agency_tokens), 1)
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
