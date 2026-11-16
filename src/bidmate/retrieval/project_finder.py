"""Search-B: 질문에 적합한 사업 문서를 선별한다.

기관과 사업의 조건을 적용하고, 사업명에서 드물게 나오는 단어로 후보를 좁힌다.

CSV 로딩이나 문서 검색은 담당하지 않으며,
선택한 문서의 원본 project_id만 반환한다.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bidmate.schemas import ProjectInfo


COMMON_TITLE_TERM_MIN_DOCUMENTS = 10


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
        "비교해줘",
        "비교해주세요",
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
    "만을",
    "만",
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
REQUEST_SEPARATOR = re.compile(r"\?\s*(?:그리고|또|아울러)\s+")

EXCLUSION_VERBS = r"말고|제외하고|제외한|제외하여|빼고"
PROJECT_EXCLUSION = re.compile(EXCLUSION_VERBS)
ISSUER_REFERENCE = re.compile(
    r"(?<![가-힣a-z0-9])(?:발주처|발주\s*기관)\s*(?:(?P<field>:)|[가이는은])\s*"
    r"(?!(?:어느|어떤|어디|누구|무슨|무엇|누가))"
    r"(?P<issuer>(?:(?:사단|재단)법인\s+)?[가-힣a-z0-9(][^?,;:]*?)"
    r"(?(field)(?:[,;]|$)|인(?=\s|$))"
)
ISSUER_PREFIX = re.compile(
    r"^(?:(?:주식회사|유한회사)\s+)?"
    r"(?P<issuer>[가-힣a-z0-9]{2,}(?:공단|재단|대학교|전자|연구원))\s+"
)
PROJECT_REFERENCE = re.compile(
    r"(?<![가-힣a-z0-9])(?:사업|용역)(?:에서|에|의|은|는)?(?=\s|[?.,]|$)"
)
COMPARISON_FIELD = re.compile(
    r"(?:예산|사업비|마감일|기간|요구사항|조건|일정|제출기한)(?:을|를)\s+"
    r"(?=[^?!]+?(?:과|와|랑|하고)\s+비교)"
)
OPEN_SCOPE_WORDS = {"어떤", "어느", "모든", "전체", "추천", "찾아줘", "뭐가", "무슨"}
FIELD_EXCLUSION = re.compile(
    r"(?<![가-힣a-z0-9])(?:부가세|부가가치세|vat|세금|예산|마감일|사업비|제출기한)"
    rf"(?:은|는|이|가|을|를)?\s*(?:{EXCLUSION_VERBS})(?![가-힣a-z0-9])"
)
PROJECT_YEAR = re.compile(
    r"(?<![가-힣a-z0-9])(20\d{2})(?:\s*년(?:도)?)?(?=(?:의|은|는|에)?(?![가-힣a-z0-9]))"
)
PROJECT_ROUND = re.compile(r"(?<![a-z0-9])(\d{1,2})\s*차(?=(?:의|은|는|에)?(?![가-힣a-z0-9]))")


TITLE_END = re.compile(
    rf"(?:사업|용역|비|예산|마감일|기간)?(?:{'|'.join(PARTICLES)}|에|인)?(?![가-힣a-z0-9])"
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


def _expand_issuer_keys(issuer_keys: set[str]) -> set[str]:
    keys = set(issuer_keys)
    for alias, canonical in ALIASES.items():
        if _compact_agency_name(alias) in issuer_keys:
            keys.add(_compact_agency_name(canonical))
    for alias, names in AGENCY_ALIASES.items():
        if _compact_agency_name(alias) in issuer_keys:
            keys.update(_compact_agency_name(name) for name in names)
    return keys


def _issuer_key_matches(agency: str, issuer_keys: set[str]) -> bool:
    key = _compact_agency_name(agency)
    return key in issuer_keys or (key.startswith("한국") and key[2:] in issuer_keys)


def _has_unlisted_issuer_prefix(question: str, projects: list[ProjectInfo]) -> bool:
    question = question.lstrip()
    prefix = ISSUER_PREFIX.match(question)
    if prefix is None or PROJECT_REFERENCE.search(question, prefix.end()) is None:
        return False
    issuer = prefix["issuer"]
    keys = _expand_issuer_keys({_compact_agency_name(issuer)})
    question = " ".join(question.split())
    for project in projects:
        agency = " ".join(_normalize(project.agency_name).split())
        if (
            _issuer_key_matches(project.agency_name, keys)
            or (agency.startswith(issuer + " ") and question.startswith(agency + " "))
            or _normalize(project.project_name).startswith(issuer + " ")
        ):
            return False
    return True


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

    issuer_names = {
        _compact_agency_name(match["issuer"])
        for match in ISSUER_REFERENCE.finditer(_normalize(question))
    }
    if issuer_names:
        keys = _expand_issuer_keys(issuer_names)
        return [agency for agency in agencies if _issuer_key_matches(agency, keys)]

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
    """제목의 연도·차수를 비교하고 같은 사업의 버전 없는 문서도 구분한다."""
    for pattern in (PROJECT_YEAR, PROJECT_ROUND):
        requested = set(pattern.findall(question))
        if len(requested) != 1:
            continue
        titles = []
        for project in projects:
            normalized = _normalize(project.project_name)
            family = (
                project.agency_name,
                tuple(re.findall(r"[가-힣a-z0-9]+", pattern.sub(" ", normalized))),
            )
            titles.append((project, family, set(pattern.findall(normalized))))
        versioned_families = {family for _, family, editions in titles if editions}
        projects = [
            project
            for project, family, editions in titles
            if editions & requested or (not editions and family not in versioned_families)
        ]
    return projects


def _title_edition_pattern(title: str) -> tuple[str, tuple[str, ...]]:
    markers = sorted(
        (match.start(), match.end(), match[1], replacement)
        for pattern, replacement in (
            (PROJECT_YEAR, r"(20\d{2})(?:년(?:도)?)?"),
            (PROJECT_ROUND, r"(\d{1,2})차"),
        )
        for match in pattern.finditer(title)
    )
    parts = []
    start = 0
    for left, right, _, replacement in markers:
        parts.extend((re.escape(re.sub(r"\s+", "", title[start:left])), replacement))
        start = right
    parts.append(re.escape(re.sub(r"\s+", "", title[start:])))
    return "".join(parts), tuple(number for _, _, number, _ in markers)


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


def _split_project_exclusion(question: str) -> tuple[str, str] | None:
    """사업을 제외하는 표현만 분리한다."""
    for operator in PROJECT_EXCLUSION.finditer(question):
        before = question[: operator.start()]
        content = CONTENT_BOUNDARY.search(before)
        if (
            content
            and content.group().strip() not in {"은", "는"}
            and before[content.end() :].strip()
        ):
            continue
        return before, question[operator.end() :]
    return None


def find_project_ids(
    question: str,
    projects: list[ProjectInfo],
) -> list[str]:
    """기관 및 사업명 근거를 바탕으로 관련 문서 ID를 반환한다.

    기관명만 식별된 질문은 해당 기관의 전체 후보를 반환한다.
    특정 사업을 물었는데 일치하는 후보가 없으면 빈 목록을 반환한다.
    """
    selected = _find_project_ids(question, projects)
    if len(selected) > 1 or ("?" not in question and "？" not in question):
        return selected
    normalized = _normalize(question)
    parts = REQUEST_SEPARATOR.split(normalized, maxsplit=2)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        return selected
    if PROJECT_EXCLUSION.search(normalized):
        return selected
    first, second = (_find_project_ids(part, projects, allow_agency_pool=False) for part in parts)
    if len(first) == len(second) == 1 and first != second and selected in (first, second):
        return first + second
    if not selected:
        if len(first) == 1 and not second:
            return _keep_exact_project(parts[0], first[0], parts[1], projects)
        if len(second) == 1 and not first:
            return _keep_exact_project(parts[1], second[0], parts[0], projects)
    return selected


def _keep_exact_project(
    question: str, project_id: str, unmatched: str, projects: list[ProjectInfo]
) -> list[str]:
    if not has_explicit_project_scope(unmatched, projects):
        return []
    if _find_project_ids(unmatched, projects):
        return []
    for project in projects:
        if project.project_id != project_id:
            continue
        title = _normalize(project.project_name).split()
        if sum(map(len, title)) < 4:
            continue
        pattern = r"\s*".join(map(re.escape, title))
        if any(
            TITLE_END.match(question, match.end())
            for match in re.finditer(rf"(?<![가-힣a-z0-9]){pattern}", question)
        ):
            return [project_id]
    return []


def select_project_scope(question: str, projects: list[ProjectInfo]) -> tuple[list[str], bool]:
    """문서 ID와 명시한 사업 범위에 후보가 없는지를 반환한다."""
    project_ids = find_project_ids(question, projects)
    return project_ids, not project_ids and has_explicit_project_scope(question, projects)


def has_explicit_project_scope(question: str, projects: list[ProjectInfo]) -> bool:
    """문서 후보가 비었을 때 질문에 목록 조회 범위가 명시됐는지 확인한다."""
    if not projects:
        return False
    question = _normalize(question).strip()
    if ISSUER_REFERENCE.search(question) or _has_unlisted_issuer_prefix(question, projects):
        return True
    detail = CONTENT_BOUNDARY.search(question)
    subject = question[: detail.start()] if detail else question
    if _has_explicit_edition(subject, projects):
        return True
    reference = PROJECT_REFERENCE.search(question)
    if reference is None:
        return False
    subject = question[: reference.start()]
    subject_tokens = _tokenize(subject) - ACTION_WORDS
    if not subject_tokens or subject_tokens & OPEN_SCOPE_WORDS:
        return False
    agencies = _get_agency_matches(subject, projects)
    if not agencies:
        return False
    agency_tokens: set[str] = set()
    agency_keys = {_compact_agency_name(agency) for agency in agencies}
    for agency in agencies:
        agency_tokens.update(_tokenize(agency))
        key = _compact_agency_name(agency)
        if key.startswith("한국"):
            agency_tokens.update(_tokenize(key[2:]))
    for alias, names in AGENCY_ALIASES.items():
        if agency_keys & {_compact_agency_name(name) for name in names}:
            agency_tokens.update(_tokenize(alias))
    return not _is_agency_wide_query(subject_tokens - agency_tokens)


def _has_explicit_edition(subject: str, projects: list[ProjectInfo]) -> bool:
    without_editions = PROJECT_ROUND.sub(" ", PROJECT_YEAR.sub(" ", subject))
    if without_editions == subject or _tokenize(subject) & OPEN_SCOPE_WORDS:
        return False
    positions = [index for index, char in enumerate(subject) if not char.isspace()]
    compact_subject = "".join(subject[index] for index in positions)
    agencies = _get_agency_matches(subject, projects)
    prefixes = {"", *(_compact_agency_name(agency) for agency in agencies)}
    starts = {len(prefix) for prefix in prefixes if compact_subject.startswith(prefix)}
    for project in projects:
        if agencies and project.agency_name not in agencies:
            continue
        pattern, editions = _title_edition_pattern(_normalize(project.project_name))
        if not editions:
            continue
        for start in starts:
            match = re.match(pattern, compact_subject[start:])
            if match and TITLE_END.match(subject, positions[start + match.end() - 1] + 1):
                return True
    unversioned_ids = set(_find_project_ids(without_editions, projects, allow_agency_pool=False))
    candidates = [project for project in projects if project.project_id in unversioned_ids]
    return bool(candidates) and not _exclude_conflicting_editions(subject, candidates)


def _find_project_ids(
    question: str,
    projects: list[ProjectInfo],
    *,
    allow_agency_pool: bool = True,
    inherited_agencies: list[str] | None = None,
) -> list[str]:
    """문서 ID를 고른다. 추가 사업을 찾을 때는 기관 전체를 고르지 않는다."""
    question = FIELD_EXCLUSION.sub(" ", _normalize(question))
    inherited_agencies = inherited_agencies or []
    exclusion = _split_project_exclusion(question)
    if exclusion:
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
        if PROJECT_EXCLUSION.search(question):
            return _find_project_ids(
                question,
                projects,
                allow_agency_pool=allow_agency_pool,
                inherited_agencies=inherited_agencies,
            )
    projects = [project for project in projects if project.project_id.strip()]

    if not question.strip() or not projects:
        return []

    issuer_spans = [match.span() for match in ISSUER_REFERENCE.finditer(question)]
    issuer_explicit = bool(issuer_spans)
    question_tokens = _tokenize(ISSUER_REFERENCE.sub(r"\g<issuer> ", question))

    if not question_tokens:
        return []

    matched_agencies = _get_agency_matches(question, projects)
    if not matched_agencies and not issuer_explicit:
        matched_agencies = inherited_agencies
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

    character_positions = [index for index, char in enumerate(question) if not char.isspace()]
    compact_question = "".join(question[index] for index in character_positions)
    exact_titles = []
    title_positions: list[tuple[int, int]] = []
    edition_positions: list[tuple[int, int]] = []
    conflicting_spans: list[tuple[int, int]] = []
    conflicting_editions: set[str] = set()
    matching_editions: set[str] = set()
    exact_spans: dict[str, list[tuple[int, int]]] = {}
    embedded_titles: set[str] = set()
    agency_keys = {_compact_agency_name(agency) for agency in matched_agencies}
    has_edition = bool(PROJECT_YEAR.search(question) or PROJECT_ROUND.search(question))
    for project in sorted(
        search_projects,
        key=lambda value: len(re.sub(r"\s+", "", _normalize(value.project_name))),
        reverse=True,
    ):
        normalized_title = _normalize(project.project_name)
        title = re.sub(r"\s+", "", normalized_title)
        if len(title) < 4:
            continue
        title_matches = [(match, ()) for match in re.finditer(re.escape(title), compact_question)]
        if has_edition:
            edition_pattern, editions = _title_edition_pattern(normalized_title)
            if editions:
                title_matches.extend(
                    (match, editions) for match in re.finditer(edition_pattern, compact_question)
                )
        for match, editions in title_matches:
            start, end = match.span()
            if any(
                left <= start and end <= right and end - start < right - left
                for left, right in title_positions
            ):
                continue
            raw_start = character_positions[start]
            raw_end = character_positions[end - 1] + 1
            starts_name = (
                raw_start == 0
                or not re.fullmatch(r"[가-힣a-z0-9]", question[raw_start - 1])
                or compact_question[:start] in agency_keys
            )
            if not starts_name or not TITLE_END.match(question, raw_end):
                if not editions:
                    embedded_titles.add(project.project_id)
                continue
            if editions:
                edition_positions.append((start, end))
                if match.groups() == editions:
                    matching_editions.add(project.project_id)
                else:
                    conflicting_editions.add(project.project_id)
                    conflicting_spans.append((raw_start, raw_end))
                continue
            title_positions.append((start, end))
            exact_spans.setdefault(project.project_id, []).append((raw_start, raw_end))
            exact_titles.append(project)
    if embedded_titles:
        search_projects = [
            project
            for project in search_projects
            if project.project_id not in embedded_titles or project.project_id in exact_spans
        ]
    protected_positions = title_positions + edition_positions
    detail = None
    for marker in CONTENT_BOUNDARY.finditer(question):
        offset = len(re.sub(r"\s+", "", question[: marker.start()]))
        if not any(start < offset < end for start, end in protected_positions):
            detail = marker
            break

    field_separators = list(COMPARISON_FIELD.finditer(question))
    title_words: set[str] | None = None
    boundaries = []
    for match in sorted(
        [*PROJECT_SEPARATOR.finditer(question), *field_separators],
        key=lambda candidate: candidate.start(),
    ):
        if any(start <= match.start() < end for start, end in issuer_spans):
            continue
        if match.group() in {"과", "와", "랑", "이랑", "하고"}:
            word = re.search(r"[가-힣a-z0-9]+$", question[: match.end()])
            if word:
                if title_words is None:
                    title_words = {
                        token
                        for project in search_projects
                        for token in re.findall(r"[가-힣a-z0-9]+", _normalize(project.project_name))
                    }
                if word.group() in title_words:
                    continue
        offset = len(re.sub(r"\s+", "", question[: match.start()]))
        if not any(start < offset < end for start, end in protected_positions):
            boundaries.append(match)

    separators = []
    compares_projects = bool(re.search(r"각각|비교", question))
    for index, match in enumerate(boundaries):
        if detail and match.start() > detail.start() and match not in field_separators:
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
        missing_qualified_part = False
        unlisted_issuer = False
        for part in parts:
            explicit_agencies = _get_agency_matches(part, projects)
            issuer_named = bool(ISSUER_REFERENCE.search(part))
            unlisted_prefix = _has_unlisted_issuer_prefix(part, projects)
            if issuer_named or unlisted_prefix or explicit_agencies:
                unlisted_issuer = unlisted_prefix or (issuer_named and not explicit_agencies)
            if unlisted_issuer:
                missing_qualified_part = True
                continue
            part_agencies = explicit_agencies or part_agencies
            qualified_part = bool(
                issuer_named or PROJECT_YEAR.search(part) or PROJECT_ROUND.search(part)
            )
            part_ids = _find_project_ids(
                part,
                projects,
                allow_agency_pool=False,
                inherited_agencies=part_agencies,
            )
            if not part_ids and qualified_part:
                missing_qualified_part = True
            if (
                not part_ids
                and not explicit_agencies
                and not qualified_part
                and part_agencies
                and coordinated_ids
            ):
                part_ids = _find_project_ids(part, projects, allow_agency_pool=False)
            if len(part_ids) > 1:
                break
            coordinated_ids.extend(part_ids)
        else:
            unique_ids = list(dict.fromkeys(coordinated_ids))
            if len(unique_ids) >= 2 or (
                unique_ids and (compares_projects or missing_qualified_part)
            ):
                return unique_ids

    if (issuer_explicit and not matched_agencies) or _has_unlisted_issuer_prefix(
        question, projects
    ):
        return []

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
    if conflicting_spans and not matching_editions:
        exact_titles = [
            project
            for project in exact_titles
            if any(
                not any(left <= start and end <= right for left, right in conflicting_spans)
                for start, end in exact_spans[project.project_id]
            )
        ]
        if not exact_titles:
            return []
    scoped_projects = _exclude_conflicting_editions(
        question[: detail.start()] if detail else question,
        search_projects,
    )
    excluded_editions = conflicting_editions - matching_editions - exact_spans.keys()
    if excluded_editions:
        scoped_projects = [
            project for project in scoped_projects if project.project_id not in excluded_editions
        ]
    if len(scoped_projects) != len(search_projects):
        search_projects = scoped_projects
        exact_titles = [project for project in exact_titles if project in search_projects]
    if not search_projects:
        return []

    if exact_titles:
        remaining_characters = list(question)
        for project in exact_titles:
            for start, end in exact_spans[project.project_id]:
                remaining_characters[start:end] = [" "] * (end - start)
        remainder = "".join(remaining_characters)
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
    document_frequency = Counter(
        token for project in projects for token in tokens_by_title[project.project_name]
    )
    name_query_terms = project_query_tokens - ACTION_WORDS
    required_terms = {
        token
        for token in name_query_terms
        if 0 < match_counts[token]
        and max(match_counts[token], document_frequency[token]) < COMMON_TITLE_TERM_MIN_DOCUMENTS
    }
    if not required_terms:
        return []

    candidates = [
        (project.project_id, len(matches & required_terms), len(matches & ACTION_WORDS))
        for project, matches, title_terms in zip(
            search_projects, matches_by_project, search_tokens, strict=True
        )
        if matches & required_terms
        and (
            len(matches & required_terms) > 1
            or any(
                len(token) >= 4
                or re.fullmatch(r"[a-z]{2,}", token)
                or matched_agencies
                and (token in title_terms or len(name_query_terms) == 1)
                for token in matches & required_terms
            )
        )
    ]
    if not candidates:
        return []

    best = max((name_hits, action_hits) for _, name_hits, action_hits in candidates)
    return list(
        dict.fromkeys(
            project_id
            for project_id, name_hits, action_hits in candidates
            if (name_hits, action_hits) == best
        )
    )
