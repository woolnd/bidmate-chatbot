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

# 의미가 명확한 표현만 별칭으로 연결한다.
ALIASES = {
    "국민연금": "국민연금공단",
    "이러닝": "이러닝시스템",
    "고려대": "고려대학교",
    "포털": "차세대포털",
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
        token: math.log(
            (total_documents + 1) / (frequency + 1)
        ) + 1
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


def find_project_ids(
    question: str,
    projects: list[ProjectInfo],
) -> list[str]:
    """질문과 관련된 문서의 원본 project_id 목록을 반환한다.

    검색 기준:
    1. 사업명과 기관명을 각각 비교한다.
    2. 사업명 단어의 희귀도에 따라 가중치를 부여한다.
    3. 기관명 일치는 별도 점수로 반영한다.
    4. 충분한 사업명 근거가 있는 후보만 선택한다.
    5. 점수가 비슷한 후보는 함께 반환할 수 있다.

    원본 project_id는 변경하거나 재구성하지 않는다.
    """

    if not question.strip() or not projects:
        return []

    question_tokens = _tokenize(question)

    if not question_tokens:
        return []

    idf, total_documents = _calculate_idf(projects)

    if total_documents == 0:
        return []

    candidates: list[tuple[str, float, float]] = []

    for project in projects:
        project_tokens = _tokenize(project.project_name)
        agency_tokens = _tokenize(project.agency_name)

        # 후보 기관명과 일치하는 단어는 사업명 점수 계산에서 분리한다.
        agency_matches = question_tokens & agency_tokens
        project_query_tokens = question_tokens - agency_matches

        project_matches = project_query_tokens & project_tokens

        if not project_query_tokens or not project_matches:
            continue

        project_coverage = _weighted_coverage(
            project_query_tokens,
            project_matches,
            idf,
        )

        # 한 개의 단어만 일치하는 경우에는 매우 희귀하고
        # 구별력이 높은 단어일 때만 허용한다.
        distinctive_single_match = (
            len(project_matches) == 1
            and any(
                len(token) >= 4
                and sum(
                    1
                    for item in projects
                    if token in _tokenize(item.project_name)
                ) <= 2
                for token in project_matches
            )
        )

        enough_project_evidence = (
            len(project_matches) >= 2
            or distinctive_single_match
        )

        # 기관명 일치 여부는 사업명 점수와 별도로 반영한다.
        agency_bonus = 0.0

        if agency_matches:
            agency_coverage = len(agency_matches) / max(
                len(agency_tokens),
                1,
            )
            agency_bonus = min(agency_coverage, 1.0) * 0.15

        # 사업명 근거가 부족하면 기관명이 일치해도 선택하지 않는다.
        if not enough_project_evidence or project_coverage < 0.50:
            continue

        score = project_coverage * 0.85 + agency_bonus

        candidates.append(
            (
                project.project_id,
                score,
                project_coverage,
            )
        )

    if not candidates:
        return []

    # 점수가 높은 후보부터 정렬한다.
    candidates.sort(key=lambda item: item[1], reverse=True)

    best_score = candidates[0][1]

    # 최고 점수와 충분히 가까운 후보만 남긴다.
    # 점수가 낮은 공통 단어 기반 오탐은 제외한다.
    selected = [
        project_id
        for project_id, score, _ in candidates
        if score >= best_score - 0.10
    ]

    return selected
