"""기존 오타 교정, RapidFuzz, 기관명 Aho-Corasick 비교.

실행: uv run python -m scripts.compare_search_b_libraries QUESTIONS --projects PROJECTS
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import ahocorasick

from bidmate.retrieval import project_finder as pf
from bidmate.schemas import ProjectInfo
from evaluation.score_project_finder import score as score_questions
from scripts.evaluate_project_finder import load_projects


def _correct_typo_reference(token: str, search_tokens: list[set[str]]) -> set[str]:
    return {
        word
        for terms in search_tokens
        for word in terms
        if len(word) >= len(token)
        and word[0] == token[0]
        and sum(a != b for a, b in zip(token, word[: len(token)])) == 1
    }


class _AgencyAutomatons:
    """전체 기관명용과 별칭용, 자동 탐색 기계를 따로 둔다.

    원래 규칙이 그랬던 것처럼 둘은 보는 문자열 자체가 다르다.
    - 전체 기관명(1번 통과): 공백까지 다 지운 압축 문자열에서 포함 여부만 본다.
      ('평택시청' vs '평택 시청'처럼 표기가 달라도 찾아야 하니까)
    - 짧은 별칭(3번 통과): 공백을 남긴 문자열에서, 앞뒤가 단어 경계인지까지 본다.
      ('평택시'의 '평택'처럼 다른 단어 안에 파묻힌 건 걸러야 하니까)
    압축 문자열은 공백을 지우면서 이 경계 정보도 같이 지워버리기 때문에,
    별칭은 반드시 공백 있는 문자열 쪽에서 찾아야 한다.
    """

    def __init__(self, projects: list[ProjectInfo]) -> None:
        self.exact = ahocorasick.Automaton()
        exact_keys = {
            pf._compact_agency_name(project.agency_name)
            for project in projects
            if len(pf._compact_agency_name(project.agency_name)) >= 3
        }
        for key in exact_keys:
            self.exact.add_word(key, key)
        self.exact.make_automaton()

        self.alias = ahocorasick.Automaton()
        alias_keys: dict[str, set[str]] = {}
        for alias, canonical_names in pf.AGENCY_ALIASES.items():
            alias_keys.setdefault(alias.casefold(), set()).update(canonical_names)
        for key, agencies in alias_keys.items():
            self.alias.add_word(key, (len(key), agencies))
        self.alias.make_automaton()


_WORD_CHAR = re.compile(r"[가-힣a-z0-9]")


def _has_clean_boundary(text: str, start: int, end: int) -> bool:
    """찾은 구간 앞뒤가 단어 경계인지 확인한다 (예: '평택시'의 '평택'은 거부).

    뒤쪽은 조사가 붙어도 경계로 인정한다 ('평택은 ...' -> 통과).
    """
    if start > 0 and _WORD_CHAR.match(text[start - 1]):
        return False
    tail = text[end + 1 :]
    if not tail or not _WORD_CHAR.match(tail[0]):
        return True
    for particle in pf.PARTICLES:
        if tail.startswith(particle):
            rest = tail[len(particle) :]
            if not rest or not _WORD_CHAR.match(rest[0]):
                return True
    return False


def _get_agency_matches_ahocorasick(
    automatons: _AgencyAutomatons, question: str, projects: list[ProjectInfo]
) -> list[str]:
    """자동 탐색 기계로 기관명을 찾되, 기존 규칙의 '긴 기관명 우선, 겹치면 짧은 쪽 버림'
    중복 제거(occupied 구간 추적)를 그대로 재현한다.

    자동 탐색 기계는 겹치는 패턴도 전부 찾아주는 게 장점이지만, 그래서 오히려
    원래 규칙이 일부러 버리던 '긴 기관명 안에 파묻힌 짧은 기관명'까지 다 살려내
    버린다. 이 함수는 한 번의 자동 탐색으로 모든 위치를 먼저 모은 뒤, 원래
    규칙과 똑같은 순서(긴 기관명부터)로 훑으며 이미 차지된 구간에 완전히
    포함되는 매치만 버리는 후처리를 별도로 둔다.
    """
    agencies = sorted(
        {project.agency_name for project in projects if project.agency_name},
        key=lambda value: len(pf._compact_agency_name(value)),
        reverse=True,
    )

    question_key = pf._compact_agency_name(question)
    spans_by_key: dict[str, list[tuple[int, int]]] = {}
    for end_index, key in automatons.exact.iter(question_key):
        start_index = end_index - len(key) + 1
        spans_by_key.setdefault(key, []).append((start_index, end_index))

    matched: list[str] = []
    occupied: list[tuple[int, int]] = []
    for agency in agencies:
        spans = spans_by_key.get(pf._compact_agency_name(agency))
        if not spans:
            continue
        independent = [
            (start, end)
            for start, end in spans
            if not any(left <= start and end <= right for left, right in occupied)
        ]
        if independent:
            matched.append(agency)
            occupied.extend(independent)

    spaced_question = pf._normalize(question)
    for end_index, (length, canonical_names) in automatons.alias.iter(spaced_question):
        start_index = end_index - length + 1
        if not _has_clean_boundary(spaced_question, start_index, end_index):
            continue
        canonical_keys = {pf._compact_agency_name(name) for name in canonical_names}
        matched.extend(
            agency
            for agency in agencies
            if pf._compact_agency_name(agency) in canonical_keys and agency not in matched
        )

    known_agencies = {project.agency_name for project in projects}
    return [agency for agency in matched if agency in known_agencies]


def _weighted_coverage_vocab_filter(
    query_tokens: set[str], matched_tokens: set[str], idf: dict[str, float]
) -> float:
    """IDF 가중치 대신, 목록표 사업명에 실제로 나오는 단어만 남기고 나머지는 버린다.

    질문이 길어질수록 사업명에 없는 단어가 늘어나는데, 기존 `_weighted_coverage`는
    그런 단어도 기본 가중치 1.0으로 분모에 넣어서 점수를 깎는다. 여기서는 사업명에
    아예 없는 단어는 분모에서 빼버리고, 남은 단어는 전부 동일한 가중치로 센다.

    "있다"의 기준은 완전 일치가 아니라 `_match_project_tokens`와 같은 부분 일치다
    (예: 질문의 "문서관리"는 사업명 토큰 "문서관리시스템"의 부분이라 '있는' 단어다).
    완전 일치로만 걸렀더니 이런 부분 일치 단어가 전부 '없는 단어'로 빠지는
    버그가 있었음.
    """
    known_tokens = {
        token
        for token in query_tokens
        if len(token) >= 2 and any(token in title_token for title_token in idf)
    }
    if not known_tokens:
        return 0.0
    return len(matched_tokens & known_tokens) / len(known_tokens)


def run_candidate(name: str, questions: list[dict], projects: list[ProjectInfo], patch) -> dict:
    restore = patch()
    started = time.perf_counter()
    try:
        result = score_questions(questions, projects, selector=pf.find_project_ids)
    finally:
        restore()

    result["ms_per_question"] = (time.perf_counter() - started) * 1000 / result["counted"]
    print(f"{name}  {result['correct']}/{result['counted']}  {result['ms_per_question']:.2f}ms")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--projects", required=True, type=Path)
    args = parser.parse_args()
    projects = load_projects(args.projects)
    with args.questions.open(encoding="utf-8-sig", newline="") as source:
        questions = list(csv.DictReader(source))

    def patch_reference():
        original = pf._correct_typo
        pf._correct_typo = _correct_typo_reference

        def restore():
            pf._correct_typo = original

        return restore

    run_candidate("기존 오타 교정", questions, projects, patch_reference)
    run_candidate("RapidFuzz", questions, projects, lambda: lambda: None)

    def patch_ahocorasick():
        """_get_agency_matches는 호출마다 다른 projects 범위를 받는다(비교·제외
        질의가 재귀 호출에서 범위를 좁혀 넘긴다). 자동 탐색 기계를 한 번만 전체
        목록으로 지어 두면 좁혀진 범위에만 있는 기관명은 영영 못 찾으므로,
        호출 시점의 범위로 매번 다시 짓되 같은 범위가 반복되면 캐시한다.
        """
        original = pf._get_agency_matches
        cache: dict[tuple[str, ...], _AgencyAutomatons] = {}

        def wrapped(question: str, search_projects: list[ProjectInfo]) -> list[str]:
            key = tuple(sorted({p.agency_name for p in search_projects if p.agency_name}))
            automatons = cache.get(key)
            if automatons is None:
                automatons = _AgencyAutomatons(search_projects)
                cache[key] = automatons
            return _get_agency_matches_ahocorasick(automatons, question, search_projects)

        pf._get_agency_matches = wrapped

        def restore():
            pf._get_agency_matches = original

        return restore

    run_candidate("Aho-Corasick", questions, projects, patch_ahocorasick)

    def patch_vocab_filter():
        original = pf._weighted_coverage
        pf._weighted_coverage = _weighted_coverage_vocab_filter

        def restore():
            pf._weighted_coverage = original

        return restore

    run_candidate("IDF 대신 목록표 단어 필터", questions, projects, patch_vocab_filter)


if __name__ == "__main__":
    main()
