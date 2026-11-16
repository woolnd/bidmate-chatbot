from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from urllib.request import Request

from pydantic import BaseModel

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo
from evaluation.score_project_finder import score
from evaluation.search_b_experiments.build_local_chroma import LocalEmbeddings
from scripts.evaluate_project_finder import load_projects

MODEL = "qwen3:1.7b"
SYSTEM = """한국어 입찰 문서 질문의 검색 조건만 JSON으로 읽어라.
intent는 compare(복수/각각), exclude(빼고), single(하나), list(기관 전체), unknown 중 하나.
include에는 찾을 사업의 원문 구절을 하나씩, exclude에는 뺄 사업의 원문 구절을 넣어라.
기관·연도·차수가 각 사업 앞에 있으면 같은 구절에 포함한다. 공통 기관은 scope에 원문 그대로 적는다.
원문에 없는 사업명이나 기관명을 만들지 않는다. 질문에 없는 값은 빈 문자열 또는 빈 배열로 둔다.
field에는 물어본 항목을 짧게 적는다. 예산·마감일·요구사항 같은 내용 항목을 사업명에 포함하지 않는다.
'부가세 빼고'처럼 금액 조건을 물어본 경우 사업 제외로 읽지 않는다.
복수 비교는 각 사업을 분리한다. 나머지 사업을 묻는 경우 include는 빈 배열이다."""


class Slots(BaseModel):
    intent: str
    scope: str
    include: list[str]
    exclude: list[str]
    field: str


def key(text):
    return re.sub(r"[^가-힣a-z0-9]", "", text.casefold())


def parse(client, question):
    data = {
        "model": MODEL,
        "think": False,
        "stream": False,
        "keep_alive": "2m",
        "format": Slots.model_json_schema(),
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "options": {"temperature": 0, "seed": 1611, "num_predict": 260},
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        json.dumps(data).encode(),
        {"Content-Type": "application/json"},
    )
    with client.open(request, timeout=90) as response:
        answer = json.load(response)
    return Slots.model_validate_json(answer["message"]["content"])


def resolve(slots, question, projects):
    values = [slots.scope, *slots.include, *slots.exclude]
    literal = all(not word.strip() or key(word) and key(word) in key(question) for word in values)
    if not literal or slots.intent not in {"compare", "exclude", "single", "list", "unknown"}:
        return [], False
    if slots.intent == "compare" and len(slots.include) < 2:
        return [], False
    if slots.intent == "exclude" and not slots.exclude:
        return [], False

    def find_all(mentions):
        found = set()
        for mention in mentions:
            if mention.strip():
                found.update(find_project_ids(f"{slots.scope} {mention}".strip(), projects))
        return found

    ids = find_all(slots.include)
    if not ids and slots.scope and slots.intent in {"list", "exclude"}:
        ids.update(find_project_ids(f"{slots.scope} 사업 목록", projects))
    ids -= find_all(slots.exclude)
    return sorted(ids), True


def synthetic():
    projects = [
        ProjectInfo("a25", "가나다기관", "2025년 기록보존 시스템 운영", None, None),
        ProjectInfo("a26", "가나다기관", "2026년 기록보존 시스템 운영", None, None),
        ProjectInfo("b", "가나다기관", "통합 민원 접수창구 구축", None, None),
        ProjectInfo("c", "라마바기관", "통합 민원 접수창구 구축", None, None),
        ProjectInfo("d", "라마바기관", "교육 성과 기록 이관", None, None),
    ]
    rows = [
        (
            "비교",
            "가나다기관 2025년 기록보존과 라마바기관 교육 성과 기록 이관 각각 예산",
            ["a25", "d"],
        ),
        ("비교", "가나다기관 2025년 기록보존이랑 2026년 기록보존 마감일 비교", ["a25", "a26"]),
        (
            "비교",
            "라마바기관 교육 성과 기록 이관 예산을 가나다기관 민원 접수창구와 비교",
            ["b", "d"],
        ),
        ("비교", "가나다기관 2099년 기록보존과 라마바기관 교육 성과 기록 이관 각각", ["d"]),
        ("비교", "라마바기관 민원 접수창구 / 교육 성과 기록 이관 각각", ["c", "d"]),
        ("제외", "가나다기관 2025년 기록보존만 빼고 나머지", ["a26", "b"]),
        ("제외", "라마바기관 민원 접수창구 제외한 사업 합계", ["d"]),
        ("제외 아님", "라마바기관 교육 성과 기록 이관의 부가세 빼고 예산", ["d"]),
        ("기관", "가나다기관 요구사항", ["a25", "a26", "b"]),
        ("누락", "가상조달재단 민원 접수창구 사업 마감", []),
        ("누락", "가나다기관 2099년 기록보존 사업 마감", []),
        ("한 문서", "라마바기관 민원 접수창구", ["c"]),
    ]
    return projects, [(str(i + 1), q, x, kind) for i, (kind, q, x) in enumerate(rows)]


def run(cases, projects, client):
    result = []
    for case_id, question, target, category in cases:
        base = sorted(find_project_ids(question, projects))
        started = time.perf_counter()
        error = None
        slots = None
        try:
            slots = parse(client, question)
            docs, valid = resolve(slots, question, projects)
        except (ValueError, OSError, KeyError) as caught:
            docs, valid, error = [], False, type(caught).__name__
        elapsed = time.perf_counter() - started
        result.append(
            {
                "id": case_id,
                "question": question,
                "category": category,
                "expected": sorted(target),
                "base": base,
                "selected": docs,
                "valid": valid,
                "error": error,
                "seconds": elapsed,
                "slots": None if slots is None else slots.model_dump(),
            }
        )
    return result


def summarize(rows):
    times = sorted(r["seconds"] for r in rows)
    return {
        "n": len(rows),
        "baseline": sum(r["base"] == r["expected"] for r in rows),
        "experimental": sum(r["valid"] and r["selected"] == r["expected"] for r in rows),
        "valid": sum(r["valid"] for r in rows),
        "seconds_p95": round(times[math.ceil(0.95 * len(times)) - 1], 3),
        "new_wrong_document": sum(
            bool(set(r["selected"]) - set(r["expected"]))
            for r in rows
            if r["base"] == r["expected"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    for field in ("catalog", "questions", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    with args.questions.open(encoding="utf-8-sig", newline="") as source:
        team = list(csv.DictReader(source))
    by_id = {row["question_id"]: row for row in team}
    expected = score(team, projects)["rows"]
    comparisons = [
        (r["question_id"], by_id[r["question_id"]]["question"], r["expected_ids"], "복수 사업")
        for r in expected
        if len(r["expected_ids"]) > 1
    ]
    fixture, hand = synthetic()
    client = LocalEmbeddings().client
    results = {}
    for name, records, catalog in (
        ("synthetic", hand, fixture),
        ("team_multi", comparisons, projects),
    ):
        rows = run(records, catalog, client)
        counts = summarize(rows)
        print(name, counts, flush=True)
        results[name] = {"scores": counts, "rows": rows}
    args.output.write_text(
        json.dumps(
            {"experiment": "slot_trial", "seed": 1611, "model": MODEL, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
