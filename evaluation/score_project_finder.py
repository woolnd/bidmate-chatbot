"""Search-B 문서 고르기 정확도를 팀 평가 CSV로 잰다.

실행: uv run python -m evaluation.score_project_finder evaluation/questions.csv
질문별 결과는 Git에서 제외된 evaluation/results/에 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from bidmate.config import PROJECT_LIST_PATH, PROJECT_ROOT
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo

Selector = Callable[[str, list[ProjectInfo]], list[str]]

FOLLOW_UP = "후속질문"
NO_ANSWER = "없는내용"
REQUIRED_COLUMNS = {"question_id", "question_type", "question", "expected_project_names"}


def _name_key(name: str) -> str:
    """CSV와 목록표의 사업명에서 유니코드와 공백 표기를 맞춘다."""
    return " ".join(unicodedata.normalize("NFKC", name).split())


def score(
    questions: list[dict[str, str]],
    projects: list[ProjectInfo] | None = None,
    *,
    selector: Selector = find_project_ids,
) -> dict:
    """팀 질문 행을 받아 완전 일치 점수와 과다 선택·누락 문서 ID를 돌려준다."""
    if projects is None:
        projects = load_project_list(PROJECT_LIST_PATH)
    if not projects:
        raise ValueError("목록표에 문서가 없습니다.")

    ids_by_name: dict[str, set[str]] = defaultdict(set)
    for project in projects:
        ids_by_name[_name_key(project.project_name)].add(project.project_id)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "counted": 0})
    rows = []
    skipped_followups = 0
    seen_ids: set[str] = set()
    for row in questions:
        question_id = (row.get("question_id") or "").strip()
        question_type = (row.get("question_type") or "").strip()
        question = (row.get("question") or "").strip()
        if not question_id or question_id in seen_ids or not question_type or not question:
            raise ValueError("질문 번호, 유형, 질문이 비었거나 질문 번호가 중복됩니다.")
        seen_ids.add(question_id)
        type_key = "".join(question_type.split())
        if type_key == FOLLOW_UP:
            skipped_followups += 1
            continue

        names = (row.get("expected_project_names") or "").split("|")
        expected_ids: set[str] = set()
        for name in filter(None, map(_name_key, names)):
            matching_ids = ids_by_name.get(name, set())
            if len(matching_ids) != 1:
                raise ValueError(
                    f"{question_id}: 정답 사업명을 목록표의 문서 하나로 연결할 수 없습니다."
                )
            expected_ids.update(matching_ids)
        if not expected_ids and type_key != NO_ANSWER:
            raise ValueError(f"{question_id}: 정답 문서가 비어 있습니다.")

        predicted_ids = set(selector(question, projects))
        exact = predicted_ids == expected_ids
        counts[question_type]["counted"] += 1
        counts[question_type]["correct"] += int(exact)
        rows.append(
            {
                "question_id": question_id,
                "expected_ids": sorted(expected_ids),
                "predicted_ids": sorted(predicted_ids),
                "extra_ids": sorted(predicted_ids - expected_ids),
                "missing_ids": sorted(expected_ids - predicted_ids),
                "exact": exact,
            }
        )
    if not rows:
        raise ValueError("채점할 질문이 없습니다.")

    correct = sum(row["exact"] for row in rows)
    return {
        "score": correct / len(rows),
        "correct": correct,
        "counted": len(rows),
        "skipped_followups": skipped_followups,
        "by_type": dict(counts),
        "extra_documents": sum(len(row["extra_ids"]) for row in rows),
        "missing_documents": sum(len(row["missing_ids"]) for row in rows),
        "rows": rows,
    }


def _load_baseline(path: Path) -> Selector:
    """지정한 변경 전 코드 사본에서 문서 선택 함수를 읽는다."""
    spec = importlib.util.spec_from_file_location("search_b_baseline", path)
    if spec is None or spec.loader is None:
        raise ValueError("기준 코드 사본을 읽을 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_project_ids


def _sha256(path: Path) -> str:
    """비교한 파일을 다시 확인할 수 있도록 해시를 돌려준다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """같은 CSV로 현재 코드와 선택한 기준 코드를 채점한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--projects", type=Path, default=PROJECT_LIST_PATH)
    parser.add_argument("--baseline", type=Path, help="비교할 project_finder.py 사본")
    args = parser.parse_args()

    try:
        with args.questions.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
                raise ValueError("평가 CSV의 필수 칸이 빠졌습니다.")
            questions = list(reader)
        projects = load_project_list(args.projects)
        results = {"current": score(questions, projects)}
        if args.baseline:
            results["baseline"] = score(questions, projects, selector=_load_baseline(args.baseline))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    output = PROJECT_ROOT / "evaluation" / "results" / "search_b.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "questions_sha256": _sha256(args.questions),
        "projects_sha256": _sha256(args.projects),
        "current_sha256": _sha256(PROJECT_ROOT / "src/bidmate/retrieval/project_finder.py"),
        "baseline_sha256": _sha256(args.baseline) if args.baseline else None,
    }
    output.write_text(json.dumps({"inputs": metadata, **results}, ensure_ascii=False, indent=2))
    for label, result in results.items():
        print(f"{label}: {result['correct']}/{result['counted']} ({result['score']:.1%})")
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False
            )
        )
    print(f"질문별 결과: {output}")


if __name__ == "__main__":
    main()
