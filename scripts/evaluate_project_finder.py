"""실제 data_list.csv를 이용한 Search-B 진단 스크립트."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR.parent / "data_list.csv"


def parse_budget(value: str | None) -> int | None:
    """CSV 예산을 정수로 변환한다."""
    if not value or not value.strip():
        return None

    try:
        budget = int(float(value.replace(",", "").strip()))
        return budget if budget > 0 else None
    except (ValueError, TypeError):
        return None


def load_projects(csv_path: Path) -> list[ProjectInfo]:
    """CSV를 ProjectInfo 목록으로 변환한다."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {csv_path}"
        )

    projects: list[ProjectInfo] = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "파일명",
            "발주 기관",
            "사업명",
            "사업 금액",
            "입찰 참여 마감일",
        }

        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"CSV 필수 컬럼이 없습니다: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            project_id = (row.get("파일명") or "").strip()
            agency_name = (row.get("발주 기관") or "").strip()
            project_name = (row.get("사업명") or "").strip()

            if not project_id or not agency_name or not project_name:
                print(
                    f"[경고] {row_number}행 필수값 누락: "
                    f"project_id={project_id!r}"
                )
                continue

            projects.append(
                ProjectInfo(
                    project_id=project_id,
                    agency_name=agency_name,
                    project_name=project_name,
                    budget=parse_budget(row.get("사업 금액")),
                    deadline=(
                        (row.get("입찰 참여 마감일") or "").strip()
                        or None
                    ),
                )
            )

    return projects


def evaluate_self_search(
    projects: list[ProjectInfo],
) -> tuple[int, int, int, list[dict[str, object]]]:
    """각 문서의 기관명과 사업명으로 자기 문서 검색을 검사한다."""

    success_count = 0
    miss_count = 0
    multiple_count = 0
    problems: list[dict[str, object]] = []

    for project in projects:
        question = f"{project.agency_name} {project.project_name}"
        results = find_project_ids(question, projects)

        if project.project_id not in results:
            miss_count += 1
            problems.append(
                {
                    "type": "MISS",
                    "question": question,
                    "expected": project.project_id,
                    "results": results,
                }
            )
            continue

        success_count += 1

        if len(results) > 1:
            multiple_count += 1
            problems.append(
                {
                    "type": "MULTIPLE",
                    "question": question,
                    "expected": project.project_id,
                    "results": results,
                }
            )

    return success_count, miss_count, multiple_count, problems


def main() -> int:
    print("=" * 72)
    print("Search-B 실제 CSV 진단")
    print("=" * 72)
    print(f"CSV 경로: {CSV_PATH}")

    try:
        projects = load_projects(CSV_PATH)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\n[오류] {error}")
        return 1

    project_ids = [project.project_id for project in projects]

    duplicate_count = len(project_ids) - len(set(project_ids))

    print("\n[1] 데이터 무결성")
    print(f"전체 프로젝트: {len(projects)}")
    print(f"고유 project_id: {len(set(project_ids))}")
    print(f"중복 project_id: {duplicate_count}")

    if not projects:
        print("[오류] 검사할 프로젝트가 없습니다.")
        return 1

    print("\n[2] 자기 문서 검색 검사")
    success, misses, multiples, problems = evaluate_self_search(
        projects
    )

    print(f"자기 문서 검색 성공: {success}")
    print(f"자기 문서 검색 실패: {misses}")
    print(f"복수 문서 검색: {multiples}")

    total = len(projects)
    accuracy = success / total * 100

    print(f"자기 문서 검색 성공률: {accuracy:.2f}%")

    print("\n[3] 오탐 및 미탐 상세")

    if not problems:
        print("문제 사례가 없습니다.")
    else:
        for index, problem in enumerate(problems, start=1):
            print("\n" + "-" * 72)
            print(f"사례 {index}: {problem['type']}")
            print(f"질문: {problem['question']}")
            print(f"기대 문서: {problem['expected']}")

            results = problem["results"]
            print(f"검색 결과 수: {len(results)}")

            for result in results:
                marker = (
                    "  [기대 문서]"
                    if result == problem["expected"]
                    else "  [추가 후보]"
                )
                print(f"- {result}{marker}")

    print("\n[4] 최종 요약")
    print(f"전체 데이터: {total}건")
    print(f"검색 성공: {success}건")
    print(f"검색 실패: {misses}건")
    print(f"복수 후보: {multiples}건")
    print(f"중복 ID: {duplicate_count}건")

    print("\n진단 완료")

    return 0


if __name__ == "__main__":
    sys.exit(main())
