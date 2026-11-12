"""3단계 검색 적중률(Hit@5)을 채점한다. 담당: 검색 파트(검색-A)

질문: search_chunks()가 찾아온 조각 안에 정답 문서의 조각이 있는가?

    계산 대상   expected_project_names가 있는 질문 (없는 내용 질문은 뺀다)
    맞음        찾아온 조각 TOP_K(5)개 안에 정답 문서 조각이 있다
                여러 문서 질문은 정답 문서마다 조각이 하나씩은 있어야 맞다
    점수        맞은 질문 수 ÷ 계산 대상 질문 수 (0.0 ~ 1.0)

두 가지로 잰다:
    실제 흐름   2단계 find_project_ids()가 고른 문서 안에서 찾는다 (2단계 + 3단계 실력)
    검색만      정답 문서 목록을 그대로 넣고 찾는다 (2단계 영향을 뺀 3단계만의 실력)

혼자 돌려 보기 (레포 맨 위 폴더에서):
    uv run python evaluation/score_search.py evaluation/questions.csv
    - Chroma에 조각이 저장돼 있어야 한다 (scripts/build_index.py를 먼저 돌린다)
    - 질문마다 질문 임베딩을 한 번 만들어서 아주 적은 비용이 든다 (질문 25개 × 2번)
    - 질문별 결과는 data/evaluation/ 에 저장한다 (평가셋 내용이 들어 있어서 Git에 안 올린다)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bidmate.config import PROJECT_LIST_PATH, PROJECT_ROOT, TOP_K
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.project_finder import find_project_ids
from bidmate.retrieval.vector_store import get_vector_store, search_chunks
from bidmate.schemas import ProjectInfo

RESULT_FOLDER = PROJECT_ROOT / "data" / "evaluation"  # 질문별 결과 (Git에 안 올라간다)


def split_names(text: str | None) -> list[str]:
    """ "사업A|사업B" 모양의 칸을 사업명 목록으로 나눈다. 비어 있으면 빈 목록.

    받는 것: 평가셋의 expected_project_names 칸 (| 로 구분)
    넘기는 것: 사업명 목록 (앞뒤 공백만 지운다)
    """
    if not text:
        return []
    return [name.strip() for name in text.split("|") if name.strip()]


def check_search_hit(
    chunks: list[tuple[Document, float]], expected_project_names: list[str]
) -> bool:
    """찾아온 조각이 정답 문서를 모두 덮는지 확인한다.

    받는 것: search_chunks()가 돌려준 (Document, 점수) 목록, 정답 사업명 목록
    넘기는 것: 정답 문서마다 조각이 하나 이상 있으면 True, 하나라도 없으면 False
               (정답 사업명이 비어 있으면 채점할 수 없어서 False)
    """
    if len(expected_project_names) == 0:
        return False
    found_project_names = {document.metadata["project_name"] for document, _ in chunks}
    return all(name in found_project_names for name in expected_project_names)


def load_questions(csv_path: Path) -> list[dict]:
    """평가셋 csv를 읽어 한 줄씩 사전으로 돌려준다.

    받는 것: 평가셋 경로 (예: evaluation/questions.csv)
    넘기는 것: [{"question_id": "Q001", "question": "...", ...}, ...]
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def score(
    questions: list[dict],
    vector_store: Chroma | None = None,
    projects: list[ProjectInfo] | None = None,
    use_expected_projects: bool = False,
    k: int = TOP_K,
) -> dict:
    """검색 적중률을 채점한다. 모든 채점 파일이 지키는 약속: score(questions) -> dict

    받는 것: questions (평가셋 csv를 읽은 줄 목록),
             vector_store (테스트에서 가짜 임베딩 저장소를 넣는다. 비우면 get_vector_store()),
             projects (목록표. 비우면 data_list.csv를 읽는다),
             use_expected_projects (True면 정답 문서 목록을 그대로 넣어 "검색만" 잰다),
             k (찾아올 조각 수, 기본 config.TOP_K)
    넘기는 것: {
        "score": 맞은 질문 수 ÷ 계산 대상 질문 수 (0.0~1.0),
        "counted": 계산 대상 질문 수,
        "wrong_question_ids": 틀린 질문 번호 목록,
        "average_search_seconds": 질문 1개 검색에 걸린 평균 시간(초),
        "slowest_search_seconds": 가장 오래 걸린 검색 시간(초),
        "unknown_project_names": 목록표에 없는 정답 사업명 (평가셋 오타 의심),
        "rows": 질문별 결과 (틀린 이유를 볼 때 쓴다),
    }
    """
    # 1. 준비: 저장소와 목록표 (테스트에서는 가짜가 들어온다)
    if vector_store is None:
        vector_store = get_vector_store()
    if projects is None:
        projects = load_project_list(PROJECT_LIST_PATH)
    project_id_by_name = {project.project_name: project.project_id for project in projects}

    rows: list[dict] = []
    unknown_project_names: set[str] = set()

    for question in questions:
        expected_project_names = split_names(question.get("expected_project_names"))

        # 2. 정답 문서가 없는 질문(없는 내용)은 채점하지 않는다
        if len(expected_project_names) == 0:
            continue

        # 3. 목록표에 없는 정답 사업명 모으기. 오타면 검색이 맞아도 틀림으로 나온다
        unknown_project_names.update(
            name for name in expected_project_names if name not in project_id_by_name
        )

        # 4. 어느 문서 안에서 찾을지 정하기
        if use_expected_projects:
            project_ids = [
                project_id_by_name[name]
                for name in expected_project_names
                if name in project_id_by_name
            ]
        else:
            project_ids = find_project_ids(question["question"], projects)

        # 5. 찾기 + 걸린 시간 재기
        start_time = time.perf_counter()
        chunks = search_chunks(question["question"], project_ids, k, vector_store)
        search_seconds = time.perf_counter() - start_time

        # 6. 맞았는지 채점하고 기록하기
        rows.append(
            {
                "question_id": question["question_id"],
                "question_type": question.get("question_type", ""),
                "expected_project_names": "|".join(expected_project_names),
                "searched_project_ids": "|".join(project_ids),
                "found_project_names": "|".join(
                    document.metadata["project_name"] for document, _ in chunks
                ),
                "hit": check_search_hit(chunks, expected_project_names),
                "search_seconds": round(search_seconds, 4),
            }
        )

    # 7. 점수 계산 (계산 대상이 0개면 0점)
    counted = len(rows)
    hit_count = sum(row["hit"] for row in rows)
    search_times = [row["search_seconds"] for row in rows]
    return {
        "score": hit_count / counted if counted else 0.0,
        "counted": counted,
        "wrong_question_ids": [row["question_id"] for row in rows if not row["hit"]],
        "average_search_seconds": sum(search_times) / counted if counted else 0.0,
        "slowest_search_seconds": max(search_times, default=0.0),
        "unknown_project_names": sorted(unknown_project_names),
        "rows": rows,
    }


def save_rows(rows: list[dict], result_path: Path) -> None:
    """질문별 결과를 csv로 저장한다. data/ 아래라 Git에 안 올라간다."""
    if len(rows) == 0:
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(title: str, result: dict) -> None:
    """채점 결과를 사람이 읽기 좋게 출력한다."""
    print(f"\n[{title}]")
    print(
        f"  Hit@{TOP_K}: {result['score']:.1%}  ({result['counted']}개 중 틀림 "
        f"{len(result['wrong_question_ids'])}개)"
    )
    print(
        f"  검색 시간: 평균 {result['average_search_seconds']:.3f}초, "
        f"가장 느림 {result['slowest_search_seconds']:.3f}초"
    )
    print(f"  틀린 질문: {', '.join(result['wrong_question_ids']) or '없음'}")


if __name__ == "__main__":
    questions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation/questions.csv")
    questions = load_questions(questions_path)
    vector_store = get_vector_store()

    # 저장소가 비어 있으면 모든 질문이 틀림으로 나오니 먼저 멈춘다
    if vector_store._collection.count() == 0:
        sys.exit("저장된 조각이 0개입니다. scripts/build_index.py를 먼저 돌려 주세요.")

    projects = load_project_list(PROJECT_LIST_PATH)
    pipeline_result = score(questions, vector_store, projects, use_expected_projects=False)
    search_only_result = score(questions, vector_store, projects, use_expected_projects=True)

    if pipeline_result["unknown_project_names"]:
        print("[경고] 목록표에 없는 정답 사업명 (평가셋 오타 의심):")
        for name in pipeline_result["unknown_project_names"]:
            print("  -", name)

    print_result("실제 흐름: 2단계가 고른 문서 안에서", pipeline_result)
    print_result("검색만: 정답 문서를 그대로 넣고", search_only_result)

    save_rows(pipeline_result["rows"], RESULT_FOLDER / "score_search_pipeline.csv")
    save_rows(search_only_result["rows"], RESULT_FOLDER / "score_search_only.csv")
    print(f"\n질문별 결과: {RESULT_FOLDER}")
