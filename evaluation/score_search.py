"""3단계 검색 적중률(Hit@5)을 채점한다. 담당: 검색 파트(검색-A)

질문: search_chunks()가 찾아온 조각 안에 정답 문서의 조각이 있는가?

    계산 대상   expected_project_names가 있는 질문 (없는 내용 질문은 뺀다)
    맞음        찾아온 조각 TOP_K(5)개 안에 정답 문서 조각이 있다
                여러 문서 질문은 정답 문서마다 조각이 하나씩은 있어야 맞다
    점수        맞은 질문 수 ÷ 계산 대상 질문 수 (0.0 ~ 1.0)

내용 기준 점수도 같이 낸다 (사업명만 맞으면 맞음이라 너그러운 점수를 보완):
    맞음        정답 문서마다, 찾아온 조각 중 그 문서 조각이면서 평가셋 키워드가 든 조각이 있다
    키워드      평가셋 키워드 칸 (answer_keywords, | 로 구분)
                한 문서 질문: 키워드 중 하나라도 들어 있으면 포함
                여러 문서 질문: 정답 사업 순서대로 키워드를 짝지어 확인
                (예: "A사업|B사업" + "5개월|180일" → A 조각에 5개월, B 조각에 180일)
                짝 키워드가 빈 사업은 사업명만 확인한다
    비교        띄어쓰기와 쉼표를 지우고 비교한다 (986,945,000 = 986945000, 작 성 = 작성)
    빼는 질문   후속 질문 (앞 질문이 있어야 뜻이 통해서 1단계 담당. 두 점수 모두에서 뺀다)
                키워드 칸이 빈 질문 (판정 불가)
                정답 문서 조각 어디에도 키워드가 없는 질문 (글자 깨짐·잘림 = 데이터 단계 문제)

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
import re
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bidmate.config import PROJECT_LIST_PATH, PROJECT_ROOT, TOP_K
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.project_finder import select_project_scope
from bidmate.retrieval.vector_store import get_vector_store, search_chunks
from bidmate.schemas import ProjectInfo

RESULT_FOLDER = PROJECT_ROOT / "data" / "evaluation"  # 질문별 결과 (Git에 안 올라간다)

FOLLOW_UP_QUESTION_TYPE = "후속 질문"  # 앞 질문이 있어야 뜻이 통하는 질문. 채점에서 뺀다
# 평가셋 키워드 칸 이름 (앞에 있는 이름부터 찾는다)
KEYWORD_COLUMNS = ("keywords", "answer_keywords", "expected_keywords")


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


def normalize_text(text: str) -> str:
    """띄어쓰기와 쉼표를 지운다. 표기만 다른 같은 말을 같게 보려고 쓴다.

    받는 것: 글 (예: "986,945,000 원", "작 성 방 법")
    넘기는 것: 띄어쓰기·쉼표가 없는 글 (예: "986945000원", "작성방법")
    """
    return re.sub(r"[\s,]", "", text)


def get_keywords(question: dict) -> list[str]:
    """평가셋 한 줄에서 키워드(정답에 꼭 들어가야 하는 말) 목록을 꺼낸다.

    받는 것: 평가셋 한 줄
    넘기는 것: 키워드 목록 (| 로 나눈 것, 빈 칸은 뺀다). 칸이 없거나 비어 있으면 빈 목록
    """
    for column in KEYWORD_COLUMNS:
        keywords = split_names(question.get(column))
        if keywords:
            return keywords
    return []


def match_keywords_to_projects(
    question: dict, expected_project_ids: list[str]
) -> dict[str, list[str]]:
    """정답 문서마다 확인할 키워드를 정한다.

    받는 것: 평가셋 한 줄, 정답 문서 project_id 목록 (평가셋 사업명 순서)
    넘기는 것: {project_id: 키워드 목록}
        - 여러 문서 질문이고 | 로 나눈 칸 수가 문서 수와 같으면 순서대로 짝짓는다
          (짝이 빈 칸이면 그 문서는 빈 목록 = 사업명만 확인)
        - 그 밖에는 모든 문서에 전체 키워드를 쓴다
    """
    keywords = get_keywords(question)
    for column in KEYWORD_COLUMNS:
        raw_text = question.get(column)
        if raw_text and len(expected_project_ids) > 1:
            slots = [slot.strip() for slot in raw_text.split("|")]
            if len(slots) == len(expected_project_ids):
                return {
                    project_id: [slot] if slot else []
                    for project_id, slot in zip(expected_project_ids, slots, strict=True)
                }
        if raw_text:
            break
    return {project_id: keywords for project_id in expected_project_ids}


def keyword_in_document(keywords: list[str], document_texts: list[str]) -> bool:
    """정답 문서의 조각 중 어디에라도 키워드가 하나라도 들어 있는지 확인한다.

    받는 것: 키워드 목록, 정답 문서의 모든 조각 본문
    넘기는 것: 하나라도 있으면 True. 없으면 글자가 깨졌거나 잘린 것이라 검색으로는 못 찾는다
    """
    normalized_keywords = [normalize_text(keyword) for keyword in keywords]
    return any(
        keyword in normalize_text(text)
        for text in document_texts
        for keyword in normalized_keywords
    )


def check_content_hit(
    chunks: list[tuple[Document, float]], keywords_by_project_id: dict[str, list[str]]
) -> int | None:
    """찾아온 조각에 정답 문서마다 키워드가 든 조각이 있는지 확인한다.

    받는 것: search_chunks()가 돌려준 (Document, 점수) 목록,
             {정답 문서 project_id: 키워드 목록}
    넘기는 것: 모든 정답 문서의 정답 조각이 나온 등수 중 가장 늦은 등수 (1부터).
               정답 조각이 없는 문서가 하나라도 있으면 None (틀림)
               키워드 목록이 빈 문서는 그 문서 조각이기만 하면 된다 (사업명만 확인)
    """
    needed_ranks: list[int] = []
    for project_id, keywords in keywords_by_project_id.items():
        normalized_keywords = [normalize_text(keyword) for keyword in keywords]
        first_rank = next(
            (
                rank
                for rank, (document, _) in enumerate(chunks, start=1)
                if document.metadata["project_id"] == project_id
                and (
                    not normalized_keywords
                    or any(
                        keyword in normalize_text(document.page_content)
                        for keyword in normalized_keywords
                    )
                )
            ),
            None,
        )
        if first_rank is None:
            return None
        needed_ranks.append(first_rank)
    return max(needed_ranks, default=None)


def judge_content(
    question: dict,
    chunks: list[tuple[Document, float]],
    expected_project_ids: list[str],
    vector_store: Chroma,
    document_texts_cache: dict[str, list[str]],
) -> tuple[str, int | None, list[str]]:
    """한 질문의 내용 기준 결과를 정한다.

    받는 것: 평가셋 한 줄, 찾아온 조각, 정답 문서 project_id 목록, 저장소,
             문서 조각 본문을 한 번만 읽으려고 기억해 두는 곳
    넘기는 것: (상태, 정답 조각 등수, 키워드 목록)
        상태: "맞음" / "틀림" / "키워드 없음" / "조각에 정답 없음"
    """
    keywords = get_keywords(question)
    if len(keywords) == 0 or len(expected_project_ids) == 0:
        return "키워드 없음", None, keywords

    # 정답 문서마다 확인할 키워드를 정하고 (여러 문서면 순서대로 짝),
    # 그 키워드가 문서 조각 어딘가에 있는지 먼저 본다 (없으면 검색 탓이 아니다)
    keywords_by_project_id = match_keywords_to_projects(question, expected_project_ids)
    for project_id, project_keywords in keywords_by_project_id.items():
        if not project_keywords:
            continue
        if project_id not in document_texts_cache:
            document_texts_cache[project_id] = vector_store.get(
                where={"project_id": project_id}, include=["documents"]
            )["documents"]
        if not keyword_in_document(project_keywords, document_texts_cache[project_id]):
            return "조각에 정답 없음", None, keywords

    rank = check_content_hit(chunks, keywords_by_project_id)
    return ("맞음" if rank is not None else "틀림"), rank, keywords


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
        "skipped_follow_up": 뺀 후속 질문 수,
        "content_score": 내용 기준 맞은 질문 수 ÷ 내용 판정한 질문 수,
        "content_counted": 내용 판정한 질문 수,
        "content_wrong_question_ids": 내용 기준 틀린 질문 번호,
        "content_unjudged_question_ids": 판정 불가 (키워드 칸이 빈 질문),
        "answer_missing_question_ids": 정답 문서 조각에 정답 글자가 없는 질문 (데이터 단계 확인),
        "rows": 질문별 결과 (틀린 이유를 볼 때 쓴다),
    }
    """
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")

    # 1. 준비: 저장소와 목록표 (테스트에서는 가짜가 들어온다)
    if vector_store is None:
        vector_store = get_vector_store()
    if projects is None:
        projects = load_project_list(PROJECT_LIST_PATH)
    project_id_by_name = {project.project_name: project.project_id for project in projects}

    rows: list[dict] = []
    unknown_project_names: set[str] = set()
    skipped_follow_up = 0
    document_texts_cache: dict[str, list[str]] = {}

    for question in questions:
        expected_project_names = split_names(question.get("expected_project_names"))

        # 2. 정답 문서가 없는 질문(없는 내용)은 채점하지 않는다
        if len(expected_project_names) == 0:
            continue

        # 2-1. 후속 질문은 앞 질문이 있어야 뜻이 통해서(1단계 담당) 채점하지 않는다
        if question.get("question_type") == FOLLOW_UP_QUESTION_TYPE:
            skipped_follow_up += 1
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
            catalog_miss = False
        else:
            project_ids, catalog_miss = select_project_scope(question["question"], projects)

        # 5. 찾기 + 걸린 시간 재기
        start_time = time.perf_counter()
        chunks = (
            []
            if catalog_miss
            else search_chunks(question["question"], project_ids, k, vector_store)
        )
        search_seconds = time.perf_counter() - start_time

        # 6. 내용 기준 판정 (정답 문서 안에 정답 글자가 든 조각을 찾아왔나)
        expected_project_ids = [
            project_id_by_name[name]
            for name in expected_project_names
            if name in project_id_by_name
        ]
        content_status, content_rank, keywords = judge_content(
            question,
            chunks,
            expected_project_ids,
            vector_store,
            document_texts_cache,
        )

        # 7. 맞았는지 채점하고 기록하기
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
                "keywords": "|".join(keywords),
                "content_status": content_status,
                "content_rank": content_rank if content_rank is not None else "",
                "search_seconds": round(search_seconds, 4),
            }
        )

    # 8. 점수 계산 (계산 대상이 0개면 0점)
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
        "skipped_follow_up": skipped_follow_up,
        **summarize_content(rows),
        "rows": rows,
    }


def summarize_content(rows: list[dict]) -> dict:
    """질문별 내용 판정 결과로 내용 기준 점수를 계산한다.

    받는 것: score()가 만든 질문별 결과
    넘기는 것: content_score, content_counted, content_wrong_question_ids,
               content_unjudged_question_ids, answer_missing_question_ids
    """
    judged_rows = [row for row in rows if row["content_status"] in ("맞음", "틀림")]
    content_hit_count = sum(row["content_status"] == "맞음" for row in judged_rows)
    return {
        "content_score": content_hit_count / len(judged_rows) if judged_rows else 0.0,
        "content_counted": len(judged_rows),
        "content_wrong_question_ids": [
            row["question_id"] for row in judged_rows if row["content_status"] == "틀림"
        ],
        "content_unjudged_question_ids": [
            row["question_id"] for row in rows if row["content_status"] == "키워드 없음"
        ],
        "answer_missing_question_ids": [
            row["question_id"] for row in rows if row["content_status"] == "조각에 정답 없음"
        ],
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
    print(
        f"  Hit@{TOP_K} (내용 기준): {result['content_score']:.1%}  "
        f"({result['content_counted']}개 중 틀림 {len(result['content_wrong_question_ids'])}개)"
    )
    print(f"  내용 틀린 질문: {', '.join(result['content_wrong_question_ids']) or '없음'}")
    print(
        f"  내용 판정에서 뺀 질문: 키워드 없음 {len(result['content_unjudged_question_ids'])}개, "
        f"조각에 정답 없음 {len(result['answer_missing_question_ids'])}개, "
        f"후속 질문 {result['skipped_follow_up']}개"
    )


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
