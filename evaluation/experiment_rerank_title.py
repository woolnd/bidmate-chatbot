"""리랭커 제목 실험: 조각 앞의 [기관 / 사업명] 제목을 빼면 리랭커가 더 잘 고르는지 본다.

담당: 검색 파트(검색-A). 실험용이라 search_chunks와 앱이 쓰는 저장소(data/chroma)는 바꾸지 않는다.

왜 하나:
    Q033에서 리랭커가 정답(CCTV 홍수 감시 조각) 대신 테스트·목차·보안 조각을 골랐다.
    모든 조각 앞에 붙는 제목("[케빈랩 / 평택시 ... 영상감시 ... 홍수감시 ...]")에
    끌려간 것으로 보인다.
    2단계가 이미 기관명·사업명으로 문서를 골라 주니,
    3단계에서는 제목이 순서를 정하는 데 도움이 안 될 수 있다.

비교하는 조건 (검색만 = 정답 문서 넣고, 내용 기준, 후보 20개 → 리랭커 → 5개):
    1. 지금           임베딩 검색: 제목 + 내용 / 리랭커: 제목 + 내용
    2. 리랭커 내용만  임베딩 검색: 제목 + 내용 / 리랭커: 내용만
    3. 둘 다 내용만   임베딩 검색: 내용만      / 리랭커: 내용만
       (3번은 제목을 뺀 조각으로 따로 만든 실험용 저장소 data/chroma_notitle 을 쓴다)
    여러 문서 질문은 문서마다 후보를 찾고, 나눠 검색과 같은 개수(split_k)를 고른다

실행 (레포 맨 위 폴더에서, 리랭커가 들어가서 GPU 서버에서 돌린다):
    1) 실험용 저장소 만들기 (처음 한 번, 조각 전체 임베딩 약 $0.17):
       uv run python -m evaluation.experiment_rerank_title --build
    2) 3가지 조건 비교:
       uv run python -m evaluation.experiment_rerank_title
    - 리랭커 패키지가 있어야 한다 (서버는 uv sync --extra rerank 가 돼 있다)
    - Mac에서 몇 개만 빠르게 보려면: --only Q033,Q001 (Mac 리랭커는 질문당 약 13초)

결과 저장 (data/evaluation/, Git에 안 올라간다):
    experiment_rerank_title_summary.csv   조건마다 점수 한 줄
    experiment_rerank_title_rows.csv      조건 × 질문마다 맞음/틀림 한 줄
"""

import csv
import sys
from datetime import datetime

from langchain_core.documents import Document

import evaluation.score_search as score_search
from bidmate.config import PROJECT_LIST_PATH, PROJECT_ROOT, RERANK_CANDIDATE_K
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.vector_store import get_vector_store, save_chunks, split_k

NOTITLE_FOLDER = (
    PROJECT_ROOT / "data" / "chroma_nottitle"
)  # 제목 뺀 실험용 저장소 (Git에 안 올라간다)
WATCH_QUESTION_IDS = ("Q001", "Q012", "Q033")  # 따로 지캬볼 질문
RESULT_FOLDER = score_search.RESULT_FOLDER  # data/evaluation


def strip_title(document: Document) -> str:
    """조각 본문에서 맨 앞의 "[기관 / 사업명]" 제목 줄을 뺀 내용만 돌려준다.

    제목은 조각 자르기에서 "[기관 / 사업명]" 모양으로 붙인다 (chunker.py).
    사업명 안에 [재공고] 같은 대괄호가 있을 수 있어서,
    글자 모양 대신 이름표(metadata)로 제목을 만든다.
    제목을 빼고 나면 아무 글자도 없는 조각은 원래 본문을 그대로 쓴다 (빈 글은 저장할 수 없어서).
    """
    title = f"[{document.metadata['agency_name']} / {document.metadata['project_name']}]"
    text = document.page_content
    if text.startswith(title):
        content = text[len(title) :].lstrip()
        return content if content else text
    return text


def build_notitle_store() -> None:
    """앱이 쓰는 저장소의 조각을 모두 꺼내 제목을 빼고 실험용 저장소에 다시 저장한다.

    조각 전체를 다시 임베딩해서 비용이 든다 (약 $0.17).
    """
    saved = get_vector_store().get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(saved["documents"], saved["metadatas"], strict=True)
    ]
    notitle_chunks = [
        Document(page_content=strip_title(chunk), metadata=chunk.metadata) for chunk in chunks
    ]
    notitle_store = get_vector_store(persist_folder=NOTITLE_FOLDER)
    notitle_store.reset_collection()  # 예전 실험용 조각과 섞이지 않게 비우고 새로 넣는다
    print(
        f"제목 뺀 조각 {len(notitle_chunks)}개를 {NOTITLE_FOLDER}에 저장합니다 (임베딩 약 $0.17)..."
    )
    save_chunks(notitle_chunks, notitle_store)
    print("저장 완료")


def make_rerank_search(candidate_store, reranker, reranker_sees_title: bool):
    """후보를 찾고 리랭커로 최종 k개를 고르는 검색 함수를 만든다 (채점 코드에 바꿔 끼우려고).

    candidate_store: 후보를 찾을 저장소 (앱 저장소 = 제목 포함, 실험용 저장소 = 내용만)
    reranker_sees_title: True면 리랭커에 조각 본문 그대로, False면 제목을 뺀 내용만 넣는다
    여러 문서 질문은 문서마다 후보를 찾고, 나눠 검색과 같은 개수(split_k)만큼 고른다.
    """

    def search(question, project_ids, k, vector_store=None):
        results: list[tuple[Document, float]] = []
        for project_id, keep in zip(project_ids, split_k(k, len(project_ids)), strict=True):
            if keep == 0:
                continue
            candidates = candidate_store.similarity_search_with_score(
                question, k=RERANK_CANDIDATE_K, filter={"project_id": project_id}
            )
            if not candidates:
                continue
            texts = [
                document.page_content if reranker_sees_title else strip_title(document)
                for document, _ in candidates
            ]
            scores = reranker.predict([(question, text) for text in texts])
            ranked = sorted(
                zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
            )
            results.extend(candidate for candidate, _ in ranked[:keep])
        return results

    return search


def save_csv(path, rows: list[dict]) -> None:
    """결과를 csv로 저장한다. data/ 아래라 Git에 안 올라간다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_only_question_ids() -> set[str] | None:
    """--only Q033,Q001 처럼 적으면 그 질문만 채점한다 (Mac에서 빠르게 볼 때)."""
    for index, argument in enumerate(sys.argv):
        if argument == "--only" and index + 1 < len(sys.argv):
            return {question_id.strip() for question_id in sys.argv[index + 1].split(",")}
    return None


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_notitle_store()
        sys.exit(0)

    from bidmate.retrieval.reranker import get_reranker

    main_store = get_vector_store()
    notitle_store = get_vector_store(persist_folder=NOTITLE_FOLDER)
    if notitle_store._collection.count() == 0:
        sys.exit("실험용 저장소가 비어 있습니다. 먼저 --build 로 만들어 주세요.")

    questions = score_search.load_questions("evaluation/questions.csv")
    only_question_ids = read_only_question_ids()
    if only_question_ids:
        questions = [q for q in questions if q["question_id"] in only_question_ids]
    projects = load_project_list(PROJECT_LIST_PATH)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("리랭커 모델 불러오는 중...")
    reranker = get_reranker()

    conditions = [
        ("1. 지금", main_store, True),
        ("2. 리랭커 내용만", main_store, False),
        ("3. 둘 다 내용만", notitle_store, False),
    ]

    summary_rows: list[dict] = []
    question_rows: list[dict] = []
    for name, candidate_store, reranker_sees_title in conditions:
        score_search.search_chunks = make_rerank_search(
            candidate_store, reranker, reranker_sees_title
        )
        result = score_search.score(questions, main_store, projects, use_expected_projects=True)
        status = {row["question_id"]: row["content_status"] for row in result["rows"]}
        summary_rows.append(
            {
                "실행 시각": run_time,
                "조건": name,
                "내용 기준": f"{result['content_score']:.1%}",
                "채점 질문 수": result["content_counted"],
                **{question_id: status.get(question_id, "") for question_id in WATCH_QUESTION_IDS},
                "틀린 질문": " ".join(result["content_wrong_question_ids"]),
                "검색 시간 평균(초)": round(result["average_search_seconds"], 3),
            }
        )
        question_rows.extend({"실행 시각": run_time, "조건": name, **row} for row in result["rows"])

        print("\n[리랭커 제목 실험] 검색만(정답 문서 넣고), 내용 기준, 후보 20개 → 리랭커 → 5개\n")
    for row in summary_rows:
        watch = "  ".join(
            f"{question_id} {row[question_id] or '-'}" for question_id in WATCH_QUESTION_IDS
        )
        print(f"  {row['조건']:<14} {row['내용 기준']:>6} ({row['채점 질문 수']}개)  {watch}")
        print(f"  {'':<14} 틀림: {row['틀린 질문'] or '없음'}")
        print(f"  {'':<14} 검색 시간 평균 {row['검색 시간 평균(초)']}초\n")

    save_csv(RESULT_FOLDER / "experiment_rerank_title_summary.csv", summary_rows)
    save_csv(RESULT_FOLDER / "experiment_rerank_title_rows.csv", question_rows)
    print(f"결과 저장: {RESULT_FOLDER}/experiment_rerank_title_summary.csv, ..._rows.csv")
