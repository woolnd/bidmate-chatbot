"""BM25(키워드 검색) 하이브리드 실험: 임베딩이 놓친 정답 조각을 글자 검색으로 건질 수 있는지 본다.

담당: 검색 파트(검색-A). 실험용이라 search_chunks 코드는 바꾸지 않는다.

왜 하나:
    리랭커는 후보 안에서만 고를 수 있다. Q007, Q011은 정답 조각이 임베딩 후보 20개 안에도 없어서
    리랭커로도 못 고쳤다 (답이 다른 주제 조각 안에 작게 섞여 있어서 뜻으로는 멀게 계산됨).
    BM25는 질문 단어가 "글자 그대로" 많이 든 조각에 높은 점수를 줘서, 이런 조각을 건질 수 있다.
    임베딩 후보와 BM25 후보를 합쳐 쓰는 것을 하이브리드 검색이라고 한다.

무엇을 비교하나 (검색만 = 정답 문서 넣고, 내용 기준):
    단어 자르기 방식 TOKENIZERS × BM25 후보 수 BM25_CANDIDATE_KS 를 모두 돌려 본다.
    1. 후보 안에 정답 조각이 있는 비율 (= 리랭커가 낼 수 있는 최고 점수, 리랭커 없이 재서 빠르다)
    2. --rerank: 리랭커로 최종 5개를 골랐을 때 점수 (= 실제 앱 점수, GPU가 있는 VM에서)
    여러 문서 질문은 문서마다 따로 후보를 찾고, 나눠 검색과 같은 개수(split_k)를 고른다

단어 자르기 방식 (한국어는 "하자보수는"처럼 조사가 붙어서 그대로 세면 다른 단어가 된다):
    뜻 품사      형태소 분석기(kiwipiepy)로 자르고 명사·숫자·영어·어근·동사·형용사만 남긴다
    명사만       형태소 분석기로 자르고 명사·숫자·영어만 남긴다 (동사·형용사는 버린다)
    두 글자씩    띄어쓰기로 나눈 뒤 두 글자씩 겹쳐 자른다 ("하자보수" → 하자, 자보, 보수)

--inspect: Q007, Q011이 왜 못 찾히는지 들여다본다
    질문을 어떻게 잘랐는지, 정답 조각이 임베딩·BM25에서 각각 몇 등인지, BM25 1~3등은 무슨 조각인지

결과 저장 (data/evaluation/, Git에 안 올라간다):
    experiment_bm25_summary.csv   설정마다 점수 한 줄
    experiment_bm25_rows.csv      설정 × 질문마다 맞음/틀림 한 줄

1차 결과 (11/18, Mac, 뜻 품사, 후보 안에 정답 조각이 있는 비율):
    임베딩 20개 94.3% → 임베딩 20개 + BM25 5개 97.1% (Q007이 후보에 들어옴, Q011은 여전히 없음)

실행 (레포 맨 위 폴더에서):
    uv run --with kiwipiepy --with rank-bm25 python -m evaluation.experiment_bm25
    - 끝에 --inspect를 붙이면 Q007, Q011 들여다보기까지 한다
    - 끝에 --rerank를 붙이면 리랭커 점수까지 잰다 (VM에서 uv sync --extra rerank를 한 번 한 뒤)
    - BM25와 리랭커는 내 컴퓨터에서 계산해서 비용 0원. 질문 임베딩만 아주 적은 비용이 든다
"""

import csv
import sys
from collections.abc import Callable
from datetime import datetime

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import evaluation.score_search as score_search
from bidmate.config import PROJECT_LIST_PATH, RERANK_CANDIDATE_K
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.vector_store import get_vector_store, split_k

BM25_CANDIDATE_KS = (5, 10)  # 더해 볼 BM25 후보 수 (문서마다)
WATCH_QUESTION_IDS = ("Q007", "Q011")  # 임베딩 후보 20개 안에도 없던 질문
RESULT_FOLDER = score_search.RESULT_FOLDER  # data/evaluation (Git에 안 올라간다)

kiwi = Kiwi()


def tokenize_with_tags(text: str, keep_tag_prefixes: tuple[str, ...]) -> list[str]:
    """형태소 분석기로 자르고, 품사 이름이 keep_tag_prefixes로 시작하는 조각만 남긴다."""
    return [token.form for token in kiwi.tokenize(text) if token.tag.startswith(keep_tag_prefixes)]


def tokenize_char_bigrams(text: str) -> list[str]:
    """띄어쓰기로 나눈 뒤 두 글자씩 겹쳐 자른다. 예: "하자보수 기간" → 하자, 자보, 보수, 기간"""
    tokens: list[str] = []
    for word in text.split():
        word = "".join(character for character in word if character.isalnum())
        if len(word) == 1:
            tokens.append(word)
        tokens.extend(word[index : index + 2] for index in range(len(word) - 1))
    return tokens


# 단어 자르기 방식: 이름 → 자르는 함수
# 품사 이름: 명사 N..., 숫자 SN, 영어 SL, 한자 SH, 어근 XR, 동사 VV, 형용사 VA
TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "뜻 품사": lambda text: tokenize_with_tags(text, ("N", "SN", "SL", "SH", "XR", "VV", "VA")),
    "명사만": lambda text: tokenize_with_tags(text, ("N", "SN", "SL", "SH")),
    "두 글자씩": tokenize_char_bigrams,
}


class ProjectBM25:
    """문서 하나의 조각들로 만든 BM25 검색기. (자르기 방식, 문서)마다 한 번만 만들고 기억해 둔다."""

    def __init__(self, documents: list[Document], tokenize: Callable[[str], list[str]]) -> None:
        self.documents = documents
        self.tokenize = tokenize
        self.bm25 = BM25Okapi([tokenize(document.page_content) for document in documents])

    def rank_all(self, question: str) -> list[tuple[Document, float]]:
        """모든 조각을 BM25 점수(클수록 관련 있음)가 큰 순서로 돌려준다."""
        scores = self.bm25.get_scores(self.tokenize(question))
        return sorted(
            zip(self.documents, (float(score) for score in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )

    def search(self, question: str, k: int) -> list[tuple[Document, float]]:
        """질문 단어가 많이 든 조각 k개 (점수가 0인 조각은 뺀다)."""
        return [(document, score) for document, score in self.rank_all(question)[:k] if score > 0]


bm25_cache: dict[tuple[str, str], ProjectBM25] = {}


def get_project_bm25(vector_store, project_id: str, tokenizer_name: str) -> ProjectBM25:
    """문서 하나의 BM25 검색기를 가져온다. 처음이면 Chroma에서 그 문서 조각을 모두 읽어 만든다."""
    key = (tokenizer_name, project_id)
    if key not in bm25_cache:
        saved = vector_store.get(
            where={"project_id": project_id}, include=["documents", "metadatas"]
        )
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(saved["documents"], saved["metadatas"], strict=True)
        ]
        bm25_cache[key] = ProjectBM25(documents, TOKENIZERS[tokenizer_name])
    return bm25_cache[key]


def find_candidates(question, project_id, tokenizer_name, bm25_k, vector_store):
    """문서 하나에서 후보를 찾는다: 임베딩 후보 RERANK_CANDIDATE_K개 + BM25 후보 bm25_k개.

    bm25_k가 0이면 임베딩 후보만. 같은 조각이 두 곳에서 나오면 한 번만 넣는다.
    search_chunks 대신 Chroma를 바로 불러서, .env에 USE_RERANKER가 켜져 있어도 영향을 안 받는다.
    """
    found = vector_store.similarity_search_with_score(
        question, k=RERANK_CANDIDATE_K, filter={"project_id": project_id}
    )
    if bm25_k > 0:
        bm25 = get_project_bm25(vector_store, project_id, tokenizer_name)
        found += bm25.search(question, bm25_k)

    candidates: list[tuple[Document, float]] = []
    seen_chunk_ids: set[str] = set()
    for document, score in found:
        if document.metadata["chunk_id"] not in seen_chunk_ids:
            seen_chunk_ids.add(document.metadata["chunk_id"])
            candidates.append((document, score))
    return candidates


def make_search(tokenizer_name: str, bm25_k: int, reranker=None):
    """채점 코드(score_search)의 검색 함수 자리에 끼울 함수를 만든다.

    reranker가 없으면 후보를 전부 돌려준다 (후보 안에 정답이 있는지 재려고).
    reranker가 있으면 문서마다 나눠 검색과 같은 개수(split_k)만큼 리랭커로 고른다
    (실제 앱과 같은 방식).
    """
    from bidmate.retrieval.reranker import rerank

    def search(question, project_ids, k, vector_store=None):
        results: list[tuple[Document, float]] = []
        for project_id, keep in zip(project_ids, split_k(k, len(project_ids)), strict=True):
            candidates = find_candidates(question, project_id, tokenizer_name, bm25_k, vector_store)
            results.extend(
                candidates if reranker is None else rerank(question, candidates, keep, reranker)
            )
        return results

    return search


def prebuild_bm25(questions, projects, vector_store, tokenizer_name: str) -> None:
    """채점 전에 필요한 문서의 BM25 검색기를 미리 만든다 (검색 시간에 만드는 시간이 섞이지 않게)."""
    project_id_by_name = {project.project_name: project.project_id for project in projects}
    for question in questions:
        for name in score_search.split_names(question.get("expected_project_names")):
            if name in project_id_by_name:
                get_project_bm25(vector_store, project_id_by_name[name], tokenizer_name)


def inspect_question(question, projects, vector_store) -> None:
    """정답 조각이 임베딩·BM25에서 각각 몇 등인지 보여 준다 (자르기 방식마다)."""
    project_id_by_name = {project.project_name: project.project_id for project in projects}
    names = score_search.split_names(question["expected_project_names"])
    project_ids = [project_id_by_name[name] for name in names]
    keywords_by_project_id = score_search.match_keywords_to_projects(question, project_ids)
    print(f"\n=== {question['question_id']}: {question['question']}")
    print(f"    정답 키워드: {', '.join(score_search.get_keywords(question))}")

    for name, project_id in zip(names, project_ids, strict=True):
        keywords = [
            score_search.normalize_text(word) for word in keywords_by_project_id[project_id]
        ]
        if not keywords:
            print(f"    [{name}] 정답 키워드가 없는 문서 (사업명만 확인)")
            continue
        bm25_any = get_project_bm25(vector_store, project_id, "뜻 품사")
        answer_ids = {
            document.metadata["chunk_id"]
            for document in bm25_any.documents
            if any(word in score_search.normalize_text(document.page_content) for word in keywords)
        }
        embedding_ranked = vector_store.similarity_search_with_score(
            question["question"], k=len(bm25_any.documents), filter={"project_id": project_id}
        )
        embedding_ranks = [
            rank
            for rank, (document, _) in enumerate(embedding_ranked, start=1)
            if document.metadata["chunk_id"] in answer_ids
        ]
        print(f"    [{name}] 조각 {len(bm25_any.documents)}개, 정답 조각 {sorted(answer_ids)}")
        print(f"      임베딩 등수: {embedding_ranks}")

        for tokenizer_name in TOKENIZERS:
            bm25 = get_project_bm25(vector_store, project_id, tokenizer_name)
            ranked = bm25.rank_all(question["question"])
            bm25_ranks = [
                rank
                for rank, (document, _) in enumerate(ranked, start=1)
                if document.metadata["chunk_id"] in answer_ids
            ]
            print(f"      BM25 {tokenizer_name} 등수: {bm25_ranks}")
            print(f"        질문 단어: {bm25.tokenize(question['question'])}")
            for rank, (document, score) in enumerate(ranked[:3], start=1):
                snippet = document.page_content[:60].replace("\n", " ")
                print(f"        {rank}등 {score:.2f} {document.metadata['chunk_id']} | {snippet}")


def save_csv(path, rows: list[dict]) -> None:
    """결과를 csv로 저장한다. data/ 아래라 Git에 안 올라간다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    questions = score_search.load_questions("evaluation/questions.csv")
    vector_store = get_vector_store()
    projects = load_project_list(PROJECT_LIST_PATH)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    reranker = None
    if "--rerank" in sys.argv:
        from bidmate.retrieval.reranker import get_reranker

        print("리랭커 모델 불러오는 중...")
        reranker = get_reranker()

    # 비교할 설정: 임베딩만 1개 + (자르기 방식 × BM25 후보 수)
    settings = [("임베딩만", "-", 0)] + [
        (f"{tokenizer_name} BM25 {bm25_k}개", tokenizer_name, bm25_k)
        for tokenizer_name in TOKENIZERS
        for bm25_k in BM25_CANDIDATE_KS
    ]

    summary_rows: list[dict] = []
    question_rows: list[dict] = []
    for setting_name, tokenizer_name, bm25_k in settings:
        if bm25_k > 0:
            prebuild_bm25(questions, projects, vector_store, tokenizer_name)
        for mode, mode_reranker in [("후보 상한선", None)] + (
            [("리랭커 5개", reranker)] if reranker else []
        ):
            score_search.search_chunks = make_search(tokenizer_name, bm25_k, mode_reranker)
            result = score_search.score(
                questions, vector_store, projects, use_expected_projects=True
            )
            status = {row["question_id"]: row["content_status"] for row in result["rows"]}
            summary_rows.append(
                {
                    "실행 시각": run_time,
                    "방식": mode,
                    "설정": setting_name,
                    "자르기": tokenizer_name,
                    "BM25 후보 수": bm25_k,
                    "내용 기준": f"{result['content_score']:.1%}",
                    **{
                        question_id: status.get(question_id, "")
                        for question_id in WATCH_QUESTION_IDS
                    },
                    "틀린 질문": " ".join(result["content_wrong_question_ids"]),
                    "검색 시간 평균(초)": round(result["average_search_seconds"], 3),
                }
            )
            question_rows.extend(
                {"실행 시각": run_time, "방식": mode, "설정": setting_name, **row}
                for row in result["rows"]
            )

    # 결과를 표 모양으로 출력한다
    print("\n[BM25 하이브리드 실험] 검색만(정답 문서 넣고), 내용 기준")
    print("  후보 상한선 = 후보 안에 정답 조각이 있는 비율, 리랭커 5개 = 실제 앱 점수\n")
    header = (
        f"  {'방식':<8} {'설정':<18} {'점수':>6}  "
        + "  ".join(WATCH_QUESTION_IDS)
        + "   시간(초)  틀린 질문"
    )
    print(header)
    print("  " + "-" * (len(header) + 10))
    for row in summary_rows:
        watch = "  ".join(
            f"{'O' if row[question_id] == '맞음' else 'X':^4}" for question_id in WATCH_QUESTION_IDS
        )
        print(
            f"  {row['방식']:<8} {row['설정']:<18} {row['내용 기준']:>6}  {watch}"
            f"   {row['검색 시간 평균(초)']:>6}   {row['틀린 질문']}"
        )

    save_csv(RESULT_FOLDER / "experiment_bm25_summary.csv", summary_rows)
    save_csv(RESULT_FOLDER / "experiment_bm25_rows.csv", question_rows)
    print(f"\n결과 저장: {RESULT_FOLDER}/experiment_bm25_summary.csv, experiment_bm25_rows.csv")

    if "--inspect" in sys.argv:
        print("\n[들여다보기] 정답 조각이 임베딩·BM25에서 각각 몇 등인가 (등수가 작을수록 위)")
        for question in questions:
            if question["question_id"] in WATCH_QUESTION_IDS:
                inspect_question(question, projects, vector_store)

    if reranker is None:
        print("\n리랭커 점수까지 재려면 GPU가 있는 VM에서 --rerank를 붙여 실행해 주세요.")
