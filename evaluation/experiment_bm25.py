"""BM25(키워드 검색) 하이브리드 실험: 임베딩이 놓친 정답 조각을 글자 검색으로 건질 수 있는지 본다.

담당: 검색 파트(검색-A). 실험용이라 search_chunks 코드는 바꾸지 않는다.

왜 하나:
    리랭커는 후보 안에서만 고를 수 있다. Q007, Q011은 정답 조각이 임베딩 후보 20개 안에도 없어서
    리랭커로도 못 고쳤다 (답이 다른 주제 조각 안에 작게 섞여 있어서 뜻으로는 멀게 계산됨).
    BM25는 질문 단어가 "글자 그대로" 많이 든 조각에 높은 점수를 줘서, 이런 조각을 건질 수 있다.
    임베딩 후보와 BM25 후보를 합쳐 쓰는 것을 하이브리드 검색이라고 한다.

무엇을 재나 (검색만 = 정답 문서 넣고, 내용 기준):
    1. 후보 안에 정답 조각이 있는 비율 (= 리랭커가 낼 수 있는 최고 점수, 리랭커 없이 재서 빠르다)
         임베딩 20개 / 임베딩 20개 + BM25 5개
    2. --rerank를 붙이면: 리랭커로 최종 5개를 골랐을 때 점수 (= 실제 앱 점수, GPU가 있는 VM에서)
         리랭커만 (지금 앱) / 하이브리드 + 리랭커
    여러 문서 질문은 문서마다 따로 후보를 찾고, 나눠 검색과 같은 개수(split_k)를 고른다

한국어 단어 자르기:
    "하자보수는", "하자보수를"처럼 조사가 붙어서 그대로 세면 다른 단어가 된다.
    형태소 분석기(kiwipiepy)로 자르고 뜻을 가진 조각(명사, 숫자, 영어, 동사·형용사 뿌리)만 남긴다.

1차 결과 (11/18, Mac, 후보 안에 정답 조각이 있는 비율):
    임베딩 20개 94.3% → 임베딩 20개 + BM25 5개 97.1% (Q007이 후보에 들어옴, Q011은 여전히 없음)

실행 (레포 맨 위 폴더에서):
    후보만 (Mac):   uv run --with kiwipiepy --with rank-bm25 python -m evaluation.experiment_bm25
    리랭커까지 (VM): uv sync --extra rerank 를 한 번 한 뒤, 위 명령 끝에 --rerank를 붙인다
    - --with: 이번 실행에서만 임시로 설치한다 (pyproject.toml은 그대로)
    - BM25와 리랭커는 내 컴퓨터에서 계산해서 비용 0원. 질문 임베딩만 아주 적은 비용이 든다
"""

import sys

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import evaluation.score_search as score_search
from bidmate.config import PROJECT_LIST_PATH, RERANK_CANDIDATE_K
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.vector_store import get_vector_store, split_k

BM25_CANDIDATE_K = 5  # 더할 BM25 후보 수 (문서마다). 1차 실험에서 5개와 10개 결과가 같았다
WATCH_QUESTION_IDS = ("Q007", "Q011")  # 임베딩 후보 20개 안에도 없던 질문

# 뜻을 가진 품사만 남긴다: 명사(N...), 숫자(SN), 영어(SL), 어근(XR), 동사(VV), 형용사(VA)
KEEP_TAG_PREFIXES = ("N", "SN", "SL", "XR", "VV", "VA")

kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    """글을 뜻을 가진 단어 목록으로 자른다. 예: "하자보수 기간은?" → ["하자", "보수", "기간"]"""
    return [token.form for token in kiwi.tokenize(text) if token.tag.startswith(KEEP_TAG_PREFIXES)]


class ProjectBM25:
    """문서 하나의 조각들로 만든 BM25 검색기. 문서마다 한 번만 만들고 기억해 둔다."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(document.page_content) for document in documents])

    def search(self, question: str, k: int) -> list[tuple[Document, float]]:
        """질문 단어가 많이 든 조각 k개를 BM25 점수(클수록 관련 있음)가 큰 순서로 돌려준다."""
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(
            zip(self.documents, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [(document, float(score)) for document, score in ranked[:k] if score > 0]


bm25_cache: dict[str, ProjectBM25] = {}


def get_project_bm25(vector_store, project_id: str) -> ProjectBM25:
    """문서 하나의 BM25 검색기를 가져온다. 처음이면 Chroma에서 그 문서 조각을 모두 읽어 만든다."""
    if project_id not in bm25_cache:
        saved = vector_store.get(
            where={"project_id": project_id}, include=["documents", "metadatas"]
        )
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(saved["documents"], saved["metadatas"], strict=True)
        ]
        bm25_cache[project_id] = ProjectBM25(documents)
    return bm25_cache[project_id]


def find_candidates(question, project_id, bm25_k, vector_store) -> list[tuple[Document, float]]:
    """문서 하나에서 후보를 찾는다: 임베딩 후보 RERANK_CANDIDATE_K개 + BM25 후보 bm25_k개.

    bm25_k가 0이면 임베딩 후보만. 같은 조각이 두 곳에서 나오면 한 번만 넣는다.
    search_chunks 대신 Chroma를 바로 불러서, .env에 USE_RERANKER가 켜져 있어도 영향을 안 받는다.
    """
    found = vector_store.similarity_search_with_score(
        question, k=RERANK_CANDIDATE_K, filter={"project_id": project_id}
    )
    if bm25_k > 0:
        found += get_project_bm25(vector_store, project_id).search(question, bm25_k)

    candidates: list[tuple[Document, float]] = []
    seen_chunk_ids: set[str] = set()
    for document, score in found:
        if document.metadata["chunk_id"] not in seen_chunk_ids:
            seen_chunk_ids.add(document.metadata["chunk_id"])
            candidates.append((document, score))
    return candidates


def make_candidate_search(bm25_k: int):
    """후보를 전부 돌려주는 검색 함수를 만든다 (후보 안에 정답이 있는지 재려고).

    채점 코드(score_search)의 검색 함수 자리에 바꿔 끼워서 쓴다.
    """

    def candidate_search(question, project_ids, k, vector_store=None):
        results: list[tuple[Document, float]] = []
        for project_id in project_ids:
            results.extend(find_candidates(question, project_id, bm25_k, vector_store))
        return results

    return candidate_search


def make_rerank_search(bm25_k: int, reranker):
    """후보를 찾고 리랭커로 최종 k개를 고르는 검색 함수를 만든다 (실제 앱과 같은 방식).

    여러 문서 질문은 문서마다 후보를 찾고, 나눠 검색과 같은 개수(split_k)만큼 고른다.
    """
    from bidmate.retrieval.reranker import rerank

    def rerank_search(question, project_ids, k, vector_store=None):
        results: list[tuple[Document, float]] = []
        for project_id, keep in zip(project_ids, split_k(k, len(project_ids)), strict=True):
            candidates = find_candidates(question, project_id, bm25_k, vector_store)
            results.extend(rerank(question, candidates, keep, reranker))
        return results

    return rerank_search


def print_result(name: str, result: dict) -> None:
    """한 설정의 결과를 한 덩어리로 출력한다."""
    rows = {row["question_id"]: row for row in result["rows"]}
    watch = ", ".join(
        f"{question_id} {rows[question_id]['content_status']}"
        for question_id in WATCH_QUESTION_IDS
        if question_id in rows
    )
    print(f"  {name:<24}: {result['content_score']:.1%}  ({watch})")
    print(f"  {'':<24}  틀림 {result['content_wrong_question_ids']}")
    print(f"  {'':<24}  검색 시간 평균 {result['average_search_seconds']:.3f}초\n")


if __name__ == "__main__":
    use_reranker = "--rerank" in sys.argv
    questions = score_search.load_questions("evaluation/questions.csv")
    vector_store = get_vector_store()
    projects = load_project_list(PROJECT_LIST_PATH)

    def run(search_function) -> dict:
        score_search.search_chunks = search_function
        return score_search.score(questions, vector_store, projects, use_expected_projects=True)

    print("[1. 후보 안에 정답 조각이 있는 비율] 검색만(정답 문서 넣고), 내용 기준")
    print("  = 리랭커가 낼 수 있는 최고 점수\n")
    print_result("임베딩 20개", run(make_candidate_search(0)))
    print_result(
        f"임베딩 20개 + BM25 {BM25_CANDIDATE_K}개", run(make_candidate_search(BM25_CANDIDATE_K))
    )

    if use_reranker:
        from bidmate.retrieval.reranker import get_reranker

        print("리랭커 모델 불러오는 중...")
        reranker = get_reranker()
        print("\n[2. 리랭커로 고른 5개] 검색만(정답 문서 넣고), 내용 기준")
        print("  = 실제 앱 점수\n")
        print_result("리랭커만 (지금 앱)", run(make_rerank_search(0, reranker)))
        print_result(
            f"BM25 {BM25_CANDIDATE_K}개 + 리랭커",
            run(make_rerank_search(BM25_CANDIDATE_K, reranker)),
        )
    else:
        print("리랭커까지 재려면 GPU가 있는 VM에서 --rerank를 붙여 실행해 주세요.")
