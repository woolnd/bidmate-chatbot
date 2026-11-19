"""3단계 BM25(키워드 검색): 질문 단어가 글자 그대로 든 조각을 리랭커 후보에 더한다. 담당: 검색 파트

왜 쓰나:
    임베딩 검색은 뜻이 비슷한 조각을 찾는다. 그런데 답이 다른 주제 조각 안에 작게 섞여 있으면
    뜻으로는 멀게 계산돼서 후보 20개 안에도 못 들어온다 (리랭커는 후보 안에서만 고를 수 있다).
    BM25는 질문 단어가 글자 그대로 많이 든 조각에 높은 점수를 줘서 이런 조각을 건진다.
    임베딩 후보 + BM25 후보를 합쳐 리랭커에게 넘기는 것을 하이브리드 검색이라고 한다.

켜기/끄기 (.env):
    USE_BM25=true   리랭커(USE_RERANKER=true)와 같이 켤 때만 쓴다
                    (BM25 점수와 임베딩 거리는 단위가 달라서, 순서는 리랭커가 다시 매겨야 한다)
    켜기 전에 선택 패키지를 설치한다: uv sync --extra rerank (형태소 분석기 kiwipiepy 포함)

단어 자르기 (명사만):
    한국어는 "하자보수는"처럼 조사가 붙어서 그대로 세면 다른 단어가 된다.
    형태소 분석기(kiwipiepy)로 자르고 명사·숫자·영어·한자만 남긴다.

실험 결과 (evaluation/experiment_bm25.py, VM GPU, 리랭커 5개, 내용 기준):
    임베딩 20개만 90.9% → 임베딩 20개 + BM25 5개 97.0%, 검색 시간 0.46초 → 0.48초
    BM25 10개도 97.0%라서 5개를 쓴다
"""

from collections.abc import Callable
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# 남길 품사: 명사 N..., 숫자 SN, 영어 SL, 한자 SH
KEEP_TAG_PREFIXES = ("N", "SN", "SL", "SH")


@lru_cache(maxsize=1)
def get_kiwi():
    """형태소 분석기를 한 번만 불러 두고 계속 같은 것을 돌려준다."""
    try:
        from kiwipiepy import Kiwi  # 선택 패키지
    except ImportError as error:
        raise RuntimeError(
            "USE_BM25=true인데 형태소 분석기 패키지가 없습니다. "
            "uv sync --extra rerank 로 설치해 주세요."
        ) from error
    return Kiwi()


def tokenize(text: str) -> list[str]:
    """형태소로 자르고 명사·숫자·영어·한자만 남긴다. 예: "하자보수 기간은?" → 하자, 보수, 기간"""
    return [
        token.form for token in get_kiwi().tokenize(text) if token.tag.startswith(KEEP_TAG_PREFIXES)
    ]


class ProjectBM25:
    """문서 하나의 조각들로 만든 BM25 검색기."""

    def __init__(self, documents: list[Document], tokenize: Callable[[str], list[str]]) -> None:
        self.documents = documents
        self.tokenize = tokenize
        self.bm25 = BM25Okapi([tokenize(document.page_content) for document in documents])

    def search(self, question: str, k: int) -> list[Document]:
        """질문 단어가 많이 든 조각 k개 (BM25 점수가 큰 순서, 점수가 0인 조각은 뺀다)."""
        if k <= 0 or len(self.documents) == 0:
            return []
        scores = self.bm25.get_scores(self.tokenize(question))
        ranked = sorted(
            zip(self.documents, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [document for document, score in ranked[:k] if score > 0]


@lru_cache(maxsize=None)
def get_project_bm25(vector_store: Chroma, project_id: str) -> ProjectBM25:
    """문서 하나의 BM25 검색기를 가져온다. 처음이면 Chroma에서 그 문서 조각을 모두 읽어 만든다.

    @lru_cache: 문서마다 처음 한 번만 만들고 그다음부터는 기억해 둔 것을 쓴다.
    그래서 문서마다 첫 질문만 조금(1~2초) 느리다.
    build_index.py로 조각을 다시 만들면 서버를 다시 켜야 새 조각으로 만든다.
    """
    saved = vector_store.get(where={"project_id": project_id}, include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(saved["documents"], saved["metadatas"], strict=True)
    ]
    return ProjectBM25(documents, tokenize)


def squared_distance(vector_a, vector_b) -> float:
    """두 숫자 목록의 L2 제곱 거리 (Chroma 기본 거리와 같은 계산). 작을수록 비슷하다.

    예: [1, 2]와 [1, 4] → (1-1)² + (2-4)² = 0 + 4 = 4
    """
    return float(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b, strict=True)))


def add_bm25_candidates(
    question: str,
    question_vector: list[float],
    candidates: list[tuple[Document, float]],
    project_id: str,
    bm25_k: int,
    vector_store: Chroma,
) -> list[tuple[Document, float]]:
    """임베딩 후보에 BM25 후보 bm25_k개를 더한다. 이미 있는 조각은 한 번만 넣는다.

    받는 것: 질문, 질문 임베딩, 임베딩 후보 [(Document, 거리), ...], 문서 하나의 project_id,
             더할 BM25 후보 수, 저장소
    넘기는 것: [(Document, 거리), ...] 임베딩 후보 뒤에 새 BM25 후보가 붙는다
        새 후보의 거리도 저장된 임베딩으로 계산해서 넣는다 (점수는 거리라는 약속을 지키려고)
    """
    seen_chunk_ids = {document.metadata["chunk_id"] for document, _ in candidates}
    new_documents = [
        document
        for document in get_project_bm25(vector_store, project_id).search(question, bm25_k)
        if document.metadata["chunk_id"] not in seen_chunk_ids
    ]
    if not new_documents:
        return candidates

    # 새 후보의 저장된 임베딩을 꺼내서 질문과의 거리를 잰다 (내 컴퓨터에서 계산, 비용 0원)
    chunk_ids = [document.metadata["chunk_id"] for document in new_documents]
    saved = vector_store.get(ids=chunk_ids, include=["embeddings"])
    vector_by_id = dict(zip(saved["ids"], saved["embeddings"], strict=True))
    return candidates + [
        (document, squared_distance(question_vector, vector_by_id[document.metadata["chunk_id"]]))
        for document in new_documents
    ]
