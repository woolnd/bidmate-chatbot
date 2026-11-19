"""BM25(src/bidmate/retrieval/bm25.py)와 BM25를 켠 search_chunks 테스트.

형태소 분석기(kiwipiepy)는 선택 패키지라 CI에 없다. 그래서 띄어쓰기로 잘라서 테스트한다.
임베딩은 가짜(DeterministicFakeEmbedding)라서 OpenAI를 부르지 않는다.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import bidmate.retrieval.bm25 as bm25_module
import bidmate.retrieval.vector_store as vector_store_module
from bidmate.retrieval.bm25 import ProjectBM25, add_bm25_candidates, squared_distance
from bidmate.retrieval.vector_store import get_vector_store, save_chunks, search_chunks
from tests.fake_chunks import ELEARNING, make_fake_chunks

FAKE_EMBEDDING_SIZE = 16
CERTIFICATE_CHUNK_ID = f"{ELEARNING[0]}-0002"  # 이러닝 '수료증 발급' 조각


class FakeReranker:
    """본문에 target 글자가 있으면 1점, 없으면 0점을 주는 가짜 리랭커."""

    def __init__(self, target: str) -> None:
        self.target = target

    def predict(self, sentences: list[tuple[str, str]]) -> list[float]:
        return [1.0 if self.target in text else 0.0 for _, text in sentences]


@pytest.fixture(autouse=True)
def split_by_spaces(monkeypatch):
    """단어 자르기를 띄어쓰기로 바꾸고, 테스트마다 BM25 검색기 기억을 비운다."""
    monkeypatch.setattr(bm25_module, "tokenize", str.split)
    bm25_module.get_project_bm25.cache_clear()
    yield
    bm25_module.get_project_bm25.cache_clear()


def make_saved_store(folder: Path):
    """지어낸 조각 10개를 가짜 임베딩으로 저장한 테스트용 저장소."""
    vector_store = get_vector_store(DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE), folder)
    save_chunks(make_fake_chunks(), vector_store)
    return vector_store


def test_project_bm25_finds_chunk_with_question_words() -> None:
    """질문 단어가 든 조각이 1등이고, 질문 단어가 하나도 없는 조각(점수 0)은 빠진다."""
    documents = [
        Document(page_content="사업 목적 통합", metadata={"chunk_id": "a"}),
        Document(page_content="수료증 발급 기능", metadata={"chunk_id": "b"}),
        Document(page_content="예산 금액", metadata={"chunk_id": "c"}),
    ]

    results = ProjectBM25(documents, str.split).search("수료증 발급은?", 3)

    assert [document.metadata["chunk_id"] for document in results] == ["b"]


def test_squared_distance() -> None:
    """(1-1)² + (2-4)² = 4"""
    assert squared_distance([1.0, 2.0], [1.0, 4.0]) == 4.0


def test_add_bm25_candidates_adds_new_chunk_with_distance(tmp_path: Path) -> None:
    """임베딩 후보에 없던 BM25 조각을 더한다. 거리는 Chroma가 재는 거리와 같고, 겹치면 한 번만."""
    vector_store = make_saved_store(tmp_path)
    question = "수료증 발급"
    question_vector = vector_store.embeddings.embed_query(question)
    all_chunks = vector_store.similarity_search_by_vector_with_relevance_scores(
        question_vector, k=10, filter={"project_id": ELEARNING[0]}
    )
    distance_by_id = {document.metadata["chunk_id"]: score for document, score in all_chunks}
    others = [pair for pair in all_chunks if pair[0].metadata["chunk_id"] != CERTIFICATE_CHUNK_ID]

    added = add_bm25_candidates(question, question_vector, others, ELEARNING[0], 5, vector_store)
    added_again = add_bm25_candidates(
        question, question_vector, added, ELEARNING[0], 5, vector_store
    )

    assert added[-1][0].metadata["chunk_id"] == CERTIFICATE_CHUNK_ID
    assert added[-1][1] == pytest.approx(distance_by_id[CERTIFICATE_CHUNK_ID], rel=1e-4)
    assert len(added_again) == len(added)


def test_search_chunks_with_bm25_gives_reranker_bm25_chunk(tmp_path, monkeypatch) -> None:
    """임베딩 후보가 1개뿐이어도 BM25를 켜면 '수료증' 조각이 후보에 들어와 리랭커가 고른다."""
    vector_store = make_saved_store(tmp_path)
    monkeypatch.setattr(vector_store_module, "RERANK_CANDIDATE_K", 1)
    monkeypatch.setattr(vector_store_module, "USE_BM25", True)

    results = search_chunks("수료증 발급", [ELEARNING[0]], 1, vector_store, FakeReranker("수료증"))

    assert results[0][0].metadata["chunk_id"] == CERTIFICATE_CHUNK_ID


def test_bm25_used_only_with_reranker_and_chosen_document(tmp_path, monkeypatch) -> None:
    """BM25는 리랭커를 켜고, 문서가 정해졌을 때만 쓴다. 꺼져 있으면 부르지 않는다."""
    vector_store = make_saved_store(tmp_path)
    calls: list[str] = []

    def spy(question, question_vector, candidates, project_id, bm25_k, vector_store):
        calls.append(project_id)
        return candidates

    monkeypatch.setattr(vector_store_module, "add_bm25_candidates", spy)

    monkeypatch.setattr(vector_store_module, "USE_BM25", False)
    search_chunks("수료증", [ELEARNING[0]], 1, vector_store, FakeReranker("수료증"))
    assert calls == []  # BM25 꺼짐

    monkeypatch.setattr(vector_store_module, "USE_BM25", True)
    monkeypatch.setattr(vector_store_module, "USE_RERANKER", False)
    search_chunks("수료증", [ELEARNING[0]], 1, vector_store)
    assert calls == []  # 리랭커 꺼짐

    search_chunks("수료증", [], 1, vector_store, FakeReranker("수료증"))
    assert calls == []  # 문서를 못 골라 전체에서 찾을 때

    search_chunks("수료증", [ELEARNING[0]], 1, vector_store, FakeReranker("수료증"))
    assert calls == [ELEARNING[0]]
