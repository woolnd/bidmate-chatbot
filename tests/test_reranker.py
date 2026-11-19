"""리랭커(src/bidmate/retrieval/reranker.py)와 리랭커를 켠 search_chunks 테스트.

진짜 리랭커 모델(약 2.3GB)은 받지 않는다. 가짜 리랭커로 "점수 높은 순서로 다시 고르는가"만 확인한다.
가짜 리랭커는 조각 본문에 정해 둔 글자가 있으면 1점, 없으면 0점을 준다.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import bidmate.retrieval.vector_store as vector_store_module
from bidmate.retrieval.reranker import rerank
from bidmate.retrieval.vector_store import get_vector_store, save_chunks, search_chunks
from tests.fake_chunks import ACADEMIC, HOMEPAGE, make_fake_chunks

FAKE_EMBEDDING_SIZE = 16


class FakeReranker:
    """본문에 target 글자가 있으면 1점, 없으면 0점을 주는 가짜 리랭커. 불린 횟수도 센다."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.calls = 0

    def predict(self, sentences: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return [1.0 if self.target in text else 0.0 for _, text in sentences]


def make_saved_store(folder: Path):
    """지어낸 조각 10개를 가짜 임베딩으로 저장한 테스트용 저장소."""
    vector_store = get_vector_store(DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE), folder)
    save_chunks(make_fake_chunks(), vector_store)
    return vector_store


def make_candidate(text: str, distance: float) -> tuple[Document, float]:
    return (Document(page_content=text, metadata={}), distance)


def test_rerank_orders_by_reranker_score_and_keeps_distance() -> None:
    """리랭커 점수가 높은 조각이 앞에 오고, 거리 값은 바꾸지 않는다."""
    candidates = [
        make_candidate("목차", 0.1),
        make_candidate("참가 자격 3년", 0.9),
        make_candidate("사업 범위", 0.5),
    ]

    results = rerank("참가 자격은?", candidates, 2, FakeReranker("자격"))

    assert [document.page_content for document, _ in results] == ["참가 자격 3년", "목차"]
    assert results[0][1] == 0.9  # 거리 값 그대로


def test_rerank_with_nothing_to_keep() -> None:
    """후보가 없거나 keep이 0이면 리랭커를 부르지 않고 빈 목록을 돌려준다."""
    reranker = FakeReranker("자격")

    assert rerank("질문", [], 3, reranker) == []
    assert rerank("질문", [make_candidate("본문", 0.1)], 0, reranker) == []
    assert reranker.calls == 0


def test_search_chunks_with_reranker_picks_reranker_choice(tmp_path: Path) -> None:
    """리랭커를 넣으면 후보 중에서 리랭커 점수가 높은 조각을 고른다 (임베딩 순서와 상관없이)."""
    qualification_chunk = make_fake_chunks()[6]  # 라마대학교 '참가 자격' 조각 ("3년"이 들어 있다)
    vector_store = make_saved_store(tmp_path)

    results = search_chunks("아무 질문", [ACADEMIC[0]], 1, vector_store, FakeReranker("3년"))

    assert len(results) == 1
    assert results[0][0].metadata["chunk_id"] == qualification_chunk.metadata["chunk_id"]


def test_search_chunks_with_reranker_keeps_split_shares(tmp_path: Path) -> None:
    """여러 문서 질문은 리랭커를 써도 문서마다 나눠 검색과 같은 개수를 남긴다 (k=3 → 2개 + 1개)."""
    vector_store = make_saved_store(tmp_path)

    results = search_chunks(
        "아무 질문", [ACADEMIC[0], HOMEPAGE[0]], 3, vector_store, FakeReranker("3년")
    )

    project_ids = [document.metadata["project_id"] for document, _ in results]
    assert project_ids.count(ACADEMIC[0]) == 2
    assert project_ids.count(HOMEPAGE[0]) == 1


def test_search_chunks_uses_reranker_only_when_setting_on(tmp_path, monkeypatch) -> None:
    """USE_RERANKER가 꺼져 있으면 리랭커를 불러오지 않고, 켜져 있으면 불러와서 쓴다."""
    vector_store = make_saved_store(tmp_path)
    fake_reranker = FakeReranker("3년")
    monkeypatch.setattr(vector_store_module, "get_reranker", lambda: fake_reranker)

    monkeypatch.setattr(vector_store_module, "USE_RERANKER", False)
    search_chunks("아무 질문", [ACADEMIC[0]], 1, vector_store)
    assert fake_reranker.calls == 0

    monkeypatch.setattr(vector_store_module, "USE_RERANKER", True)
    search_chunks("아무 질문", [ACADEMIC[0]], 1, vector_store)
    assert fake_reranker.calls == 1
