"""비슷한 사업 찾기 테스트. 진짜 Chroma 대신 가짜 저장소(임베딩을 직접 정한 것)를 쓴다."""

import pytest

from bidmate.retrieval import similar_projects


class FakeCollection:
    """get(include=...)에 정해 둔 임베딩·이름표를 돌려주는 가짜 Chroma 컬렉션."""

    def __init__(self, chunks: list[tuple[str, list[float]]]):
        self.chunks = chunks

    def get(self, include):
        return {
            "embeddings": [embedding for _, embedding in self.chunks],
            "metadatas": [
                {
                    "project_id": project_id,
                    "agency_name": f"{project_id}기관",
                    "project_name": f"{project_id}사업",
                    "chunk_id": f"{project_id}-1",
                }
                for project_id, _ in self.chunks
            ],
        }


class FakeStore:
    def __init__(self, chunks):
        self._collection = FakeCollection(chunks)


@pytest.fixture
def use_fake_store(monkeypatch):
    """가짜 조각으로 저장소를 바꾸고, 기억해 둔 사업 벡터를 지운다."""

    def install(chunks):
        monkeypatch.setattr(similar_projects, "get_vector_store", lambda: FakeStore(chunks))
        similar_projects.get_project_vectors.cache_clear()

    yield install
    similar_projects.get_project_vectors.cache_clear()


# 사업 4개. a와 b는 같은 방향(비슷), c는 다른 방향, d는 a·c 사이
CHUNKS = [
    ("a", [1.0, 0.0, 0.0]),
    ("a", [0.9, 0.1, 0.0]),  # 조각 2개 → 평균
    ("b", [0.95, 0.05, 0.0]),
    ("c", [0.0, 0.0, 1.0]),
    ("d", [0.5, 0.0, 0.5]),
]


def test_averages_chunks_per_project_in_fixed_order(use_fake_store):
    use_fake_store(CHUNKS)
    project_infos, project_vectors = similar_projects.get_project_vectors()
    assert [info["project_id"] for info in project_infos] == ["a", "b", "c", "d"]
    assert project_infos[0]["project_name"] == "a사업"
    assert project_vectors.shape == (4, 3)


def test_most_similar_first_and_excludes_itself(use_fake_store, monkeypatch):
    use_fake_store(CHUNKS)
    monkeypatch.setattr(similar_projects, "SIMILAR_MIN_SCORE", 0.5)
    result = similar_projects.find_similar_projects("a")
    assert [project.project_id for project in result][0] == "b"
    assert "a" not in [project.project_id for project in result]
    assert all(project.similarity >= 0.5 for project in result)
    assert isinstance(result[0].similarity, float)


def test_falls_back_to_top_when_none_pass_threshold(use_fake_store, monkeypatch):
    # 기준 점수를 넘는 사업이 없으면 상위 SIMILAR_FALLBACK_COUNT개
    use_fake_store(CHUNKS)
    monkeypatch.setattr(similar_projects, "SIMILAR_MIN_SCORE", 1.1)
    monkeypatch.setattr(similar_projects, "SIMILAR_FALLBACK_COUNT", 2)
    result = similar_projects.find_similar_projects("a")
    assert len(result) == 2
    assert result[0].project_id == "b"


def test_unknown_project_returns_empty(use_fake_store):
    use_fake_store(CHUNKS)
    assert similar_projects.find_similar_projects("없는사업") == []
