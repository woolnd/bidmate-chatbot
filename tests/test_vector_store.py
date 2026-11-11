"""검색-A(vector_store.py) 테스트.

가짜 임베딩(DeterministicFakeEmbedding)과 임시 폴더를 써서 OpenAI를 부르지 않는다 (비용 0원).
가짜 임베딩은 같은 글자에 늘 같은 숫자를 주지만 뜻은 모른다.
그래서 여기서는 "제대로 연결되는지"만 확인하고, 검색 품질은 진짜 데이터로 따로 본다.
"""

from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import bidmate.retrieval.vector_store as vector_store_module
from bidmate.config import COLLECTION_NAME
from bidmate.retrieval.vector_store import (
    get_vector_store,
    make_project_filter,
    save_chunks,
    search_chunks,
)
from bidmate.schemas import CHUNK_METADATA_KEYS
from tests.fake_chunks import ACADEMIC, ELEARNING, HOMEPAGE, make_fake_chunks

FAKE_EMBEDDING_SIZE = 16  # 가짜 임베딩 숫자 개수. 진짜(1536개)일 필요 없다


def test_get_vector_store_opens_collection_with_config_name(tmp_path: Path) -> None:
    """저장소 이름이 config.COLLECTION_NAME이고, 처음 열면 비어 있다."""
    vector_store = get_vector_store(DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE), tmp_path)

    assert vector_store._collection.name == COLLECTION_NAME
    assert vector_store._collection.count() == 0


def test_get_vector_store_reuses_saved_folder(tmp_path: Path) -> None:
    """같은 폴더를 다시 열면 전에 넣은 것이 그대로 남아 있다 (프로그램을 꺼도 유지)."""
    embedding_model = DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE)
    first_store = get_vector_store(embedding_model, tmp_path)
    first_store.add_texts(["저장 유지 확인용 글"], ids=["test-0001"])

    reopened_store = get_vector_store(embedding_model, tmp_path)

    assert reopened_store._collection.count() == 1
    assert reopened_store.get(ids=["test-0001"])["documents"] == ["저장 유지 확인용 글"]


# ---------- save_chunks ----------


def make_test_store(folder: Path) -> Chroma:
    """테스트용 저장소: 가짜 임베딩 + 임시 폴더 (OpenAI를 부르지 않는다)."""
    return get_vector_store(DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE), folder)


def test_save_chunks_keeps_content_and_metadata(tmp_path: Path) -> None:
    """지어낸 조각 10개가 모두 저장되고, 본문과 이름표 4칸이 그대로 남는다."""
    chunks = make_fake_chunks()
    vector_store = make_test_store(tmp_path)

    save_chunks(chunks, vector_store)

    assert vector_store._collection.count() == 10
    for chunk in chunks:
        saved = vector_store.get(ids=[chunk.metadata["chunk_id"]])
        assert saved["documents"] == [chunk.page_content]
        assert saved["metadatas"][0] == chunk.metadata


def test_save_chunks_twice_does_not_duplicate(tmp_path: Path) -> None:
    """같은 조각을 두 번 저장해도 개수가 늘지 않고, 바뀐 본문으로 덮어쓴다."""
    chunks = make_fake_chunks()
    vector_store = make_test_store(tmp_path)
    save_chunks(chunks, vector_store)

    chunks[1].page_content = chunks[1].page_content + " (수정됨)"
    save_chunks(chunks, vector_store)

    assert vector_store._collection.count() == 10
    saved = vector_store.get(ids=[chunks[1].metadata["chunk_id"]])
    assert saved["documents"][0].endswith("(수정됨)")


def test_save_chunks_empty_list_saves_nothing(tmp_path: Path) -> None:
    """빈 목록이 와도 오류 없이 끝나고 아무것도 저장하지 않는다."""
    vector_store = make_test_store(tmp_path)

    save_chunks([], vector_store)

    assert vector_store._collection.count() == 0


def test_save_chunks_keeps_project_id_exactly(tmp_path: Path) -> None:
    """띄어쓰기·괄호·대괄호·가운뎃점이 든 실제 모양의 파일 이름도 글자 그대로 저장된다."""
    project_ids = [
        "한국철도공사 (용역)_[재공고][긴급][협상형] 운행정보기록 자동분석시스템.hwp",
        "고려대학교_차세대포털·학사 정보시스템 구축사업.pdf",
    ]
    chunks = [
        Document(
            page_content=f"[기관 / 사업]\n내용 {number}",
            metadata={
                "chunk_id": f"{project_id}-0001",
                "project_id": project_id,
                "agency_name": "기관",
                "project_name": "사업",
            },
        )
        for number, project_id in enumerate(project_ids)
    ]
    vector_store = make_test_store(tmp_path)

    save_chunks(chunks, vector_store)

    saved_project_ids = {metadata["project_id"] for metadata in vector_store.get()["metadatas"]}
    assert saved_project_ids == set(project_ids)


def test_save_chunks_splits_into_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """조각이 SAVE_BATCH_SIZE보다 많아도 나눠서 모두 저장한다 (한 번에 3개씩으로 줄여서 확인)."""
    monkeypatch.setattr(vector_store_module, "SAVE_BATCH_SIZE", 3)
    vector_store = make_test_store(tmp_path)

    save_chunks(make_fake_chunks(), vector_store)

    assert vector_store._collection.count() == 10


def test_save_chunks_rejects_duplicate_chunk_id(tmp_path: Path) -> None:
    """한 번에 넘긴 목록 안에 chunk_id가 겹치면 오류를 내고 아무것도 저장하지 않는다."""
    chunks = make_fake_chunks()
    chunks.append(chunks[0])
    vector_store = make_test_store(tmp_path)

    with pytest.raises(ValueError, match="겹치는"):
        save_chunks(chunks, vector_store)
    assert vector_store._collection.count() == 0


@pytest.mark.parametrize("missing_key", ["chunk_id", "project_id", "agency_name", "project_name"])
def test_save_chunks_rejects_missing_metadata(tmp_path: Path, missing_key: str) -> None:
    """이름표 4칸 중 하나라도 없으면 오류를 내고 아무것도 저장하지 않는다."""
    chunks = make_fake_chunks()
    del chunks[3].metadata[missing_key]
    vector_store = make_test_store(tmp_path)

    with pytest.raises(ValueError, match=missing_key):
        save_chunks(chunks, vector_store)
    assert vector_store._collection.count() == 0


def test_save_chunks_rejects_empty_content(tmp_path: Path) -> None:
    """본문이 빈 조각이 있으면 오류를 낸다."""
    chunks = make_fake_chunks()
    chunks[0].page_content = "   "

    with pytest.raises(ValueError, match="본문"):
        save_chunks(chunks, make_test_store(tmp_path))


def test_save_chunks_rejects_non_document(tmp_path: Path) -> None:
    """Document가 아닌 것(예: 글자만)이 섞여 있으면 오류를 낸다."""
    chunks = make_fake_chunks() + ["그냥 글자"]

    with pytest.raises(TypeError):
        save_chunks(chunks, make_test_store(tmp_path))


# ---------- search_chunks ----------
# 가짜 임베딩은 뜻을 모르지만 "같은 글자 → 같은 숫자"라서,
# 조각 본문을 그대로 질문으로 넣으면 그 조각이 거리 0으로 1등이 된다.
# 이걸로 찾기가 맞게 도는지 본다.


def make_saved_store(folder: Path) -> Chroma:
    """지어낸 조각 10개(사업 3개)를 저장해 둔 테스트용 저장소."""
    vector_store = make_test_store(folder)
    save_chunks(make_fake_chunks(), vector_store)
    return vector_store


def test_make_project_filter() -> None:
    """빈 목록이면 None(전체), 있으면 $in 조건이고 project_id 글자는 그대로다."""
    assert make_project_filter([]) is None
    assert make_project_filter(["가나다공단_이러닝.hwp"]) == {
        "project_id": {"$in": ["가나다공단_이러닝.hwp"]}
    }


def test_search_chunks_returns_document_and_score_in_order(tmp_path: Path) -> None:
    """(Document, 점수) 목록을 돌려주고, 점수가 작은(가까운) 순서이며, 이름표 4칸이 그대로다."""
    vector_store = make_saved_store(tmp_path)

    results = search_chunks("예산은?", [], 5, vector_store)

    assert len(results) == 5
    for document, score in results:
        assert isinstance(document, Document)
        assert isinstance(score, float)
        assert tuple(sorted(document.metadata)) == tuple(sorted(CHUNK_METADATA_KEYS))
    scores = [score for _, score in results]
    assert scores == sorted(scores)


def test_search_chunks_finds_same_text_first(tmp_path: Path) -> None:
    """조각 본문과 똑같은 질문을 넣으면 그 조각이 1등으로 나온다."""
    target_chunk = make_fake_chunks()[5]
    vector_store = make_saved_store(tmp_path)

    results = search_chunks(target_chunk.page_content, [], 3, vector_store)

    best_document, best_score = results[0]
    assert best_document.metadata["chunk_id"] == target_chunk.metadata["chunk_id"]
    assert best_score < 1e-6


@pytest.mark.parametrize("k", [1, 3, 10])
def test_search_chunks_returns_at_most_k(tmp_path: Path, k: int) -> None:
    """k개까지만 돌려준다."""
    vector_store = make_saved_store(tmp_path)

    assert len(search_chunks("예산은?", [], k, vector_store)) == k


def test_search_chunks_empty_project_ids_searches_all(tmp_path: Path) -> None:
    """빈 목록이면 전체에서 찾는다: k를 넉넉히 주면 사업 3개 조각이 모두 나온다."""
    vector_store = make_saved_store(tmp_path)

    results = search_chunks("예산은?", [], 20, vector_store)

    assert len(results) == 10
    assert {document.metadata["project_id"] for document, _ in results} == {
        ELEARNING[0],
        ACADEMIC[0],
        HOMEPAGE[0],
    }


def test_search_chunks_filters_one_project(tmp_path: Path) -> None:
    """project_id 하나를 주면 그 문서 조각(4개)만 나온다."""
    vector_store = make_saved_store(tmp_path)

    results = search_chunks("예산은?", [ELEARNING[0]], 20, vector_store)

    assert len(results) == 4
    assert all(document.metadata["project_id"] == ELEARNING[0] for document, _ in results)


def test_search_chunks_filters_several_projects(tmp_path: Path) -> None:
    """project_id 여러 개를 주면 그 문서들 조각만 나온다 (3 + 3 = 6개)."""
    vector_store = make_saved_store(tmp_path)

    results = search_chunks("예산은?", [ACADEMIC[0], HOMEPAGE[0]], 20, vector_store)

    assert len(results) == 6
    assert {document.metadata["project_id"] for document, _ in results} == {
        ACADEMIC[0],
        HOMEPAGE[0],
    }


def test_search_chunks_project_id_must_match_exactly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """확장자를 뗀 이름처럼 글자가 하나라도 다르면 0개이고, 어떤 id가 없는지 경고를 남긴다."""
    vector_store = make_saved_store(tmp_path)

    results = search_chunks("예산은?", ["가나다공단_이러닝"], 5, vector_store)

    assert results == []
    assert "가나다공단_이러닝" in caplog.text


def test_search_chunks_empty_store_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """저장된 조각이 없으면 0개이고, build_index.py를 돌리라는 경고를 남긴다."""
    results = search_chunks("예산은?", [], 5, make_test_store(tmp_path))

    assert results == []
    assert "build_index" in caplog.text


@pytest.mark.parametrize("bad_k", [0, -1])
def test_search_chunks_rejects_bad_k(tmp_path: Path, bad_k: int) -> None:
    """k가 0이나 음수면 오류를 낸다."""
    with pytest.raises(ValueError, match="k"):
        search_chunks("예산은?", [], bad_k, make_test_store(tmp_path))


def test_search_chunks_rejects_empty_question(tmp_path: Path) -> None:
    """질문이 비어 있으면 오류를 낸다."""
    with pytest.raises(ValueError, match="질문"):
        search_chunks("   ", [], 5, make_test_store(tmp_path))
