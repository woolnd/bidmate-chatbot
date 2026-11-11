
"""조각 저장과 3단계(문서에서 관련 부분 찾기).

담당: 검색 파트

저장:
    Document 조각을 임베딩으로 변환해 Chroma에 저장한다.

검색:
    질문과 의미가 비슷한 조각을 TOP_K개 가져온다.
    project_ids가 있으면 해당 문서 안에서만 검색한다.
    project_ids가 비어 있으면 전체 문서에서 검색한다.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from bidmate.config import (
    CHROMA_FOLDER,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Chroma 저장소를 열어 반환한다.

    같은 설정의 저장소 객체는 한 번만 생성하고 재사용한다.
    """

    # 저장 폴더가 없으면 생성한다.
    CHROMA_FOLDER.mkdir(parents=True, exist_ok=True)

    # OpenAI 임베딩 모델 생성
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    # 기존 저장소가 있으면 열고, 없으면 새로 생성한다.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_FOLDER),
    )

    return vector_store


def save_chunks(chunks: list[Document]) -> None:
    """Document 조각을 Chroma에 저장한다.

    chunk_id를 저장 ID로 사용하므로 같은 ID의 조각은
    중복 생성하지 않고 기존 내용을 갱신한다.
    """

    if not chunks:
        print("[안내] 저장할 조각이 없습니다.")
        return

    ids = []

    for index, chunk in enumerate(chunks):
        chunk_id = chunk.metadata.get("chunk_id")

        if not chunk_id:
            raise ValueError(
                f"{index}번째 조각에 metadata['chunk_id']가 없습니다."
            )

        ids.append(str(chunk_id))

    if len(ids) != len(set(ids)):
        raise ValueError(
            "저장할 조각 안에 중복된 chunk_id가 있습니다."
        )

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"[저장 완료] 조각 {len(chunks)}개")


def search_chunks(
    question: str,
    project_ids: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    """질문과 유사한 문서 조각을 검색한다.

    Args:
        question: 검색할 질문
        project_ids: 검색할 문서의 원본 파일명 목록
                     빈 목록이면 전체 문서에서 검색
        k: 가져올 조각 개수

    Returns:
        (Document, 점수) 목록
        점수는 거리(distance)이므로 작을수록 유사하다.
    """

    if not question.strip():
        raise ValueError("검색 질문이 비어 있습니다.")

    if k < 1:
        raise ValueError("검색 개수 k는 1 이상이어야 합니다.")

    vector_store = get_vector_store()

    # 특정 문서만 검색하도록 필터를 설정한다.
    search_filter = None

    if project_ids:
        search_filter = {
            "project_id": {
                "$in": project_ids,
            }
        }

    # 검색 실행
    if search_filter is None:
        results = vector_store.similarity_search_with_score(
            query=question,
            k=k,
        )
    else:
        results = vector_store.similarity_search_with_score(
            query=question,
            k=k,
            filter=search_filter,
        )

    return results
