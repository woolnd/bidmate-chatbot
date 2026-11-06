"""조각 저장과 3단계(문서에서 관련 부분 찾기). 담당: 검색 파트

저장 (미리 한 번, scripts/build_index.py가 부른다):
    조각(Document)을 임베딩(숫자 목록)으로 바꿔 Chroma에 저장한다.
    metadata(chunk_id, project_id, agency_name, project_name)도 같이 저장된다.
    chunk_id를 저장 id로 써서 같은 조각이 두 번 들어가지 않게 한다.
    임베딩은 돈이 드니까 한 번 만든 것은 다시 만들지 않는다.

찾기 (질문마다):
    project_id 목록이 있으면 metadata로 걸러서 고른 문서 안에서만 찾는다.
    빈 목록이면 전체에서 찾는다.
    k(찾아올 조각 수)는 config.py에서 가져온다. 2주차에 3/5/10을 점수로 비교한다.

받는 것: (저장) Document 목록 / (찾기) 다듬은 질문, project_id 목록
넘기는 것: (찾기) 관련 조각 k개, (Document, 점수) 목록
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document


def get_vector_store() -> Chroma:
    """Chroma 저장소를 연다. 폴더에 이미 저장돼 있으면 그대로 다시 쓴다."""
    # TODO 1. 임베딩 모델 만들기 (OpenAIEmbeddings, 모델 이름은 config.EMBEDDING_MODEL_NAME)
    # TODO 2. Chroma 열기: 저장소 이름(COLLECTION_NAME), 임베딩 모델, 저장 폴더(CHROMA_FOLDER)
    raise NotImplementedError("검색 파트가 채운다")


def save_chunks(chunks: list[Document]) -> None:
    """조각을 임베딩해서 저장한다. 같은 조각은 두 번 들어가지 않는다."""
    # TODO 1. 조각마다 metadata["chunk_id"]를 꺼내 저장 id 목록 만들기
    # TODO 2. get_vector_store()로 저장소를 열고 add_documents(조각들, ids=id 목록)
    #         같은 id로 다시 넣으면 새로 생기지 않고 덮어쓴다
    print(f"TODO: 조각 {len(chunks)}개 저장 (아직 저장하지 않음)")


def search_chunks(question: str, project_ids: list[str], k: int) -> list[tuple[Document, float]]:
    """고른 문서 안에서 질문과 뜻이 비슷한 조각 k개를 찾는다. 빈 목록이면 전체에서 찾는다."""
    # TODO 1. project_ids가 있으면 거르기 조건 만들기: project_id가 목록 안에 있는 조각만
    #         ({"project_id": {"$in": project_ids}}), 빈 목록이면 조건 없음
    # TODO 2. similarity_search_with_score(질문, k=k, filter=거르기 조건)
    # TODO 3. 받은 (Document, 점수) 목록을 그대로 돌려주기
    #         점수는 거리라서 작을수록 비슷하다
    return []
