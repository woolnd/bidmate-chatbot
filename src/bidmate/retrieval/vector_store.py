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

실행해 보기 (레포 맨 위 폴더에서 `uv run python`을 켜고 한 줄씩 입력, 끝낼 때 exit()):
    (.env에 OPENAI_API_KEY가 있어야 한다. 질문 임베딩에 아주 적은 비용이 든다)
    >>> from bidmate.config import TOP_K
    >>> from bidmate.retrieval.vector_store import search_chunks
    >>> results = search_chunks("이러닝 사업 예산은?", [], TOP_K)          # 전체에서 찾기
    >>> results = search_chunks("예산은?", ["파일이름.hwp"], TOP_K)        # 고른 문서 안에서만
    >>> for document, score in results: print(score, document.metadata["chunk_id"])   # 엔터 두 번
    저장(save_chunks)은 따로 부르지 말고 scripts/build_index.py로 한다 (임베딩 비용).
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from bidmate.config import CHROMA_FOLDER, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from bidmate.schemas import CHUNK_METADATA_KEYS

# 한 번에 저장할 조각 수. Chroma는 한 번에 5,461개까지만 받고,
# OpenAI 임베딩도 한 번 요청에 넣을 수 있는 글자 양에 한도가 있어서 나눠서 넣는다.
# 1000자 조각 100개면 두 한도 모두에 여유가 있다
SAVE_BATCH_SIZE = 100

# 검색 결과가 0개일 때 이유를 남기는 기록 도구 (print와 달리 경고 단계로 남는다)
logger = logging.getLogger(__name__)


def get_vector_store(
    embedding_model: Embeddings | None = None,
    persist_folder: Path = CHROMA_FOLDER,
) -> Chroma:
    """Chroma 저장소를 연다. 폴더에 이미 저장돼 있으면 그대로 다시 쓰고, 없으면 새로 만든다.

    받는 것: embedding_model (테스트에서 가짜 임베딩을 넣는다. 비우면 진짜 OpenAI 임베딩을 만든다),
             persist_folder (저장 폴더. 테스트에서 임시 폴더를 넣는다. 비우면 config.CHROMA_FOLDER)
    넘기는 것: Chroma 저장소 (저장소 이름 = config.COLLECTION_NAME)

    거리 방식은 Chroma 기본값(L2, 제곱 거리)을 쓴다. 점수는 작을수록 비슷하다.
    OpenAI 임베딩은 길이가 1이라 L2 순위와 cosine 순위가 같다.
    """
    # 1. 임베딩 모델을 안 넣었으면 진짜 모델을 만든다 (테스트에서는 가짜 임베딩을 넣는다)
    #    OPENAI_API_KEY가 .env에 없으면 여기서 오류가 난다
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 2. Chroma 열기: 저장 폴더에 같은 이름의 저장소가 있으면 그대로 열고, 없으면 새로 만든다
    #    (폴더가 없으면 Chroma가 만든다)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_folder),
    )


def check_chunks(chunks: list[Document]) -> None:
    """저장하기 전에 조각이 조각 규칙을 지키는지 확인한다. 하나라도 틀리면 아무것도 저장하지 않는다.

    받는 것: 조각(Document) 목록
    넘기는 것: 없음. 문제가 있으면 무엇이 틀렸는지 알려 주는 오류를 낸다
        TypeError   Document가 아닌 것이 섞여 있을 때
        ValueError  본문이 비었거나, 이름표 4칸이 없거나 비었거나, chunk_id가 겹칠 때
    """
    seen_chunk_ids: set[str] = set()
    duplicate_chunk_ids: list[str] = []

    for position, chunk in enumerate(chunks):
        # 1. Document인지 확인
        if not isinstance(chunk, Document):
            raise TypeError(f"{position}번째 조각이 Document가 아닙니다: {type(chunk).__name__}")

        # 2. 본문이 비어 있지 않은지 확인 (빈 글은 임베딩해도 찾을 수 없다)
        if not chunk.page_content.strip():
            raise ValueError(f"{position}번째 조각의 본문이 비어 있습니다: {chunk.metadata}")

        # 3. 이름표 4칸(CHUNK_METADATA_KEYS)이 모두 글자로 채워져 있는지 확인
        #    project_id는 글자 그대로 저장해야 해서 앞뒤 공백도 고치지 않고 검사만 한다
        for key in CHUNK_METADATA_KEYS:
            value = chunk.metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{position}번째 조각의 이름표 '{key}'가 없거나 비었습니다: {value!r}"
                )

        # 4. chunk_id가 겹치는지 모으기 (Chroma는 한 번에 같은 id가 두 번 오면 전부 실패한다)
        chunk_id = chunk.metadata["chunk_id"]
        if chunk_id in seen_chunk_ids:
            duplicate_chunk_ids.append(chunk_id)
        seen_chunk_ids.add(chunk_id)

    if duplicate_chunk_ids:
        raise ValueError(f"chunk_id가 겹치는 조각이 있습니다: {duplicate_chunk_ids[:10]}")


def save_chunks(chunks: list[Document], vector_store: Chroma | None = None) -> None:
    """조각을 임베딩해서 저장한다. 같은 조각은 두 번 들어가지 않는다.

    받는 것: 조각(Document) 목록 (데이터 파트가 만든 것. scripts/build_index.py가 넘긴다),
             vector_store (테스트에서 가짜 임베딩으로 연 저장소를 넣는다. 비우면 get_vector_store())
    넘기는 것: 없음 (Chroma 저장 폴더에 저장된다)

    같은 chunk_id로 다시 넣으면 새로 생기지 않고 덮어쓴다(upsert).
    덮어쓸 때도 임베딩을 다시 만들어서 비용이 든다. 그래서 build_index.py는 꼭 필요할 때만 돌린다.
    """
    # 1. 조각이 없으면 저장할 것이 없다 (데이터 파트 함수가 아직 TODO일 때도 여기로 온다)
    if len(chunks) == 0:
        print("저장할 조각이 0개입니다. 아무것도 저장하지 않았습니다.")
        return

    # 2. 저장 전에 조각 규칙 확인. 틀린 조각이 있으면 여기서 멈춘다 (임베딩 비용이 들기 전에)
    check_chunks(chunks)

    # 3. 저장소 열기 (테스트에서는 가짜 임베딩으로 연 저장소가 들어온다)
    if vector_store is None:
        vector_store = get_vector_store()

    # 4. SAVE_BATCH_SIZE개씩 나눠서 저장한다. chunk_id를 저장 id로 쓴다
    #    add_documents는 안에서 upsert를 써서, 이미 있는 id면 새로 만들지 않고 덮어쓴다
    for start in range(0, len(chunks), SAVE_BATCH_SIZE):
        batch = chunks[start : start + SAVE_BATCH_SIZE]
        batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
        vector_store.add_documents(batch, ids=batch_ids)
        print(f"저장 중: {start + len(batch)} / {len(chunks)}")

    # 5. 저장 결과 확인용으로 저장소 전체 조각 수를 보여 준다
    print(
        f"조각 {len(chunks)}개 저장 완료. 저장소 전체 조각 수: {vector_store._collection.count()}"
    )


def make_project_filter(project_ids: list[str]) -> dict | None:
    """검색-B가 고른 project_id 목록으로 Chroma 거르기 조건을 만든다.

    받는 것: project_id 목록 (원본 파일 이름 그대로. 글자를 고치지 않는다)
    넘기는 것: 거르기 조건. 빈 목록이면 None (= 거르지 않고 전체에서 찾기)

    빈 목록을 {"project_id": {"$in": []}}로 넘기면 Chroma가 오류를 내서 None으로 바꾼다.
    """
    if len(project_ids) == 0:
        return None
    return {"project_id": {"$in": project_ids}}


def search_chunks(
    question: str,
    project_ids: list[str],
    k: int,
    vector_store: Chroma | None = None,
) -> list[tuple[Document, float]]:
    """고른 문서 안에서 질문과 뜻이 비슷한 조각 k개를 찾는다. 빈 목록이면 전체에서 찾는다.

    받는 것: question (1단계가 다듬은 질문), project_ids (2단계가 고른 문서. 못 고르면 빈 목록),
             k (찾아올 조각 수, 보통 config.TOP_K),
             vector_store (테스트에서 가짜 임베딩으로 연 저장소를 넣는다. 비우면 get_vector_store())
    넘기는 것: [(Document, 점수), ...] 최대 k개. 점수가 작은(더 비슷한) 조각이 앞에 온다
               점수는 L2 거리라서 확률이 아니다. 순서를 비교할 때만 쓴다
    """
    # 1. 받은 값 확인. 잘못된 값은 Chroma까지 가기 전에 알기 쉬운 오류로 멈춘다
    if not question.strip():
        raise ValueError("질문이 비어 있습니다.")
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")

    # 2. 저장소 열기 (테스트에서는 가짜 임베딩으로 연 저장소가 들어온다)
    if vector_store is None:
        vector_store = get_vector_store()

    # 3. 거르기 조건 만들기: project_ids가 있으면 그 문서 조각만, 빈 목록이면 None(전체)
    project_filter = make_project_filter(project_ids)

    # 4. 질문을 임베딩해서 가장 가까운 조각 k개 찾기 (거리가 작은 순서로 온다)
    results = vector_store.similarity_search_with_score(question, k=k, filter=project_filter)

    # 5. 0개면 이유를 경고로 남긴다. 4단계는 0개면 "찾은 문서에 없습니다"라고만 답해서
    #    여기서 이유를 남기지 않으면 저장 문제인지, project_id 문제인지 알 수 없다
    if len(results) == 0:
        warn_empty_result(vector_store, project_ids)

    return results


def warn_empty_result(vector_store: Chroma, project_ids: list[str]) -> None:
    """검색 결과가 0개일 때 이유를 찾아 경고로 남긴다. 결과가 0개일 때만 불린다.

    받는 것: 저장소, 검색에 쓴 project_id 목록
    넘기는 것: 없음 (로그에 경고를 남긴다)
    """
    # 1. 저장소가 비어 있으면 build_index.py를 아직 안 돌린 것이다
    if vector_store._collection.count() == 0:
        logger.warning("저장소에 조각이 0개입니다. scripts/build_index.py를 먼저 돌려야 합니다.")
        return

    # 2. 저장소에 없는 project_id 찾기 (띄어쓰기·괄호·확장자 등 글자가 다르면 여기에 나온다)
    missing_project_ids = [
        project_id
        for project_id in project_ids
        if len(vector_store.get(where={"project_id": project_id}, limit=1)["ids"]) == 0
    ]
    logger.warning(
        "검색 결과가 0개입니다. 저장소에 없는 project_id: %s (글자가 저장된 값과 똑같은지 확인)",
        missing_project_ids,
    )
