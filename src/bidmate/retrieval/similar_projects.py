"""비슷한 사업 찾기 (에이전트 similar 길, #64). 담당: 재웅

Chroma에 이미 저장된 조각 임베딩을 사업별로 평균 내서 사업끼리 비교한다.
새 임베딩도 AI 호출도 없다 (0원).

평균만 내면 모든 사업이 0.83~0.92로 비슷하게 나온다. RFP마다 "제안서 작성 요령",
"보안 요구사항" 같은 공통 문구가 많아서다. 그래서 100개 사업 평균(공통 방향)을 빼고 비교한다.
그러면 0.1~0.5로 벌어져 같은 종류 사업이 위로 올라온다 (11/19 확인).

받는 것: 기준 사업의 project_id
넘기는 것: SimilarProject 목록 (유사도 큰 순서)
"""

from functools import lru_cache

import numpy as np

from bidmate.config import SIMILAR_FALLBACK_COUNT, SIMILAR_MAX_COUNT, SIMILAR_MIN_SCORE
from bidmate.retrieval.vector_store import get_vector_store
from bidmate.schemas import SimilarProject


@lru_cache(maxsize=1)
def get_project_vectors() -> tuple[list[dict[str, str]], np.ndarray]:
    """사업마다 "평균 임베딩 - 전체 평균"을 길이 1로 만든 벡터. 서버가 처음 부를 때 한 번만 만든다.

    받는 것: 없음 (Chroma 저장소를 연다)
    넘기는 것: (사업 정보 목록, 벡터 행렬)
        사업 정보 목록: [{"project_id", "agency_name", "project_name"}, ...] 사업 100개
        벡터 행렬: (사업 수, 1536). i번째 줄이 사업 정보 목록 i번째 사업의 벡터
    """
    # 1. Chroma에서 모든 조각의 임베딩과 이름표를 꺼낸다 (조각 약 1만 개, 3초쯤)
    #    저장된 값을 읽기만 해서 API 호출은 없다
    #    stored = {"embeddings": [[1536개], ...], "metadatas": [{이름표}, ...]} 같은 순서
    stored = get_vector_store()._collection.get(include=["embeddings", "metadatas"])

    # 2. project_id별로 조각 임베딩을 바구니에 모은 뒤 칸마다 평균 낸다 → 사업 수 × 1536
    #    사업 정보(기관명·사업명)는 그 사업 조각의 이름표에서 가져온다 (조각마다 같다)
    embeddings_by_project: dict[str, list] = {}
    info_by_project: dict[str, dict[str, str]] = {}
    for embedding, metadata in zip(stored["embeddings"], stored["metadatas"], strict=True):
        project_id = metadata["project_id"]
        embeddings_by_project.setdefault(project_id, []).append(embedding)
        info_by_project[project_id] = {
            "project_id": project_id,
            "agency_name": metadata["agency_name"],
            "project_name": metadata["project_name"],
        }

    #    사업 순서를 항상 같게 (project_id 가나다순)
    project_infos = [info_by_project[project_id] for project_id in sorted(embeddings_by_project)]
    project_vectors = np.stack(
        [np.mean(embeddings_by_project[info["project_id"]], axis=0) for info in project_infos]
    )

    # 3. 전체 평균(100개 사업 벡터의 평균)을 빼서 공통 문구 방향을 지운다
    project_vectors = project_vectors - project_vectors.mean(axis=0)

    # 4. 줄마다 길이를 1로 맞춘다 (그러면 행렬 곱 = 코사인 유사도)
    project_vectors = project_vectors / np.linalg.norm(project_vectors, axis=1, keepdims=True)

    return project_infos, project_vectors


def find_similar_projects(project_id: str) -> list[SimilarProject]:
    """기준 사업과 비슷한 사업을 유사도 큰 순서로 돌려준다.

    받는 것: 기준 사업의 project_id
    넘기는 것: SimilarProject 목록
        SIMILAR_MIN_SCORE 이상인 사업은 모두 (최대 SIMILAR_MAX_COUNT개)
        하나도 없으면 상위 SIMILAR_FALLBACK_COUNT개
        (그래프가 "뚜렷하게 비슷한 사업은 없어요"라고 말한다)
        기준 사업이 저장소에 없으면 빈 목록
    """
    # 1. 사업 벡터 가져오기 (처음 한 번만 계산)
    project_infos, project_vectors = get_project_vectors()

    # 2. 기준 사업이 몇 번째 줄인지 찾기. 없으면 빈 목록
    project_ids = [info["project_id"] for info in project_infos]
    if project_id not in project_ids:
        return []
    base_index = project_ids.index(project_id)

    # 3. 기준 사업 벡터와 모든 사업 벡터의 유사도 (행렬 곱 한 번 → 사업 수만큼의 점수)
    #    기준 사업 자신은 빼고 유사도가 큰 순서로 정렬한다
    scores = project_vectors @ project_vectors[base_index]
    ranked_indexes = [index for index in np.argsort(-scores) if index != base_index]

    # 4. SIMILAR_MIN_SCORE 이상만 남기고 SIMILAR_MAX_COUNT개까지 자르기
    #    하나도 없으면 상위 SIMILAR_FALLBACK_COUNT개
    selected_indexes = [index for index in ranked_indexes if scores[index] >= SIMILAR_MIN_SCORE]
    selected_indexes = selected_indexes[:SIMILAR_MAX_COUNT]
    if not selected_indexes:
        selected_indexes = ranked_indexes[:SIMILAR_FALLBACK_COUNT]

    # 5. SimilarProject 목록으로 바꿔 돌려준다 (similarity는 float로)
    return [
        SimilarProject(
            project_id=project_infos[index]["project_id"],
            agency_name=project_infos[index]["agency_name"],
            project_name=project_infos[index]["project_name"],
            similarity=float(scores[index]),
        )
        for index in selected_indexes
    ]
