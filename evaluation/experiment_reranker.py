"""리랭커 실험: 후보 조각을 넉넉히 찾고 리랭커로 다시 골랐을 때 검색 점수가 오르는지 본다.

담당: 검색 파트(검색-A). 실험용이라 search_chunks 코드는 바꾸지 않는다.

    지금      문서마다 나눠서 조각 5개를 바로 찾는다
    리랭커    문서마다 후보 CANDIDATE_K개를 찾고 → 리랭커가 질문과 한 쌍씩 다시 채점 → 5개
              여러 문서 질문은 문서마다 따로 골라서, 나눠 검색과 같은 개수(예: 3개 + 2개)를 남긴다

점수 방향:
    검색 점수(L2 거리)는 작을수록 비슷하고, 리랭커 점수는 클수록 관련 있다

1차 결과 (bge-reranker-v2-m3, 후보 20개, Mac GPU + float16):
    검색만(내용 기준)     74.3% → 91.4%
    실제 흐름(내용 기준)  62.9% → 74.3%
    검색 시간             평균 0.3초 → 13.2초
    응답 10초 기준을 넘어서 아직 search_chunks에는 넣지 않았다 (속도 해결이 다음 과제)

실행 (레포 맨 위 폴더에서):
    uv run --with sentence-transformers python -m evaluation.experiment_reranker
    - --with: sentence-transformers를 이번 실행에서만 임시로 설치한다 (pyproject.toml은 그대로)
    - -m: 레포 맨 위 폴더 기준으로 실행해야 evaluation 폴더를 찾는다
    - 처음 한 번은 리랭커 모델(약 2.3GB)을 내려받아서 몇 분 걸린다
    - 리랭커는 내 컴퓨터에서 돌아서 비용 0원. 질문 임베딩만 아주 적은 비용이 든다
"""

import torch
from sentence_transformers import CrossEncoder

import evaluation.score_search as score_search
from bidmate.config import PROJECT_LIST_PATH, TOP_K
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.vector_store import get_vector_store, search_chunks, split_k

CANDIDATE_K = 20  # 리랭커에게 넘길 후보 조각 수 (문서마다)
MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # 한국어를 지원하는 리랭커


def pick_device() -> str:
    """계산 장치를 고른다. NVIDIA GPU(cuda) → Mac GPU(mps) → CPU 순서로 있는 것을 쓴다."""
    if torch.cuda.is_available():
        return "cuda"  # VM의 NVIDIA GPU
    if torch.backends.mps.is_available():
        return "mps"  # Mac GPU
    return "cpu"


def load_reranker() -> CrossEncoder:
    """리랭커 모델을 연다. GPU가 있으면 GPU + float16, 없으면 CPU로 계산한다."""
    device = pick_device()
    reranker = CrossEncoder(MODEL_NAME, max_length=512, device=device)
    if device != "cpu":
        reranker.model.half()  # 숫자를 절반 크기(float16)로 계산해서 더 빠르게
    print(f"리랭커 계산 장치: {device}")
    return reranker


def rerank(reranker, question, candidates, keep):
    """후보 조각을 리랭커 점수(클수록 관련 있음)로 다시 줄 세우고 keep개만 남긴다."""
    if keep == 0 or len(candidates) == 0:
        return []
    scores = reranker.predict([(question, document.page_content) for document, _ in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in ranked[:keep]]


def make_rerank_search(reranker):
    """search_chunks와 모양이 같은 검색 함수를 만든다 (채점 코드에 바꿔 끼우려고)."""

    def rerank_search(question, project_ids, k, vector_store=None):
        """문서마다 후보 CANDIDATE_K개를 찾고, 리랭커로 문서마다 정해진 개수만 고른다."""
        if len(project_ids) <= 1:
            groups = [project_ids]
        else:
            groups = [[project_id] for project_id in project_ids]
        results = []
        for group, keep in zip(groups, split_k(k, len(groups)), strict=True):
            candidates = search_chunks(question, group, CANDIDATE_K, vector_store)
            results.extend(rerank(reranker, question, candidates, keep))
        return results

    return rerank_search


if __name__ == "__main__":
    print("리랭커 모델 불러오는 중 (처음 한 번은 약 2.3GB를 내려받아서 몇 분 걸려요)...")
    rerank_search = make_rerank_search(load_reranker())

    questions = score_search.load_questions("evaluation/questions.csv")
    vector_store = get_vector_store()
    projects = load_project_list(PROJECT_LIST_PATH)

    for name, use_expected in (("검색만", True), ("실제 흐름", False)):
        score_search.search_chunks = search_chunks  # 지금 방식 (나눠 검색까지)
        before = score_search.score(
            questions, vector_store, projects, use_expected_projects=use_expected, k=TOP_K
        )
        score_search.search_chunks = rerank_search  # 후보 20개 → 리랭커 → 5개
        after = score_search.score(
            questions, vector_store, projects, use_expected_projects=use_expected, k=TOP_K
        )
        print(f"\n[{name}] 내용 기준 Hit@5")
        print(f"  지금        : {before['content_score']:.1%}", end="  ")
        print(f"틀림 {before['content_wrong_question_ids']}")
        print(f"  + 리랭커    : {after['content_score']:.1%}", end="  ")
        print(f"틀림 {after['content_wrong_question_ids']}")
        print(f"  사업명 기준 : {before['score']:.1%} → {after['score']:.1%}")
        print(
            f"  검색 시간   : 평균 {before['average_search_seconds']:.3f}초 → "
            f"{after['average_search_seconds']:.3f}초 "
            f"(가장 느림 {after['slowest_search_seconds']:.3f}초)"
        )
