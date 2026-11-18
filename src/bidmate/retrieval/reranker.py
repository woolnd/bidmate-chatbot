"""3단계 리랭커: 후보 조각을 질문과 한 쌍씩 다시 채점해서 순서를 다시 매긴다. 담당: 검색 파트

왜 쓰나:
    임베딩 검색은 질문과 조각을 따로 숫자 목록으로 바꿔서 거리만 잰다 (빠르지만 대충).
    리랭커는 질문과 조각을 한 쌍으로 같이 읽고 "이 조각이 답인가?"를 점수로 매긴다 (느리지만 꼼꼼).
    그래서 임베딩으로 후보를 넉넉히(RERANK_CANDIDATE_K개) 찾고, 리랭커로 k개만 다시 고른다.

켜기/끄기 (.env):
    USE_RERANKER=true   GPU가 있는 VM에서 켠다 (질문당 약 0.25초 더 걸림)
    비워 두기 (기본)    끈다. 로컬 Mac은 GPU가 약해서 질문당 약 13초가 걸린다
    켜기 전에 선택 패키지를 설치한다: uv sync --extra rerank (torch 포함, 용량이 크다)

점수 방향:
    임베딩 점수(L2 거리)는 작을수록 비슷하고, 리랭커 점수는 클수록 관련 있다.
    돌려줄 때는 원래 (Document, 거리)를 그대로 두고 순서만 리랭커 점수로 바꾼다
    (화면·로그에서 점수는 거리라는 약속을 지키려고).

1차 실험 결과 (evaluation/experiment_reranker.py, 내용 기준 Hit@5):
    VM GPU   74.3% → 88.6%, 검색 시간 평균 0.23초 → 0.46초
"""

from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document

from bidmate.config import RERANKER_MAX_LENGTH, RERANKER_MODEL_NAME


class Reranker(Protocol):
    """리랭커 모양: (질문, 조각 본문) 쌍 목록을 받아 쌍마다 점수(클수록 관련 있음)를 돌려준다.

    진짜는 sentence_transformers의 CrossEncoder, 테스트에서는 가짜를 넣는다.
    """

    def predict(self, sentences: list[tuple[str, str]]) -> list[float]: ...


def pick_device() -> str:
    """계산 장치를 고른다. NVIDIA GPU(cuda) → Mac GPU(mps) → CPU 순서로 있는 것을 쓴다."""
    import torch  # 선택 패키지라서 리랭커를 켤 때만 불러온다

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """리랭커 모델을 한 번만 불러 두고 계속 같은 모델을 돌려준다.

    받는 것: 없음 (모델 이름과 길이는 config.py에서 가져온다)
    넘기는 것: predict()를 쓸 수 있는 리랭커
    처음 한 번은 모델(약 2.3GB)을 내려받고 불러오느라 시간이 걸린다.
    그래서 서버를 켤 때 미리 한 번 불러 둔다 (app/server.py).
    @lru_cache: 처음 부를 때만 모델을 불러오고, 그다음부터는 기억해 둔 모델을 돌려준다.
    """
    try:
        from sentence_transformers import CrossEncoder  # 선택 패키지
    except ImportError as error:
        raise RuntimeError(
            "USE_RERANKER=true인데 리랭커 패키지가 없습니다. "
            "uv sync --extra rerank 로 설치해 주세요."
        ) from error

    device = pick_device()
    reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=RERANKER_MAX_LENGTH, device=device)
    if device != "cpu":
        reranker.model.half()  # GPU면 숫자를 절반 크기(float16)로 계산해서 더 빠르게
    return reranker


def rerank(
    question: str,
    candidates: list[tuple[Document, float]],
    keep: int,
    reranker: Reranker,
) -> list[tuple[Document, float]]:
    """후보 조각을 리랭커 점수(클수록 관련 있음) 순서로 다시 줄 세우고 keep개만 남긴다.

    받는 것: 질문, 후보 [(Document, 거리), ...], 남길 개수 keep, 리랭커
    넘기는 것: [(Document, 거리), ...] 최대 keep개. 거리 값은 그대로, 순서만 리랭커 순서
    """
    if keep <= 0 or len(candidates) == 0:
        return []
    pairs = [(question, document.page_content) for document, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in ranked[:keep]]
