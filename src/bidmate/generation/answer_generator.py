"""4단계: 찾은 조각만 보고 답변을 만든다. 담당: 재웅

답변은 AnswerResult 형식(답변 / 출처 사업명 / 근거 있음 여부)으로 받는다.
형식이 고정돼 있어서 화면 표시와 평가 자동 채점에 그대로 쓸 수 있다.
조각에 근거가 없으면 모른다고 답한다 (has_evidence=False).
gpt-5-mini는 temperature를 바꿀 수 없어서 최대 800토큰과 reasoning_effort로 조절한다.

받는 것: 다듬은 질문, 찾은 조각 (Document, 점수) 목록, 대화 기록
넘기는 것: AnswerResult (답변 / 출처 사업명 / 근거 있음 여부)
"""

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from bidmate.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_MODEL_NAME,
    ANSWER_REASONING_EFFORT,
    MAX_HISTORY_TURNS,
    OPENAI_TIMEOUT_SECONDS,
)
from bidmate.generation.prompts import build_answer_messages
from bidmate.schemas import AnswerResult, ChatHistory

# 조각이 없을 때 AI를 부르지 않고 돌려줄 답. 시스템 프롬프트 규칙의 문구와 같게 맞춘다
NO_EVIDENCE_ANSWER = "찾은 문서에 없습니다."


def make_answer_model() -> Runnable:
    """4단계용 AI 모델을 만든다. AnswerResult 양식을 채워서 돌려주게 한다(구조화된 출력).

    받는 것: 없음 (설정값은 config.py에서 가져온다)
    넘기는 것: invoke(메시지)하면 AnswerResult를 돌려주는 모델
    """
    # 1. ChatOpenAI 만들기 (1단계 make_rewrite_model과 같은 모양, 설정값만 ANSWER_…)
    model = ChatOpenAI(
        model=ANSWER_MODEL_NAME,
        reasoning_effort=ANSWER_REASONING_EFFORT,
        max_completion_tokens=ANSWER_MAX_TOKENS,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )

    # 2. 구조화된 출력을 붙인다: invoke 결과가 AIMessage가 아니라 AnswerResult로 나온다
    return model.with_structured_output(AnswerResult)


def generate_answer(
    question: str,
    chunks: list[tuple[Document, float]],
    chat_history: ChatHistory,
    model: Runnable | None = None,
) -> AnswerResult:
    """찾은 조각만 보고 질문에 답한다. 조각이 없으면 AI를 부르지 않고 "찾은 문서에 없다"고 답한다.

    받는 것: 다듬은 질문, 찾은 조각 [(Document, 점수), ...], 대화 기록 [(질문, 답변), ...],
             model (테스트에서 가짜 모델을 넣는다. 비우면 진짜 모델을 만든다)
    넘기는 것: AnswerResult (답변 / 출처 사업명 / 근거 있음 여부)
    """

    # 1. 찾은 조각이 없으면 AI를 부르지 않고 "찾은 문서에 없다"고 돌려준다 (비용 0원)
    if len(chunks) == 0:
        return AnswerResult(answer=NO_EVIDENCE_ANSWER, source_project_names=[], has_evidence=False)

    # 2. 대화 기록은 최근 MAX_HISTORY_TURNS번(3번)만 넘긴다
    recent_history = chat_history[-MAX_HISTORY_TURNS:]

    # 3. 메시지 만들기: [시스템 메시지(고정), 사람 메시지(대화 기록 → 찾은 조각 → 질문)]
    messages = build_answer_messages(question, chunks, recent_history)

    # 4. 모델을 안 넣었으면 진짜 모델을 만든다 (테스트에서는 가짜 모델을 넣는다)
    if model is None:
        model = make_answer_model()

    # 5. AI 부르기: 결과가 이미 AnswerResult라서 1단계처럼 다듬지 않고 그대로 돌려준다
    answer_result = model.invoke(messages)
    return answer_result
