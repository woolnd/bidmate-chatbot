"""1단계: 질문을 혼자 읽어도 뜻이 통하게 바꾸고, 질문 종류(rfp/chat)를 고른다. 담당: 재웅

예: (국민연금공단 이러닝 사업 이야기 중) "그럼 예산은?"
    → "국민연금공단 이러닝 사업 예산은?"
2단계(어느 문서인지 고르기)가 기관명을 찾을 수 있게 하는 게 목적이다.
질문 종류는 그래프가 길을 고를 때 쓴다 (chat이면 2·3·4단계를 건너뛴다).

받는 것: 질문, 대화 기록
넘기는 것: RewriteResult (다듬은 질문, 질문 종류)
"""

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from bidmate.config import (
    ANSWER_MODEL_NAME,
    MAX_HISTORY_TURNS,
    OPENAI_TIMEOUT_SECONDS,
    REWRITE_MAX_TOKENS,
    REWRITE_REASONING_EFFORT,
    USE_QUESTION_REWRITE,
)
from bidmate.generation.prompts import build_rewrite_messages
from bidmate.schemas import ChatHistory, RewriteResult


def make_rewrite_model() -> Runnable:
    """1단계용 AI 모델을 만든다. RewriteResult 양식을 채워서 돌려주게 한다(구조화된 출력).

    간단한 일이라 생각을 최소로 해서 빠르고 싸게 부른다.

    받는 것: 없음 (설정값은 config.py에서 가져온다)
    넘기는 것: RewriteResult를 돌려주는 모델
    """

    model = ChatOpenAI(
        model=ANSWER_MODEL_NAME,
        reasoning_effort=REWRITE_REASONING_EFFORT,
        max_completion_tokens=REWRITE_MAX_TOKENS,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )

    return model.with_structured_output(RewriteResult)


def rewrite_question(
    question: str, chat_history: ChatHistory, model: Runnable | None = None
) -> RewriteResult:
    """질문을 혼자 읽어도 뜻이 통하게 바꾸고, 질문 종류(rfp/chat)를 고른다.

    받는 것: 질문, 대화 기록 [(질문, 답변), ...],
             model (테스트에서 가짜 모델을 넣는다. 비우면 진짜 모델을 만든다)
    넘기는 것: RewriteResult (다듬은 질문, 질문 종류)
    첫 질문도 AI를 부른다. 인사인지 RFP 질문인지 골라야 해서다.
    """

    # 1. 기능이 꺼져 있으면 그대로 RFP 질문으로 넘긴다 (AI를 부르지 않는다. 11/23 기능 끄기 실험용)
    if not USE_QUESTION_REWRITE:
        return RewriteResult(rewritten_question=question, route="rfp")

    # 2. 대화 기록은 최근 MAX_HISTORY_TURNS번(5번)만 넘긴다
    recent_history = chat_history[-MAX_HISTORY_TURNS:]

    # 3. 메시지 만들기: [시스템 메시지(고정), 사람 메시지(대화 기록 + 새 질문)]
    messages = build_rewrite_messages(question, recent_history)

    # 4. 모델을 안 넣었으면 진짜 모델을 만든다 (테스트에서는 가짜 모델을 넣는다)
    if model is None:
        model = make_rewrite_model()

    # 5. AI 부르기 (RewriteResult 양식으로 돌아온다)
    result: RewriteResult = model.invoke(messages)

    # 6. 다듬은 질문의 앞뒤 공백·따옴표를 지운다
    #    비었으면 원래 질문으로 되돌린다 (검색이 빈 질문으로 돌지 않게)
    rewritten_question = result.rewritten_question.strip().strip("\"'")
    if len(rewritten_question) == 0:
        rewritten_question = question

    return RewriteResult(rewritten_question=rewritten_question, route=result.route)
