"""1단계: 후속 질문을 혼자 읽어도 뜻이 통하는 질문으로 바꾼다. 담당: 재웅

예: (국민연금공단 이러닝 사업 이야기 중) "그럼 예산은?"
    → "국민연금공단 이러닝 사업 예산은?"
2단계(어느 문서인지 고르기)가 기관명을 찾을 수 있게 하는 게 목적이다.

받는 것: 질문, 대화 기록
넘기는 것: 다듬은 질문
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from bidmate.config import (
    ANSWER_MODEL_NAME,
    MAX_HISTORY_TURNS,
    REWRITE_MAX_TOKENS,
    REWRITE_REASONING_EFFORT,
    USE_QUESTION_REWRITE,
)
from bidmate.generation.prompts import build_rewrite_messages
from bidmate.schemas import ChatHistory


def make_rewrite_model() -> ChatOpenAI:
    """1단계용 AI 모델을 만든다. 간단한 일이라 생각을 최소로 해서 빠르고 싸게 부른다.

    받는 것: 없음 (설정값은 config.py에서 가져온다)
    넘기는 것: ChatOpenAI 모델
    """

    model = ChatOpenAI(
        model=ANSWER_MODEL_NAME,
        reasoning_effort=REWRITE_REASONING_EFFORT,
        max_completion_tokens=REWRITE_MAX_TOKENS,
    )

    return model


def rewrite_question(
    question: str, chat_history: ChatHistory, model: BaseChatModel | None = None
) -> str:
    """질문을 혼자 읽어도 뜻이 통하게 바꾼다. 대화 기록이 없으면 그대로 돌려준다.

    받는 것: 질문, 대화 기록 [(질문, 답변), ...],
             model (테스트에서 가짜 모델을 넣는다. 비우면 진짜 모델을 만든다)
    넘기는 것: 다듬은 질문
    """

    if not USE_QUESTION_REWRITE or len(chat_history) == 0:
        return question

    recent_history = chat_history[-MAX_HISTORY_TURNS:]
    messages = build_rewrite_messages(question, recent_history)

    if model is None:
        model = make_rewrite_model()

    response = model.invoke(messages)
    rewritten_question = response.content.strip().strip("\"'")

    if len(rewritten_question) == 0:
        return question
    else:
        return rewritten_question
