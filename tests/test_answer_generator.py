"""4단계 답변 만들기 테스트. OpenAI를 부르지 않고 가짜 모델을 쓴다.

진짜 모델은 with_structured_output을 붙여 AnswerResult를 돌려주므로,
가짜 모델도 RunnableLambda로 "메시지를 받으면 AnswerResult를 돌려주는 것"을 만든다.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from bidmate.generation.answer_generator import NO_EVIDENCE_ANSWER, generate_answer
from bidmate.generation.prompts import ANSWER_SYSTEM_PROMPT
from bidmate.schemas import AnswerResult
from tests.fake_chunks import make_fake_chunks

FAKE_RESULT = AnswerResult(
    answer="사업 예산은 150,000,000원입니다.",
    source_project_names=["통합 이러닝 플랫폼 구축"],
    has_evidence=True,
)


def make_recording_model(received_messages: list):
    """받은 메시지를 received_messages에 적어 두고 FAKE_RESULT를 돌려주는 가짜 모델."""

    def fake_invoke(messages):
        received_messages.append(messages)
        return FAKE_RESULT

    return RunnableLambda(fake_invoke)


def make_chunks(count: int):
    """지어낸 조각을 (Document, 점수) 모양으로 count개 만든다."""
    return [(document, 0.1) for document in make_fake_chunks()[:count]]


def test_no_chunks_returns_no_evidence_without_calling_model():
    # 조각이 없으면 AI를 부르지 않고 "찾은 문서에 없다"를 돌려준다
    received_messages = []
    result = generate_answer("예산은?", [], [], make_recording_model(received_messages))

    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.source_project_names == []
    assert result.has_evidence is False
    assert received_messages == []


def test_returns_model_result_as_is():
    # 구조화된 출력 결과(AnswerResult)를 다듬지 않고 그대로 돌려준다
    result = generate_answer("예산은?", make_chunks(2), [], make_recording_model([]))
    assert result == FAKE_RESULT


def test_message_order_fixed_part_first():
    # 고정 부분(시스템 메시지) → 바뀌는 부분(대화 기록 → 찾은 조각 → 질문) 순서 (프롬프트 캐싱)
    received_messages = []
    history = [("가나다공단 이러닝 알려줘", "가나다공단 통합 이러닝 …")]
    generate_answer("예산은?", make_chunks(2), history, make_recording_model(received_messages))

    system_message, human_message = received_messages[0]
    assert isinstance(system_message, SystemMessage)
    assert system_message.content == ANSWER_SYSTEM_PROMPT
    assert isinstance(human_message, HumanMessage)

    text = human_message.content
    assert text.index("대화 기록:") < text.index("찾은 조각:") < text.index("질문: 예산은?")
    # 조각마다 사업명이 보여야 AI가 출처를 고를 수 있다
    assert "[1] 사업명: 통합 이러닝 플랫폼 구축" in text
    assert "[2] 사업명: 통합 이러닝 플랫폼 구축" in text


def test_passes_only_recent_turns():
    # 대화 기록이 5번이어도 AI에게는 최근 3번만 넘긴다
    received_messages = []
    five_turns = [(f"이전질문{i}", f"이전답변{i}") for i in range(5)]
    generate_answer("예산은?", make_chunks(1), five_turns, make_recording_model(received_messages))

    text = received_messages[0][1].content
    assert "이전질문0" not in text
    assert "이전질문1" not in text
    assert all(f"이전질문{i}" in text for i in (2, 3, 4))
