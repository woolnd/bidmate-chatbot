"""1단계 질문 다듬기 테스트. OpenAI를 부르지 않고 가짜 모델을 쓴다.

가짜 모델은 RunnableLambda로 만든다. 받은 메시지를 기록하고, 정해 둔 RewriteResult를 돌려준다.
예시는 지어낸 기관명으로 쓴다 (이 파일은 Git에 올라간다).
"""

from langchain_core.runnables import RunnableLambda

from bidmate.generation import question_rewriter
from bidmate.generation.question_rewriter import rewrite_question
from bidmate.schemas import RewriteResult

HISTORY = [("가나다공단 이러닝 사업 알려줘", "가나다공단 통합 이러닝 플랫폼 구축 사업은 …")]
REWRITTEN = "가나다공단 통합 이러닝 플랫폼 구축 사업 예산은?"


def fake_model(rewritten_question: str, route: str = "rfp", received: list | None = None):
    """정해 둔 RewriteResult를 돌려주는 가짜 모델. received에 받은 메시지를 모은다."""

    def answer(messages):
        if received is not None:
            received.append(messages)
        return RewriteResult(rewritten_question=rewritten_question, route=route)

    return RunnableLambda(answer)


def test_rewrites_with_history_and_strips_quotes():
    # AI가 따옴표·줄바꿈을 붙여 돌려줘도 지운다
    result = rewrite_question("그럼 예산은?", HISTORY, fake_model(f'"{REWRITTEN}"\n'))
    assert result == RewriteResult(rewritten_question=REWRITTEN, route="rfp")


def test_first_question_also_calls_ai_and_returns_route():
    # 대화 기록이 없어도 AI를 불러 질문 종류를 고른다. 기록 자리에는 "(없음)"이 들어간다
    received = []
    result = rewrite_question("안녕?", [], fake_model("안녕?", "chat", received))
    assert result.route == "chat"
    assert "대화 기록:\n(없음)" in received[0][1].content


def test_passes_only_recent_turns(monkeypatch):
    # 대화 기록이 7번이어도 AI에게는 최근 5번만 넘긴다
    passed_histories = []
    original_build = question_rewriter.build_rewrite_messages

    def recording_build(question, recent_history):
        passed_histories.append(recent_history)
        return original_build(question, recent_history)

    monkeypatch.setattr(question_rewriter, "build_rewrite_messages", recording_build)
    seven_turns = [(f"질문{i}", f"답변{i}") for i in range(7)]
    rewrite_question("그럼 예산은?", seven_turns, fake_model(REWRITTEN))

    assert passed_histories == [seven_turns[-5:]]


def test_empty_response_falls_back_to_question():
    # AI가 빈 질문을 주면 원래 질문을 돌려준다. 질문 종류는 AI가 고른 그대로
    result = rewrite_question("그럼 예산은?", HISTORY, fake_model("  \n"))
    assert result == RewriteResult(rewritten_question="그럼 예산은?", route="rfp")


def test_disabled_returns_question_unchanged(monkeypatch):
    # USE_QUESTION_REWRITE를 끄면 AI를 부르지 않고 RFP 질문으로 넘긴다 (11/23 기능 끄기 실험용)
    monkeypatch.setattr(question_rewriter, "USE_QUESTION_REWRITE", False)
    received = []
    result = rewrite_question("그럼 예산은?", HISTORY, fake_model(REWRITTEN, received=received))
    assert result == RewriteResult(rewritten_question="그럼 예산은?", route="rfp")
    assert received == []
