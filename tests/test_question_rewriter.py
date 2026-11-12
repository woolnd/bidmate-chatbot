"""1단계 질문 다듬기 테스트. OpenAI를 부르지 않고 가짜 모델(FakeListChatModel)을 쓴다.

가짜 모델은 responses에 넣은 글을 차례로 돌려준다.
예시는 지어낸 기관명으로 쓴다 (이 파일은 Git에 올라간다).
"""

from langchain_core.language_models import FakeListChatModel

from bidmate.generation import question_rewriter
from bidmate.generation.question_rewriter import rewrite_question

HISTORY = [("가나다공단 이러닝 사업 알려줘", "가나다공단 통합 이러닝 플랫폼 구축 사업은 …")]
REWRITTEN = "가나다공단 통합 이러닝 플랫폼 구축 사업 예산은?"


def test_no_history_returns_question_unchanged():
    # 대화 기록이 없으면 모델을 만들지도 부르지도 않는다 (model=None이어도 API 호출 없음)
    assert rewrite_question("그럼 예산은?", []) == "그럼 예산은?"


def test_rewrites_with_history_and_strips_quotes():
    # AI가 따옴표·줄바꿈을 붙여 돌려줘도 지운다
    model = FakeListChatModel(responses=[f'"{REWRITTEN}"\n'])
    assert rewrite_question("그럼 예산은?", HISTORY, model) == REWRITTEN


def test_passes_only_recent_turns(monkeypatch):
    # 대화 기록이 7번이어도 AI에게는 최근 5번만 넘긴다
    passed_histories = []
    original_build = question_rewriter.build_rewrite_messages

    def recording_build(question, recent_history):
        passed_histories.append(recent_history)
        return original_build(question, recent_history)

    monkeypatch.setattr(question_rewriter, "build_rewrite_messages", recording_build)
    seven_turns = [(f"질문{i}", f"답변{i}") for i in range(7)]
    rewrite_question("그럼 예산은?", seven_turns, FakeListChatModel(responses=[REWRITTEN]))

    assert passed_histories == [seven_turns[-5:]]


def test_empty_response_falls_back_to_question():
    # AI가 빈 답을 주면 원래 질문을 돌려준다
    model = FakeListChatModel(responses=["  \n"])
    assert rewrite_question("그럼 예산은?", HISTORY, model) == "그럼 예산은?"


def test_disabled_returns_question_unchanged(monkeypatch):
    # USE_QUESTION_REWRITE를 끄면 AI를 부르지 않는다 (11/23 기능 끄기 실험용)
    monkeypatch.setattr(question_rewriter, "USE_QUESTION_REWRITE", False)
    model = FakeListChatModel(responses=[REWRITTEN])
    assert rewrite_question("그럼 예산은?", HISTORY, model) == "그럼 예산은?"
