"""4단계 LangGraph 연결 테스트. OpenAI를 부르지 않는다.

graph.py가 부르는 단계 함수들을 monkeypatch로 가짜로 바꿔서,
"순서대로 불리고, 앞 단계 결과가 다음 단계로 넘어가는지"만 확인한다.
"""

from langchain_core.documents import Document

from bidmate.pipeline import graph
from bidmate.pipeline.graph import run_chatbot
from bidmate.schemas import AnswerResult

FAKE_CHUNK = (Document(page_content="[가나다공단 / 이러닝] 예산 1억", metadata={}), 0.1)
FAKE_ANSWER = AnswerResult(answer="1억입니다.", source_project_names=["이러닝"], has_evidence=True)


def test_four_steps_run_in_order_and_pass_values(monkeypatch):
    # 네 단계가 1 → 2 → 3 → 4 순서로 불리고, 앞 단계가 만든 값이 다음 단계로 넘어간다
    called_steps = []

    def fake_rewrite(question, chat_history):
        called_steps.append(("rewrite", question, chat_history))
        return "가나다공단 이러닝 예산은?"

    def fake_find(question, projects):
        called_steps.append(("find", question))
        return ["가나다공단_이러닝.hwp"]

    def fake_search(question, project_ids, k):
        called_steps.append(("search", question, project_ids))
        return [FAKE_CHUNK]

    def fake_answer(question, chunks, chat_history):
        called_steps.append(("answer", question, chunks, chat_history))
        return FAKE_ANSWER

    monkeypatch.setattr(graph, "rewrite_question", fake_rewrite)
    monkeypatch.setattr(graph, "get_projects", lambda: [])
    monkeypatch.setattr(graph, "find_project_ids", fake_find)
    monkeypatch.setattr(graph, "search_chunks", fake_search)
    monkeypatch.setattr(graph, "generate_answer", fake_answer)

    history = [("가나다공단 이러닝 알려줘", "가나다공단 이러닝 사업은 …")]
    state = run_chatbot("그럼 예산은?", history)

    assert called_steps == [
        ("rewrite", "그럼 예산은?", history),
        ("find", "가나다공단 이러닝 예산은?"),
        ("search", "가나다공단 이러닝 예산은?", ["가나다공단_이러닝.hwp"]),
        ("answer", "가나다공단 이러닝 예산은?", [FAKE_CHUNK], history),
    ]
    # 끝난 state에 모든 칸이 차 있다 (로그·채점용)
    assert state["rewritten_question"] == "가나다공단 이러닝 예산은?"
    assert state["project_ids"] == ["가나다공단_이러닝.hwp"]
    assert state["chunks"] == [FAKE_CHUNK]
    assert state["answer_result"] == FAKE_ANSWER


def test_empty_search_gives_no_evidence_answer(monkeypatch):
    # 2·3단계가 빈 목록이면 4단계가 AI를 부르지 않고 "찾은 문서에 없습니다"
    # (대화 기록이 없으면 1단계도 AI를 부르지 않는다. 4단계는 진짜 함수 그대로)
    monkeypatch.setattr(graph, "get_projects", lambda: [])
    monkeypatch.setattr(graph, "find_project_ids", lambda question, projects: [])
    monkeypatch.setattr(graph, "search_chunks", lambda question, project_ids, k: [])

    state = run_chatbot("예산은?", [])

    assert state["rewritten_question"] == "예산은?"
    assert state["answer_result"].answer == "찾은 문서에 없습니다."
    assert state["answer_result"].has_evidence is False
