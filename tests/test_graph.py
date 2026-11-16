"""4단계 LangGraph 연결 테스트. OpenAI를 부르지 않는다.

graph.py가 부르는 단계 함수들을 monkeypatch로 가짜로 바꿔서,
"순서대로 불리고, 앞 단계 결과가 다음 단계로 넘어가는지"만 확인한다.
"""

from langchain_core.documents import Document

from bidmate.pipeline import graph
from bidmate.pipeline.graph import run_chatbot
from bidmate.schemas import AnswerResult, ProjectInfo

FAKE_CHUNK = (Document(page_content="[가나다공단 / 이러닝] 예산 1억", metadata={}), 0.1)
FAKE_ANSWER = AnswerResult(
    answer="1억입니다.", source_project_names=["이러닝"], has_evidence=True, is_rfp_question=True
)


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


def test_catalog_miss_skips_search_but_ambiguous_question_keeps_fallback(monkeypatch):
    projects = [ProjectInfo("unrelated", "가나다기관", "문서관리 운영", None, None)]
    monkeypatch.setattr(graph, "get_projects", lambda: projects)
    searches = []

    def fake_search(question, project_ids, k):
        searches.append((question, project_ids))
        return []

    monkeypatch.setattr(graph, "search_chunks", fake_search)
    for question in (
        "가나다기관 위성발사 사업 예산",
        "발주처가 가상재단인 문서관리 운영 예산",
    ):
        state = run_chatbot(question, [])
        assert state["project_ids"] == []
        assert state["chunks"] == []
        assert not state["answer_result"].has_evidence
    assert searches == []

    for question in (
        "참여 자격 알려줘",
        "전기공사 참여 자격은?",
        "가나다기관의 어떤 사업이 있나요?",
    ):
        state = run_chatbot(question, [])
        assert not state["catalog_miss"]
    assert searches[0] == ("참여 자격 알려줘", [])


def test_comparison_passes_both_exact_document_ids_to_search(monkeypatch):
    projects = [
        ProjectInfo("  포털 (1).hwp", "가나다기관", "문서관리시스템 구축", None, None),
        ProjectInfo("정산 [2].pdf", "라마바기관", "보조금 정산 플랫폼 개발", None, None),
        ProjectInfo("unrelated", "가나다기관", "보안관제 운영", None, None),
    ]
    monkeypatch.setattr(graph, "get_projects", lambda: projects)
    calls = []

    def fake_search(question, project_ids, k):
        calls.append(project_ids)
        return []

    monkeypatch.setattr(graph, "search_chunks", fake_search)
    state = run_chatbot("가나다기관 문서관리와 라마바기관 보조금 정산 각각 예산", [])

    assert calls == [["  포털 (1).hwp", "정산 [2].pdf"]]
    assert state["project_ids"] == calls[0]
