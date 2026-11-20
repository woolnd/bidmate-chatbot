"""4단계 LangGraph 연결 테스트. OpenAI를 부르지 않는다.

graph.py가 부르는 단계 함수들을 monkeypatch로 가짜로 바꿔서,
"순서대로 불리고, 앞 단계 결과가 다음 단계로 넘어가는지"만 확인한다.
"""

from langchain_core.documents import Document

from bidmate.pipeline import graph
from bidmate.pipeline.graph import run_chatbot
from bidmate.schemas import AnswerResult, ProjectInfo, RewriteResult, SimilarProject

FAKE_CHUNK = (Document(page_content="[가나다공단 / 이러닝] 예산 1억", metadata={}), 0.1)
FAKE_ANSWER = AnswerResult(
    answer="1억입니다.",
    source_project_names=["이러닝"],
    has_evidence=True,
    is_rfp_question=True,
    next_questions=[],
)


def test_four_steps_run_in_order_and_pass_values(monkeypatch):
    # 네 단계가 1 → 2 → 3 → 4 순서로 불리고, 앞 단계가 만든 값이 다음 단계로 넘어간다
    called_steps = []

    def fake_rewrite(question, chat_history):
        called_steps.append(("rewrite", question, chat_history))
        return RewriteResult(rewritten_question="가나다공단 이러닝 예산은?", route="rfp")

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
    assert state["route"] == "rfp"
    assert state["project_ids"] == ["가나다공단_이러닝.hwp"]
    assert state["chunks"] == [FAKE_CHUNK]
    assert state["answer_result"] == FAKE_ANSWER


def test_empty_search_gives_no_evidence_answer(monkeypatch):
    # 2·3단계가 빈 목록이면 4단계가 AI를 부르지 않고 "찾은 문서에 없습니다" (4단계는 진짜 함수)
    monkeypatch.setattr(
        graph,
        "rewrite_question",
        lambda question, chat_history: RewriteResult(rewritten_question=question, route="rfp"),
    )
    monkeypatch.setattr(graph, "get_projects", lambda: [])
    monkeypatch.setattr(graph, "find_project_ids", lambda question, projects: [])
    monkeypatch.setattr(graph, "search_chunks", lambda question, project_ids, k: [])

    state = run_chatbot("예산은?", [])

    assert state["rewritten_question"] == "예산은?"
    assert state["answer_result"].answer == "찾은 문서에 없습니다."
    assert state["answer_result"].has_evidence is False


def test_chat_route_skips_search_and_answers_greeting(monkeypatch):
    # 1단계가 chat을 고르면 2·3·4단계를 건너뛰고 고정 인사 답변을 준다 (AI 호출 없음)
    called_steps = []
    monkeypatch.setattr(
        graph,
        "rewrite_question",
        lambda question, chat_history: RewriteResult(rewritten_question=question, route="chat"),
    )
    monkeypatch.setattr(graph, "find_project_ids", lambda *args: called_steps.append("find"))
    monkeypatch.setattr(graph, "search_chunks", lambda *args: called_steps.append("search"))
    monkeypatch.setattr(graph, "generate_answer", lambda *args: called_steps.append("answer"))

    state = run_chatbot("안녕?", [])

    assert called_steps == []
    assert state["route"] == "chat"
    assert state["answer_result"].answer == graph.CHAT_ANSWER
    assert state["answer_result"].is_rfp_question is False
    assert state["answer_result"].next_questions == []
    assert state["chunks"] == []


# ── 4단계 뒤 다시 찾기 갈림길 (#52) ──

NO_EVIDENCE = AnswerResult(
    answer="찾은 문서에 없습니다.",
    source_project_names=[],
    has_evidence=False,
    is_rfp_question=True,
    next_questions=[],
)


def run_with_answers(monkeypatch, answers, chunk_count=100):
    """가짜 단계로 그래프를 돌린다. 4단계는 answers를 차례로 답한다. 검색에 쓴 k 목록도 돌려준다.

    chunk_count: 문서에 있는 조각 수. 검색은 min(k, chunk_count)개를 돌려준다
    """
    searched_k = []
    answer_queue = list(answers)

    def fake_search(question, project_ids, k):
        searched_k.append(k)
        return [FAKE_CHUNK] * min(k, chunk_count)

    monkeypatch.setattr(
        graph,
        "rewrite_question",
        lambda question, chat_history: RewriteResult(rewritten_question=question, route="rfp"),
    )
    monkeypatch.setattr(graph, "get_projects", lambda: [])
    monkeypatch.setattr(graph, "find_project_ids", lambda question, projects: ["a.hwp"])
    monkeypatch.setattr(graph, "search_chunks", fake_search)
    monkeypatch.setattr(graph, "generate_answer", lambda *args: answer_queue.pop(0))
    return run_chatbot("예산은?", []), searched_k


def test_no_evidence_retries_once_with_more_chunks(monkeypatch):
    # 근거 없음 → RETRY_TOP_K개로 한 번만 다시 찾는다. 두 번째도 근거 없음이어도 끝난다
    state, searched_k = run_with_answers(monkeypatch, [NO_EVIDENCE, NO_EVIDENCE])
    assert searched_k == [graph.TOP_K, graph.RETRY_TOP_K]
    assert len(state["chunks"]) == graph.RETRY_TOP_K


def test_retry_answer_replaces_first_answer(monkeypatch):
    state, _ = run_with_answers(monkeypatch, [NO_EVIDENCE, FAKE_ANSWER])
    assert state["answer_result"] == FAKE_ANSWER


def test_evidence_found_does_not_retry(monkeypatch):
    _, searched_k = run_with_answers(monkeypatch, [FAKE_ANSWER])
    assert searched_k == [graph.TOP_K]


def test_few_chunks_does_not_retry(monkeypatch):
    # 처음부터 TOP_K개를 못 채웠으면 그 문서에 조각이 더 없으니 다시 찾지 않는다
    _, searched_k = run_with_answers(monkeypatch, [NO_EVIDENCE], chunk_count=2)
    assert searched_k == [graph.TOP_K]


# ── catalog 길 (#60) ──


def test_catalog_route_skips_search_and_answers_with_catalog(monkeypatch):
    # 1단계가 catalog를 고르면 2·3단계를 건너뛰고, 목록표 조각 1개로 4단계가 답한다
    called_steps = []
    catalog_document = Document(page_content="목록표", metadata={"project_name": "사업 목록표"})
    monkeypatch.setattr(
        graph,
        "rewrite_question",
        lambda question, chat_history: RewriteResult(rewritten_question=question, route="catalog"),
    )
    monkeypatch.setattr(graph, "find_project_ids", lambda *args: called_steps.append("find"))
    monkeypatch.setattr(graph, "search_chunks", lambda *args: called_steps.append("search"))
    monkeypatch.setattr(graph, "build_catalog_document", lambda: catalog_document)

    def fake_answer(question, chunks, chat_history):
        called_steps.append(("answer", question, chunks))
        return FAKE_ANSWER

    monkeypatch.setattr(graph, "generate_answer", fake_answer)

    state = run_chatbot("예산이 가장 큰 사업은?", [])

    assert called_steps == [("answer", "예산이 가장 큰 사업은?", [(catalog_document, 0.0)])]
    assert state["route"] == "catalog"
    assert state["project_ids"] == []
    assert state["answer_result"] == FAKE_ANSWER


# ── similar 길 (#64) ──


def run_similar(monkeypatch, project_ids, similar):
    """similar 질문으로 그래프를 돌린다. 2단계는 project_ids, 비슷한 사업은 similar를 돌려준다."""
    projects = [
        ProjectInfo("a.hwp", "가나다공단", "이러닝", None, None),
        ProjectInfo("b.hwp", "라마대학교", "학사 시스템", None, None),
    ]
    called_steps = []
    monkeypatch.setattr(
        graph,
        "rewrite_question",
        lambda question, chat_history: RewriteResult(rewritten_question=question, route="similar"),
    )
    monkeypatch.setattr(graph, "get_projects", lambda: projects)
    monkeypatch.setattr(graph, "find_project_ids", lambda question, projects: project_ids)
    monkeypatch.setattr(graph, "search_chunks", lambda *args: called_steps.append("search"))
    monkeypatch.setattr(graph, "generate_answer", lambda *args: called_steps.append("answer"))
    monkeypatch.setattr(graph, "find_similar_projects", lambda project_id: similar)
    state = run_chatbot("가나다공단 이러닝과 비슷한 사업은?", [])
    assert called_steps == []  # 3·4단계(검색·AI 답변)를 부르지 않는다
    return state


def test_similar_route_lists_similar_projects(monkeypatch):
    similar = [SimilarProject("b.hwp", "라마대학교", "학사 시스템", 0.52)]
    state = run_similar(monkeypatch, ["a.hwp"], similar)
    assert state["similar_projects"] == similar
    assert "라마대학교 학사 시스템 (유사도 52%)" in state["answer_result"].answer
    assert state["answer_result"].source_project_names == ["이러닝"]


def test_similar_route_says_when_nothing_is_clearly_similar(monkeypatch):
    similar = [SimilarProject("b.hwp", "라마대학교", "학사 시스템", 0.1)]
    state = run_similar(monkeypatch, ["a.hwp"], similar)
    assert "뚜렷하게 비슷한 사업은 없어요" in state["answer_result"].answer


def test_similar_route_asks_which_project_when_several(monkeypatch):
    state = run_similar(monkeypatch, ["a.hwp", "b.hwp"], [])
    assert state["answer_result"].answer.startswith("어느 사업과 비슷한 사업을 찾을까요?")
    assert "- 가나다공단 이러닝" in state["answer_result"].answer
    assert len(state["answer_result"].next_questions) == 2


def test_similar_route_asks_for_name_when_none(monkeypatch):
    state = run_similar(monkeypatch, [], [])
    assert "기관명과 사업명을 함께 알려 주세요" in state["answer_result"].answer
