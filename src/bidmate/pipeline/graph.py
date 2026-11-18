"""4단계를 LangGraph로 연결한다. 담당: 재웅

질문 다듬기 → 어느 문서인지 고르기 → 문서에서 관련 부분 찾기 → 답변 만들기
각 단계(노드)는 ChatState에서 필요한 값을 읽고, 자기가 만든 값만 돌려준다.
LangGraph는 돌려준 값을 상태에 합쳐서 다음 단계로 넘긴다.

갈림길이 두 개 있다. AI가 판단한 값을 보고 그래프(코드)가 길을 고른다.
    rewrite ─┬─ rfp  → find_projects → search → answer ─┬─ 근거 있음 / 조각 덜 참 → END
             │                                          └─ 근거 없음 → retry → END
             └─ chat → chat (고정 인사 답변, AI 호출 없음) → END
    1. 1단계 뒤: 1단계 AI가 고른 질문 종류(route)
    2. 4단계 뒤: 4단계 AI가 "근거 없음"이라 답하면 조각을 더 많이(RETRY_TOP_K) 찾아 다시 답한다
       retry 뒤에는 바로 END라서 다시 찾기는 최대 1번이다 (#52)
"""

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from bidmate.config import PROJECT_LIST_PATH, RETRY_TOP_K, TOP_K
from bidmate.data.metadata import load_project_list
from bidmate.generation.answer_generator import generate_answer
from bidmate.generation.question_rewriter import rewrite_question
from bidmate.retrieval.project_finder import find_project_ids
from bidmate.retrieval.vector_store import search_chunks
from bidmate.schemas import AnswerResult, ChatHistory, ChatState, ProjectInfo


@lru_cache(maxsize=1)
def get_projects() -> list[ProjectInfo]:
    """목록표(data_list.csv)를 한 번만 읽어 두고 계속 같은 목록을 돌려준다.

    받는 것: 없음
    넘기는 것: ProjectInfo 목록 (2단계가 질문과 비교한다)
    @lru_cache: 처음 부를 때만 파일을 읽고, 그다음부터는 기억해 둔 값을 돌려준다.
    """
    return load_project_list(PROJECT_LIST_PATH)


# 인사·잡담(chat)에 돌려주는 고정 답변. AI를 부르지 않아 0원, 바로 답한다
CHAT_ANSWER = (
    "안녕하세요! 입찰메이트 RFP 분석 도우미예요. 공공기관 RFP 100개를 검색해 출처와 함께 답해요.\n"
    "- 사업 예산이나 입찰 마감일\n"
    "- 주요 요구사항, 참가 자격\n"
    "- 제출 서류와 제출 방법\n"
    "기관명이나 사업명을 넣어 물어보세요."
)


# ── 노드: 상태에서 필요한 값을 읽고, 자기가 만든 칸만 dict로 돌려준다 ──


def rewrite_node(state: ChatState) -> ChatState:
    """1단계 질문 다듬기. 읽는 것: question, chat_history → 채우는 것: rewritten_question, route"""
    rewrite_result = rewrite_question(state["question"], state["chat_history"])
    return {"rewritten_question": rewrite_result.rewritten_question, "route": rewrite_result.route}


def route_question(state: ChatState) -> Literal["find_projects", "chat"]:
    """1단계 뒤 갈림길. 읽는 것: route → 다음 노드 이름 (rfp면 2단계, chat이면 인사 답변)"""
    if state["route"] == "chat":
        return "chat"
    return "find_projects"


def chat_node(state: ChatState) -> ChatState:
    """인사·잡담 답변. 2·3·4단계를 건너뛰고 고정 답변을 채운다. 채우는 것: chunks, answer_result"""
    answer_result = AnswerResult(
        answer=CHAT_ANSWER,
        source_project_names=[],
        has_evidence=False,
        is_rfp_question=False,
        next_questions=[],
    )
    return {"project_ids": [], "chunks": [], "answer_result": answer_result}


def find_projects_node(state: ChatState) -> ChatState:
    """2단계 문서 고르기. 읽는 것: rewritten_question → 채우는 것: project_ids"""
    project_ids = find_project_ids(state["rewritten_question"], get_projects())
    return {"project_ids": project_ids}


def search_node(state: ChatState) -> ChatState:
    """3단계 조각 찾기. 읽는 것: rewritten_question, project_ids → 채우는 것: chunks"""
    chunks = search_chunks(state["rewritten_question"], state["project_ids"], TOP_K)
    return {"chunks": chunks}


def answer_node(state: ChatState) -> ChatState:
    """4단계 답변. 읽는 것: rewritten_question, chunks, chat_history → 채우는 것: answer_result"""
    answer_result = generate_answer(
        state["rewritten_question"], state["chunks"], state["chat_history"]
    )
    return {"answer_result": answer_result}


def should_retry(state: ChatState) -> Literal["retry", "__end__"]:
    """4단계 뒤 갈림길. 읽는 것: answer_result, chunks → "retry"(다시 찾기) 또는 END

    다시 찾는 조건 (둘 다 맞을 때):
        - 4단계 AI가 근거 없음이라고 답했다 (has_evidence=False)
        - 처음 찾은 조각이 TOP_K개 꽉 찼다 (덜 찼으면 그 문서에 조각이 더 없어서 더 찾아도 같다)
    """
    if not state["answer_result"].has_evidence and len(state["chunks"]) >= TOP_K:
        return "retry"
    return END


def retry_node(state: ChatState) -> ChatState:
    """다시 찾기. 같은 질문·같은 문서 목록으로 조각을 RETRY_TOP_K개 찾아 다시 답한다.

    읽는 것: rewritten_question, project_ids, chat_history
    채우는 것: chunks, answer_result (처음 결과를 덮어쓴다. 화면·로그·채점은 두 번째 결과를 본다)
    """
    chunks = search_chunks(state["rewritten_question"], state["project_ids"], RETRY_TOP_K)
    answer_result = generate_answer(state["rewritten_question"], chunks, state["chat_history"])
    return {"chunks": chunks, "answer_result": answer_result}


@lru_cache(maxsize=1)
def build_graph() -> CompiledStateGraph:
    """노드를 잇고 1단계 뒤에 갈림길을 단 그래프를 만든다. 한 번만 만들고 재사용한다.

    받는 것: 없음
    넘기는 것: invoke(상태)하면 끝까지 돈 상태를 돌려주는 그래프
    """
    # 1. ChatState 모양의 state를 주고받는 그래프를 만든다
    graph_builder = StateGraph(ChatState)

    # 2. 노드 등록 ("이름", 함수). 화살표는 이 이름으로 잇는다
    graph_builder.add_node("rewrite", rewrite_node)
    graph_builder.add_node("find_projects", find_projects_node)
    graph_builder.add_node("search", search_node)
    graph_builder.add_node("answer", answer_node)
    graph_builder.add_node("chat", chat_node)
    graph_builder.add_node("retry", retry_node)

    # 3. 화살표로 순서 잇기: START → 1 → (갈림길) → 2 → 3 → 4 → (갈림길) → END
    graph_builder.add_edge(START, "rewrite")
    # 1단계 뒤 갈림길: route_question이 고른 이름의 노드로 간다 (rfp → 2단계, chat → 인사 답변)
    graph_builder.add_conditional_edges("rewrite", route_question)
    graph_builder.add_edge("chat", END)
    graph_builder.add_edge("find_projects", "search")
    graph_builder.add_edge("search", "answer")
    # 4단계 뒤 갈림길: should_retry가 "retry"를 고르면 한 번 더 찾고, 아니면 끝
    graph_builder.add_conditional_edges("answer", should_retry)
    graph_builder.add_edge("retry", END)

    # 4. 실행할 수 있는 그래프로 완성한다
    return graph_builder.compile()


def run_chatbot(question: str, chat_history: ChatHistory) -> ChatState:
    """질문 하나를 네 단계에 통과시킨다. 웹 서버(app/server.py)와 평가 채점이 이 함수 하나만 부른다.

    받는 것: 질문, 대화 기록 [(질문, 답변), ...]
    넘기는 것: 끝까지 채워진 ChatState. 답변은 ["answer_result"]
             (로그·채점용으로 다듬은 질문·조각도 들어 있다)
    """
    # 처음 state에는 질문과 대화 기록만 넣는다. 나머지 칸은 노드들이 채운다
    return build_graph().invoke({"question": question, "chat_history": chat_history})
