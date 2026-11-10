"""4단계를 LangGraph로 연결한다. 담당: 재웅

질문 다듬기 → 어느 문서인지 고르기 → 문서에서 관련 부분 찾기 → 답변 만들기
각 단계(노드)는 ChatState에서 필요한 값을 읽고, 자기가 만든 값만 돌려준다.
LangGraph는 돌려준 값을 상태에 합쳐서 다음 단계로 넘긴다.

순서가 항상 같은 '정해진 흐름'이다. AI가 스스로 다음 단계를 고르지 않는다.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from bidmate.config import PROJECT_LIST_PATH, TOP_K
from bidmate.data.metadata import load_project_list
from bidmate.generation.answer_generator import generate_answer
from bidmate.generation.question_rewriter import rewrite_question
from bidmate.retrieval.project_finder import find_project_ids
from bidmate.retrieval.vector_store import search_chunks
from bidmate.schemas import ChatHistory, ChatState, ProjectInfo


@lru_cache(maxsize=1)
def get_projects() -> list[ProjectInfo]:
    """목록표(data_list.csv)를 한 번만 읽어 두고 계속 같은 목록을 돌려준다.

    받는 것: 없음
    넘기는 것: ProjectInfo 목록 (2단계가 질문과 비교한다)
    @lru_cache: 처음 부를 때만 파일을 읽고, 그다음부터는 기억해 둔 값을 돌려준다.
    """
    return load_project_list(PROJECT_LIST_PATH)


# ── 노드 4개: 상태에서 필요한 값을 읽고, 자기가 만든 칸만 dict로 돌려준다 ──


def rewrite_node(state: ChatState) -> ChatState:
    """1단계 질문 다듬기. 읽는 것: question, chat_history → 채우는 것: rewritten_question"""
    rewritten_question = rewrite_question(state["question"], state["chat_history"])
    return {"rewritten_question": rewritten_question}


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


@lru_cache(maxsize=1)
def build_graph() -> CompiledStateGraph:
    """노드 4개를 정해진 순서로 이은 그래프를 만든다. 한 번만 만들고 재사용한다.

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

    # 3. 화살표로 순서 잇기: START → 1 → 2 → 3 → 4 → END (갈림길 없는 정해진 흐름)
    graph_builder.add_edge(START, "rewrite")
    graph_builder.add_edge("rewrite", "find_projects")
    graph_builder.add_edge("find_projects", "search")
    graph_builder.add_edge("search", "answer")
    graph_builder.add_edge("answer", END)

    # 4. 실행할 수 있는 그래프로 완성한다
    return graph_builder.compile()


def run_chatbot(question: str, chat_history: ChatHistory) -> ChatState:
    """질문 하나를 네 단계에 통과시킨다. 화면(Streamlit)과 평가 채점이 이 함수 하나만 부른다.

    받는 것: 질문, 대화 기록 [(질문, 답변), ...]
    넘기는 것: 끝까지 채워진 ChatState. 답변은 ["answer_result"]
             (로그·채점용으로 다듬은 질문·조각도 들어 있다)
    """
    # 처음 state에는 질문과 대화 기록만 넣는다. 나머지 칸은 노드들이 채운다
    return build_graph().invoke({"question": question, "chat_history": chat_history})
