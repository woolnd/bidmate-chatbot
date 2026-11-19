"""파트 간 데이터 약속.

각 단계가 무엇을 받고 무엇을 넘기는지를 여기서 정한다.
필드를 바꾸거나 지우면 다른 파트 코드가 깨지므로, 바꾸기 전에 PM과 상의한다.

정할 약속:
    ProjectInfo     data_list.csv 한 줄 (project_id = 원본 파일 이름, 기관명, 사업명, 예산, 마감일)
    LoadedDocument  원본 파일에서 뽑은 글자 (project_id, 본문)
    RewriteResult   1단계 결과 (rewritten_question, route)
    AnswerResult    답변 (answer, source_project_names, has_evidence, next_questions)
    ChatState       4단계가 주고받는 상태 (질문, 대화 기록, 단계별 결과)

조각과 검색 결과는 따로 만들지 않고 LangChain Document를 그대로 쓴다.
    조각        Document(page_content=머리말 붙은 본문, metadata=CHUNK_METADATA_KEYS)
    검색 결과   (Document, 점수) 목록
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field

# 조각 이름표(metadata)에 꼭 넣는 키. 데이터 파트와 검색 파트가 똑같이 쓴다.
CHUNK_METADATA_KEYS = ("chunk_id", "project_id", "agency_name", "project_name")

# 대화 기록: [(질문, 답변), (질문, 답변), ...] 오래된 것부터 차례로.
# 화면(브라우저)이 모아 두고, 1단계 질문 다듬기와 4단계 답변 만들기가 받는다.
ChatHistory = list[tuple[str, str]]


@dataclass
class ProjectInfo:
    """data_list.csv 한 줄. 목록표 읽기(데이터 파트)가 만들고 2단계·문서 읽기·조각 자르기가 쓴다.

    칸 이름은 조각 이름표(CHUNK_METADATA_KEYS)와 같게 맞췄다.
    """

    project_id: str  # csv '파일명' 그대로 (확장자 포함). 예: "가나다공단_이러닝.hwp"
    agency_name: str  # csv '발주 기관'
    project_name: str  # csv '사업명'
    budget: int | None  # csv '사업 금액'(원). 비어 있거나 0이면 None
    deadline: str | None  # csv '입찰 참여 마감일'을 "2024-05-31" 모양으로. 비어 있으면 None


@dataclass
class LoadedDocument:
    """원본 파일에서 뽑아 정리한 글자. 문서 읽기(데이터 파트)가 만들고 조각 자르기가 쓴다."""

    project_id: str  # ProjectInfo.project_id와 같은 값
    text: str  # 정리된 본문 전체


class RewriteResult(BaseModel):
    """1단계 질문 다듬기 결과. AI가 이 양식을 채워서 돌려준다(구조화된 출력). graph.py가 쓴다.

    route를 보고 그래프가 길을 고른다: rfp면 2·3·4단계로, chat이면 인사 답변으로 바로 간다.
    """

    rewritten_question: str = Field(
        description="새 질문을 혼자 읽어도 뜻이 통하게 다듬은 질문 한 줄. "
        "설명이나 따옴표를 붙이지 않는다."
    )
    route: Literal["rfp", "chat"] = Field(
        description="인사, 감사, 맞장구·짧은 대답, 챗봇 소개 질문, 공공사업과 관계없는 잡담이면 "
        "chat. 그 밖에 사업·시스템·기술·용어·계약에 관한 질문은 기관명이 없어도 rfp."
    )


class AnswerResult(BaseModel):
    """4단계 답변. AI가 이 양식을 채워서 돌려준다(구조화된 출력). 화면 표시와 평가 채점이 쓴다.

    with_structured_output은 dataclass를 못 받아서 이것만 pydantic으로 만든다.
    Field의 description은 AI에게 "이 칸에 무엇을 쓰라"는 지시로 그대로 넘어간다.
    """

    answer: str = Field(
        description="질문에 대한 답. 찾은 조각에 있는 내용만 쓴다. "
        "근거가 없으면 찾은 문서에 없다고 쓴다."
    )
    source_project_names: list[str] = Field(
        description="답의 근거가 된 조각의 사업명 목록. 근거가 없으면 빈 목록."
    )
    has_evidence: bool = Field(description="찾은 조각에 답의 근거가 있으면 true, 없으면 false.")
    is_rfp_question: bool = Field(
        description="질문이 공공기관 RFP·입찰에 관한 것이면 true. "
        "인사나 챗봇 소개처럼 RFP와 관계없는 질문이면 false. 화면이 근거 안내를 숨길 때 쓴다."
    )
    next_questions: list[str] = Field(
        description="이 답 다음에 물어볼 만한 짧은 질문 2개. 기관명이나 사업명을 넣어 "
        "혼자 읽어도 뜻이 통하게 쓴다. RFP 질문이 아니면 빈 목록. "
        "화면이 추천 질문 버튼으로 보여 준다."
    )


class ChatState(TypedDict, total=False):
    """4단계가 주고받는 상태. graph.py(LangGraph)가 단계 사이로 넘긴다.

    처음에는 question, chat_history만 넣는다. 각 단계는 자기가 만든 칸만 채워서 돌려주고,
    LangGraph가 그 칸을 상태에 합쳐 다음 단계로 넘긴다. 끝나면 모든 칸이 차 있다.
    total=False: 처음에 비어 있는 칸이 있어도 된다는 뜻.
    """

    question: str  # 사용자가 쓴 질문 그대로
    chat_history: ChatHistory  # 화면이 모아 둔 대화 기록
    rewritten_question: str  # 1단계가 채운다
    route: str  # 1단계가 채운다 ("rfp" 또는 "chat"). 그래프가 이 값으로 길을 고른다
    project_ids: list[str]  # 2단계가 채운다 (못 고르면 [])
    chunks: list[tuple[Document, float]]  # 3단계가 채운다 (Document, 점수) 목록
    answer_result: AnswerResult  # 4단계가 채운다
