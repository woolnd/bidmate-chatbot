"""파트 간 데이터 약속.

각 단계가 무엇을 받고 무엇을 넘기는지를 여기서 정한다.
필드를 바꾸거나 지우면 다른 파트 코드가 깨지므로, 바꾸기 전에 PM과 상의한다.

정할 약속:
    ProjectInfo     data_list.csv 한 줄 (project_id = 원본 파일 이름, 기관명, 사업명, 예산, 마감일)
    LoadedDocument  원본 파일에서 뽑은 글자 (project_id, 본문)
    AnswerResult    답변 (answer, source_project_names, has_evidence)
    ChatState       4단계가 주고받는 상태 (질문, 대화 기록, 단계별 결과)

조각과 검색 결과는 따로 만들지 않고 LangChain Document를 그대로 쓴다.
    조각        Document(page_content=머리말 붙은 본문, metadata=CHUNK_METADATA_KEYS)
    검색 결과   (Document, 점수) 목록
"""

from dataclasses import dataclass

# 조각 이름표(metadata)에 꼭 넣는 키. 데이터 파트와 검색 파트가 똑같이 쓴다.
CHUNK_METADATA_KEYS = ("chunk_id", "project_id", "agency_name", "project_name")

# 대화 기록: [(질문, 답변), (질문, 답변), ...] 오래된 것부터 차례로.
# 화면(Streamlit)이 모아 두고, 1단계 질문 다듬기와 4단계 답변 만들기가 받는다.
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
