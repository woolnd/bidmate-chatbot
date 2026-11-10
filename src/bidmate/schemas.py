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


@dataclass(frozen=True)
class ProjectInfo:
    """data_list.csv 한 줄. 날짜는 "2024-05-31" 모양, 비어 있는 값은 None."""

    project_id: str  # 원본 파일 이름 (data/raw/ 안의 파일)
    project_name: str  # 사업명
    agency_name: str  # 발주 기관
    budget: int | None  # 사업 금액(원)
    published_date: str  # 공개 일자
    bid_start_date: str | None  # 입찰 참여 시작일
    bid_end_date: str | None  # 입찰 참여 마감일
    notice_no: str | None  # 공고 번호
    notice_round: int | None  # 공고 차수
    summary: str  # 사업 요약
    file_type: str  # "hwp" 또는 "pdf"


@dataclass(frozen=True)
class LoadedDocument:
    """원본 파일에서 뽑은 글자."""

    project_id: str  # ProjectInfo.project_id와 같은 값
    text: str  # 본문