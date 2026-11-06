"""문서를 검색용 조각으로 자르기. 담당: 데이터 파트

조각 = 이름표(metadata) + 내용(page_content). LangChain Document로 만든다.
    page_content  "[기관명 / 사업명]" 머리말 + 본문
    metadata      chunk_id, project_id, agency_name, project_name

규칙:
    - 1000자씩 자르고 앞 조각과 200자 겹친다 (시작값. 2주차에 500/1000/1500 비교)
    - chunk_id = "파일이름-0001" 부터 순서대로
    - 문서마다 기본 정보 조각 1개 (chunk_id = "파일이름-info").
      목록표 한 줄(기관명·사업명·예산·마감일)로 만든다. 예산·마감일 질문용

받는 것: LoadedDocument, ProjectInfo
넘기는 것: Document 목록
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ProjectInfo, LoadedDocument의 칸은 11/10에 data_list.csv를 보고 schemas.py에 만든다
    from bidmate.schemas import LoadedDocument, ProjectInfo

from langchain_core.documents import Document


def split_document(document: LoadedDocument, project: ProjectInfo) -> list[Document]:
    """본문을 CHUNK_SIZE자씩, CHUNK_OVERLAP자 겹치게 잘라 조각 목록을 돌려준다."""
    # TODO 1. 머리말 만들기: "[기관명 / 사업명]"
    # TODO 2. document.text를 CHUNK_SIZE(1000)자씩, CHUNK_OVERLAP(200)자 겹치게 자르기
    #         (숫자는 config.py에서 가져온다)
    # TODO 3. 자른 글마다 Document 만들기
    #           page_content = 머리말 + 줄바꿈 + 자른 글
    #           metadata     = chunk_id("파일이름-0001"), project_id, agency_name, project_name
    # TODO 4. 조각 목록 돌려주기
    return []


def make_info_chunk(project: ProjectInfo) -> Document:
    """목록표 한 줄로 기본 정보 조각 1개를 만든다. 예산·마감일 질문에 쓴다."""
    # TODO 1. page_content = 머리말 + "기본 정보: 발주 기관 … | 예산 …원 | 입찰 마감일 …"
    # TODO 2. metadata = 보통 조각과 같은 키 4개, chunk_id만 "파일이름-info"
    raise NotImplementedError("ProjectInfo를 schemas.py에 만든 뒤 채운다 (11/10)")
