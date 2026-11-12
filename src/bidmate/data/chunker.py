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

실행해 보기 (레포 맨 위 폴더에서 `uv run python`을 켜고 한 줄씩 입력, 끝낼 때 exit()):
    >>> from bidmate.config import PROJECT_LIST_PATH, RAW_FOLDER
    >>> from bidmate.data.metadata import load_project_list
    >>> from bidmate.data.loader import load_document
    >>> from bidmate.data.chunker import split_document, make_info_chunk
    >>> projects = load_project_list(PROJECT_LIST_PATH)
    >>> chunks = split_document(load_document(projects[0], RAW_FOLDER), projects[0])
    >>> len(chunks)                 # 조각 수
    >>> print(chunks[0].page_content)   # 머리말 [기관명 / 사업명]이 붙었는지
    >>> chunks[0].metadata          # 이름표 4칸
    >>> make_info_chunk(projects[0])    # 기본 정보 조각
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bidmate.config import CHUNK_OVERLAP, CHUNK_SIZE

if TYPE_CHECKING:
    # ProjectInfo, LoadedDocument의 칸은 schemas.py에 있다 (11/10 확정)
    from bidmate.schemas import LoadedDocument, ProjectInfo

# 문단(\n\n) → 줄바꿈(\n) → 띄어쓰기 순서로 끊을 자리를 찾고, 없으면 글자 단위로 자른다
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _make_metadata(project: ProjectInfo, chunk_id: str) -> dict[str, str]:
    """조각 번호와 사업 정보로 이름표 4칸을 만든다."""
    return {
        "chunk_id": chunk_id,
        "project_id": project.project_id,
        "agency_name": project.agency_name,
        "project_name": project.project_name,
    }


def split_document(document: LoadedDocument, project: ProjectInfo) -> list[Document]:
    """본문을 CHUNK_SIZE자씩, CHUNK_OVERLAP자 겹치게 잘라 조각 목록을 돌려준다."""
    # 1. 머리말 만들기: "[기관명 / 사업명]"
    header = f"[{project.agency_name} / {project.project_name}]"

    # 2. 본문을 CHUNK_SIZE자 이하 조각으로 자르기 (앞 조각과 CHUNK_OVERLAP자 겹침)
    text_pieces = TEXT_SPLITTER.split_text(document.text)

    # 3. 자른 글마다 Document 만들기. chunk_id는 "파일이름-0001"부터
    return [
        Document(
            page_content=f"{header}\n{piece}",
            metadata=_make_metadata(project, f"{project.project_id}-{number:04d}"),
        )
        for number, piece in enumerate(text_pieces, start=1)
    ]


def make_info_chunk(project: ProjectInfo) -> Document:
    """목록표 한 줄로 기본 정보 조각 1개를 만든다. 예산·마감일 질문에 쓴다."""
    # 1. 값이 없으면 "정보 없음"으로 적는다
    budget = f"{project.budget:,}원" if project.budget else "정보 없음"
    deadline = project.deadline or "정보 없음"

    # 2. 머리말 + 기본 정보 한 줄. 이름표는 보통 조각과 같고 chunk_id만 "파일이름-info"
    return Document(
        page_content=(
            f"[{project.agency_name} / {project.project_name}]\n"
            f"기본 정보: 발주 기관 {project.agency_name} | 사업명 {project.project_name} | "
            f"예산 {budget} | 입찰 마감일 {deadline}"
        ),
        metadata=_make_metadata(project, f"{project.project_id}-info"),
    )
