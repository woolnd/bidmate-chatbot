"""실제 RFP 문서들을 읽고 자른 뒤 메타데이터를 붙여 저장소(Chroma 등)로 넘기는 파이프라인.

프로젝트 가이드라인 기준:
- 머리말 형식: "[기관명 / 사업명]"
- 메타데이터 키 4개: chunk_id, project_id, agency_name, project_name
- 기본 정보 조각의 chunk_id는 '-info'로 끝남
"""

from typing import Any

from langchain_core.documents import Document

# config.py 등에서 경로와 설정값을 가져온다고 가정
# from src.config import PROJECT_LIST_PATH, RAW_FOLDER


def load_project_list(path: str) -> list[dict[str, Any]]:
    """프로젝트 목록표(CSV 등)를 불러온다."""
    # TODO: 실제 목록표 로드 구현
    return []


def load_document(project: dict[str, Any], raw_folder: str) -> str:
    """프로젝트 정보에 해당하는 원본 파일(.hwp, .pdf)을 읽어 텍스트를 뽑는다."""
    # TODO: HWP / PDF 텍스트 추출 구현 (extract_hwp_text 등)
    return "추출된 원본 텍스트..."


from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document(raw_text: str, project: dict[str, Any]) -> list[Document]:
    """긴 텍스트를 규칙에 맞게 1000 글자 크기로 자르고 머리말과 메타데이터를 붙인다."""
    project_id = project["project_id"]
    agency_name = project["agency_name"]
    project_name = project["project_name"]
    
    chunks: list[Document] = []
    
    # 청크 사이즈 1000, 오버랩 200으로 설정 (문맥이 끊기는 것을 방지하기 위해 겹치는 구간 둠)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # 텍스트 분할 실행
    text_pieces = text_splitter.split_text(raw_text)
    
    for idx, piece in enumerate(text_pieces, start=1):
        chunk_suffix = f"{idx:04d}" # 0001, 0002 형태
        chunks.append(
            Document(
                page_content=f"[{agency_name} / {project_name}]\n{piece}",
                metadata={
                    "chunk_id": f"{project_id}-{chunk_suffix}",
                    "project_id": project_id,
                    "agency_name": agency_name,
                    "project_name": project_name,
                },
            )
        )
        
    return chunks


def make_info_chunk(project: dict[str, Any]) -> Document:
    """사업별 기본 정보 조각을 생성한다 (chunk_id가 -info로 끝남)."""
    project_id = project["project_id"]
    agency_name = project["agency_name"]
    project_name = project["project_name"]

    return Document(
        page_content=f"[{agency_name} / {project_name}]\n기본 정보: 발주 기관 {agency_name} | 예산 ...",
        metadata={
            "chunk_id": f"{project_id}-info",
            "project_id": project_id,
            "agency_name": agency_name,
            "project_name": project_name,
        },
    )


def save_chunks(all_chunks: list[Document]) -> None:
    """완성된 조각들을 검색 파트 저장소(Chroma 등)에 저장한다."""
    # TODO: 검색 파트와 맞춘 save_chunks 함수 연동
    pass


def main() -> None:
    """문서들을 읽고 자라서 저장하는 메인 파이프라인."""
    project_list = load_project_list("data/data_list.csv")
    all_chunks: list[Document] = []
    failed_files: list[str] = []

    for project in project_list:
        try:
            # 1. 파일 읽기 및 텍스트 추출
            raw_text = load_document(project, "data/raw")

            # 2. 본문 조각 자르기
            project_chunks = split_document(raw_text, project)
            all_chunks.extend(project_chunks)

            # 3. 기본 정보 조각 추가
            info_chunk = make_info_chunk(project)
            all_chunks.append(info_chunk)

        except Exception as e:
            # 에러가 나도 멈추지 않고 실패한 파일 이름을 모아둠
            failed_files.append(project.get("project_id", "unknown"))
            print(f"Error processing {project.get('project_id')}: {e}")

    # 4. 검색 파트로 전달
    save_chunks(all_chunks)

    if failed_files:
        print(f"실패한 파일 목록: {failed_files}")


if __name__ == "__main__":
    main()
