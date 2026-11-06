"""미리 한 번 해 두는 일: 목록표 읽기 → 문서 읽기 → 조각 자르기 → Chroma 저장

순서:
    1. data/metadata.py      목록표 읽기 → ProjectInfo 목록
    2. data/loader.py        문서마다 글자 뽑기 → LoadedDocument
    3. data/chunker.py       조각 자르기 + 기본 정보 조각 → Document 목록
    4. retrieval/vector_store.py   Chroma에 저장

실행: uv run python scripts/build_index.py
문서나 조각 자르기 방식이 바뀌었을 때만 다시 돌린다 (임베딩은 돈이 든다).
"""

from langchain_core.documents import Document

from bidmate.config import PROJECT_LIST_PATH, RAW_FOLDER
from bidmate.data.chunker import make_info_chunk, split_document
from bidmate.data.loader import load_document
from bidmate.data.metadata import load_project_list
from bidmate.retrieval.vector_store import save_chunks


def main() -> None:
    """문서 100개를 읽고 잘라서 Chroma에 저장한다."""
    project_list = load_project_list(PROJECT_LIST_PATH)  # ① 목록표 → ProjectInfo 목록

    all_chunks: list[Document] = []
    failed_files: list[str] = []
    for project in project_list:
        try:
            document = load_document(project, RAW_FOLDER)  # ② 글자 뽑기
            all_chunks += split_document(document, project)  # ③ 조각 자르기
            all_chunks.append(make_info_chunk(project))  # ③ 기본 정보 조각
        except Exception as error:  # 한 문서가 실패해도 나머지는 계속한다
            failed_files.append(f"{project.project_id}: {error}")

    save_chunks(all_chunks)  # ④ 검색 파트로 넘겨서 Chroma에 저장
    print(f"문서 {len(project_list)}개 → 조각 {len(all_chunks)}개")
    for failed in failed_files:
        print(f"실패: {failed}")


if __name__ == "__main__":
    main()
