
"""원본 파일을 읽어 정리된 문서로 만든다.

HWP와 PDF의 추출 결과를 LoadedDocument 형식으로 통일한다.
"""

from pathlib import Path

from bidmate.data.hwp_loader import extract_hwp_text
from bidmate.data.pdf_loader import extract_pdf_text
from bidmate.schemas import LoadedDocument, ProjectInfo


def load_document(
    project: ProjectInfo,
    raw_folder: Path,
) -> LoadedDocument:
    """원본 파일을 읽고 텍스트를 정리해 반환한다.

    원본 파일은 data/raw/files 폴더에 저장되어 있다.
    project.project_id에는 원본 파일명이 들어 있다.
    """
    file_path = raw_folder / "files" / project.project_id

    if not file_path.exists():
        raise FileNotFoundError(
            f"원본 문서를 찾을 수 없습니다: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".hwp":
        text = extract_hwp_text(file_path)
    elif extension == ".pdf":
        text = extract_pdf_text(file_path)
    else:
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {extension}"
        )

    text = clean_text(text)

    return LoadedDocument(
        project_id=project.project_id,
        text=text,
    )


def clean_text(text: str) -> str:
    """줄바꿈과 불필요한 빈 줄을 정리한다.

    표의 Markdown 구조와 페이지 구분은 유지한다.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")

    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines = []
    previous_line_empty = False

    for line in lines:
        is_empty = not line

        # 연속된 빈 줄은 한 줄만 유지한다.
        if is_empty and previous_line_empty:
            continue

        cleaned_lines.append(line)
        previous_line_empty = is_empty

    return "\n".join(cleaned_lines).strip()
