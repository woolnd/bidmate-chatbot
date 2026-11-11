
"""PDF 문서에서 텍스트를 추출한다."""

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: Path) -> str:
    """PDF 파일의 각 페이지에서 텍스트를 추출한다.

    Args:
        file_path: 읽을 PDF 파일 경로

    Returns:
        모든 페이지의 텍스트를 합친 문자열

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: PDF가 아니거나 추출된 내용이 비어 있는 경우
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF 파일이 아닙니다: {file_path}")

    reader = PdfReader(str(file_path))

    page_texts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            page_texts.append(
                f"[페이지 {page_number}]\n{page_text.strip()}"
            )

    text = "\n\n".join(page_texts)

    if not text.strip():
        raise ValueError(f"PDF 문서에서 텍스트를 추출하지 못했습니다: {file_path}")

    return text
