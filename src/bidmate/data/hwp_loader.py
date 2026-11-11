
"""HWP 문서에서 텍스트와 표를 추출한다."""

from pathlib import Path

import syhwp


def extract_hwp_text(file_path: Path) -> str:
    """HWP 파일을 Markdown 텍스트로 추출한다.

    Args:
        file_path: 읽을 HWP 파일 경로

    Returns:
        본문과 표가 포함된 Markdown 텍스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: 추출된 내용이 비어 있는 경우
    """
    if not file_path.exists():
        raise FileNotFoundError(f"HWP 파일을 찾을 수 없습니다: {file_path}")

    if file_path.suffix.lower() != ".hwp":
        raise ValueError(f"HWP 파일이 아닙니다: {file_path}")

    text = syhwp.extract_markdown(str(file_path))

    if not text or not text.strip():
        raise ValueError(f"HWP 문서에서 텍스트를 추출하지 못했습니다: {file_path}")

    return text.strip()
