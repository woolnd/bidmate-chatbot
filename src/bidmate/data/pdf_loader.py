"""pdf 파일에서 글자 뽑기. 담당: 데이터 파트

표는 뽑을 때 한 행을 한 줄로 풀어 쓴다: "칸: 값 | 칸: 값" (JSON 아님)

받는 것: pdf 파일 경로
넘기는 것: 본문 글자 전체 (정리 전. 정리는 loader.py가 한다)
"""

from pathlib import Path


def extract_pdf_text(file_path: Path) -> str:
    """pdf 파일에서 본문 글자를 뽑는다. 표는 "칸: 값 | 칸: 값"으로 풀어 쓴다."""
    # TODO: 라이브러리로 pdf 열기 → 쪽마다 글자 뽑기 → 이어 붙여 돌려주기
    return "지어낸 pdf 본문입니다. " * 100
