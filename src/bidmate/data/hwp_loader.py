"""hwp 파일에서 글자 뽑기. 담당: 데이터 파트

하루 안에 안 되면 PM과 다른 방법(예: pdf로 변환해서 읽기)을 정한다.
표는 뽑을 때 한 행을 한 줄로 풀어 쓴다: "칸: 값 | 칸: 값" (JSON 아님)

받는 것: hwp 파일 경로
넘기는 것: 본문 글자 전체 (정리 전. 정리는 loader.py가 한다)

실행해 보기 (레포 맨 위 폴더에서 `uv run python`을 켜고 한 줄씩 입력, 끝낼 때 exit()):
    >>> from bidmate.config import RAW_FOLDER
    >>> from bidmate.data.hwp_loader import extract_hwp_text
    >>> text = extract_hwp_text(RAW_FOLDER / "파일이름.hwp")   # data/raw/ 안의 hwp 하나
    >>> print(text[:1000])  # 앞 1000자만 보기
"""

from pathlib import Path


def extract_hwp_text(file_path: Path) -> str:
    """hwp 파일에서 본문 글자를 뽑는다. 표는 "칸: 값 | 칸: 값"으로 풀어 쓴다."""
    # TODO: 라이브러리로 hwp 열기 → 글자 뽑기 → 돌려주기
    return "지어낸 hwp 본문입니다. " * 100
