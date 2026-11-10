"""문서 목록표(data_list.csv) 읽기. 담당: 데이터 파트

목록표는 기관명·사업명·예산·공개일과 원본 파일 이름을 이어 준다.
2단계(어느 문서인지 고르기), 문서 읽기, 기본 정보 조각, 평가가 모두 이 결과를 쓴다.
날짜·금액 모양은 여기서 하나로 맞춘다.

받는 것: data_list.csv 경로
넘기는 것: ProjectInfo 목록 (project_id = 원본 파일 이름)

실행해 보기 (레포 맨 위 폴더에서 `uv run python`을 켜고 한 줄씩 입력, 끝낼 때 exit()):
    >>> from bidmate.config import PROJECT_LIST_PATH
    >>> from bidmate.data.metadata import load_project_list
    >>> projects = load_project_list(PROJECT_LIST_PATH)
    >>> len(projects)       # 100이 나와야 한다
    >>> projects[0]         # 첫 문서 정보 확인
"""

from future import annotations

from pathlib import Path

import pandas as pd

from bidmate.config import RAW_DATA_DIR  # config.py에 data/raw 경로가 없으면 추가하거나 이름을 맞춘다
from bidmate.schemas import ProjectInfo


def _to_int(value) -> int | None:
    """'130000000.0' 같은 문자열을 정수로 바꾼다. 비어 있으면 None."""
    if pd.isna(value):
        return None
    return int(float(value))


def _to_date(value) -> str | None:
    """'2024-10-04 13:51:23' -> '2024-10-04'. 비어 있으면 None."""
    if pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _to_text(value) -> str | None:
    """앞뒤 공백을 지운 문자열. 비어 있으면 None."""
    if pd.isna(value):
        return None
    return str(value).strip()


def load_project_list(csv_path: Path) -> list[ProjectInfo]:
    """목록표를 읽어 문서 100개의 정보를 돌려준다."""
    # 1. csv 읽기: BOM이 있어서 utf-8-sig, 값은 전부 문자열로 읽고 아래에서 직접 바꾼다
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)

    # 3. 원본 파일이 data/raw/에 다 있는지 확인. 없는 파일은 이름 출력
    missing = [name for name in df["파일명"] if not (RAW_DATA_DIR / name).exists()]
    if missing:
        print(f"[경고] data/raw/에 없는 파일 {len(missing)}개:")
        for name in missing:
            print("  -", name)

    # 2, 4. 예산은 숫자로, 날짜는 "2024-05-31" 모양으로 맞추며 한 줄씩 ProjectInfo로 바꾼다
    projects: list[ProjectInfo] = []
    for _, row in df.iterrows():
        projects.append(
            ProjectInfo(
                project_id=row["파일명"],
                project_name=row["사업명"].strip(),
                agency_name=row["발주 기관"].strip(),
                budget=_to_int(row["사업 금액"]),
                published_date=_to_date(row["공개 일자"]),
                bid_start_date=_to_date(row["입찰 참여 시작일"]),
                bid_end_date=_to_date(row["입찰 참여 마감일"]),
                notice_no=_to_text(row["공고 번호"]),
                notice_round=_to_int(row["공고 차수"]),
                summary=_to_text(row["사업 요약"]) or "",
                file_type=row["파일형식"].strip().lower(),
            )
        )
    return projects