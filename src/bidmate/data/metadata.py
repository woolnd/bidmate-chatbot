"""문서 목록표(data_list.csv) 읽기. 담당: 데이터 파트

목록표는 기관명·사업명·예산·마감일과 원본 파일 이름을 이어 준다.
2단계(어느 문서인지 고르기), 문서 읽기, 기본 정보 조각, 평가가 모두 이 결과를 쓴다.
날짜·금액 모양은 여기서 하나로 맞춘다.

받는 것: data_list.csv 경로
넘기는 것: ProjectInfo 목록 (project_id = 원본 파일 이름)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ProjectInfo, LoadedDocument의 칸은 11/10에 data_list.csv를 보고 schemas.py에 만든다
    from bidmate.schemas import ProjectInfo

from pathlib import Path


def load_project_list(csv_path: Path) -> list[ProjectInfo]:
    """목록표를 읽어 문서 100개의 정보를 돌려준다."""
    # TODO 1. csv 읽기 (pandas를 쓰면 편하다)
    # TODO 2. 예산은 숫자로, 날짜는 "2024-05-31" 모양으로 맞추기
    # TODO 3. 원본 파일이 data/raw/에 다 있는지 확인. 없는 파일은 이름 출력
    # TODO 4. 한 줄씩 ProjectInfo로 바꿔서 목록으로 돌려주기
    return []
