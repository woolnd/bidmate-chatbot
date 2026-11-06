"""2단계(어느 문서인지 고르기): 질문 속 기관명·사업명으로 찾아볼 문서를 정한다. 담당: 검색 파트

여기서 문서를 좁혀 두면 3단계가 100개 전체가 아니라 몇 개 문서 안에서만 검색해서
엉뚱한 사업 내용이 섞여 나오는 일이 줄어든다.

1주차: 질문에 기관명·사업명이 그대로 들어 있으면 그 문서를 고른다.
2주차: "국민연금"처럼 줄이거나 조금 틀리게 써도 고른다. 못 고른 질문을 보고 고친다.

받는 것: 다듬은 질문, ProjectInfo 목록 (목록표)
넘기는 것: 고른 project_id 목록 (못 고르면 빈 목록)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ProjectInfo의 칸은 11/10에 data_list.csv를 보고 schemas.py에 만든다
    from bidmate.schemas import ProjectInfo


def find_project_ids(question: str, projects: list[ProjectInfo]) -> list[str]:
    """질문에 기관명이나 사업명이 들어 있는 문서를 고른다."""
    # TODO 1. 문서(ProjectInfo)를 하나씩 보면서
    # TODO 2. 기관명이나 사업명이 질문 안에 들어 있으면 project_id를 모으기
    # TODO 3. 모은 목록 돌려주기 (하나도 없으면 빈 목록 [])
    # 2주차(11/19): "국민연금"처럼 줄이거나 조금 틀리게 써도 고르기
    return []
