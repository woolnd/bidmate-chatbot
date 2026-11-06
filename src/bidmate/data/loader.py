"""원본 파일을 읽어 정리된 문서로 만든다. 담당: 데이터 파트

확장자를 보고 hwp_loader 또는 pdf_loader를 부른 뒤,
머리말·꼬리말, 쪽번호, 연속 공백을 똑같은 규칙으로 정리한다.
hwp와 pdf 결과를 같은 형식(LoadedDocument)으로 맞춰서
다음 단계(조각 자르기)가 파일 형식을 신경 쓰지 않게 한다.

받는 것: ProjectInfo (목록표 한 줄)
넘기는 것: LoadedDocument
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ProjectInfo, LoadedDocument의 칸은 11/10에 data_list.csv를 보고 schemas.py에 만든다
    from bidmate.schemas import LoadedDocument, ProjectInfo

from pathlib import Path


def load_document(project: ProjectInfo, raw_folder: Path) -> LoadedDocument:
    """확장자를 보고 알맞은 함수로 글자를 뽑고, 똑같은 규칙으로 정리한다."""
    # TODO 1. raw_folder / project.project_id 로 원본 경로 만들기 (문서 번호가 곧 파일 이름)
    # TODO 2. .hwp면 extract_hwp_text, .pdf면 extract_pdf_text 부르기
    # TODO 3. clean_text로 정리하기
    # TODO 4. LoadedDocument(project_id, text)에 담아 돌려주기
    raise NotImplementedError("LoadedDocument를 schemas.py에 만든 뒤 채운다 (11/10)")


def clean_text(text: str) -> str:
    """쓸데없는 공백·머리말·쪽번호를 지운다."""
    # TODO 1주차: 연속 공백 줄이기만
    # TODO 2주차: 머리말·꼬리말, 쪽번호 지우기
    return text
