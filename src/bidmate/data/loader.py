"""원본 파일을 읽어 정리된 문서로 만든다. 담당: 데이터 파트

확장자를 보고 hwp_loader 또는 pdf_loader를 부른 뒤,
머리말·꼬리말, 쪽번호, 연속 공백을 똑같은 규칙으로 정리한다.
hwp와 pdf 결과를 같은 형식(LoadedDocument)으로 맞춰서
다음 단계(조각 자르기)가 파일 형식을 신경 쓰지 않게 한다.

받는 것: ProjectInfo (목록표 한 줄)
넘기는 것: LoadedDocument

실행해 보기 (레포 맨 위 폴더에서 `uv run python`을 켜고 한 줄씩 입력, 끝낼 때 exit()):
    >>> from bidmate.config import PROJECT_LIST_PATH, RAW_FOLDER
    >>> from bidmate.data.metadata import load_project_list
    >>> from bidmate.data.loader import load_document, clean_text
    >>> projects = load_project_list(PROJECT_LIST_PATH)
    >>> document = load_document(projects[0], RAW_FOLDER)   # 첫 문서 읽고 정리
    >>> print(document.text[:1000])
    >>> clean_text("공백이     많은    글")                   # 정리 규칙만 따로 확인
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from bidmate.data.hwp_loader import extract_hwp_text
from bidmate.data.pdf_loader import extract_pdf_text
from bidmate.schemas import LoadedDocument, ProjectInfo

if TYPE_CHECKING:
    # ProjectInfo, LoadedDocument의 칸은 schemas.py에 있다 (11/10 확정)
    from bidmate.schemas import LoadedDocument, ProjectInfo




def load_document(project: ProjectInfo, raw_folder: Path) -> LoadedDocument:
    """확장자를 보고 알맞은 함수로 글자를 뽑고, 똑같은 규칙으로 정리한다."""
    # TODO 1. raw_folder / project.project_id 로 원본 경로 만들기 (문서 번호가 곧 파일 이름)
    # TODO 2. .hwp면 extract_hwp_text, .pdf면 extract_pdf_text 부르기
    # TODO 3. clean_text로 정리하기
    # TODO 4. LoadedDocument(project_id, text)에 담아 돌려주기

    file_path = raw_folder / project.project_id

    if not file_path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    file_type = file_path.suffix.lower()

    if file_type == ".pdf":
        raw_text = extract_pdf_text(file_path)
    elif file_type == ".hwp":
        raw_text = extract_hwp_text(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_type}")

    cleaned_text = clean_text(raw_text)

    return LoadedDocument(
        project_id=project.project_id,
        text=cleaned_text,
    )


def clean_text(text: str) -> str:
    """쓸데없는 공백·머리말·쪽번호를 지운다."""
    # TODO 1주차: 연속 공백 줄이기만
    # TODO 2주차: 머리말·꼬리말, 쪽번호 지우기
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
