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

from pathlib import Path  # 파일 및 디렉터리 경로 관리

import syhwp  # HWP 문서 읽기 및 본문, 표 추출


def extract_hwp_text(file_path: Path) -> str:
    """
    HWP 파일의 전체 본문 및 표 텍스트를 추출합니다.
    표는 가능한 경우 '항목: 값 | 항목: 값' 형식으로 변환합니다.

    Args:
        file_path (Path): HWP 파일 경로

    Returns:
        str: HWP 전체 문서에서 추출한 정제 전 텍스트
    """

    # 1. HWP 파일 열기
    doc = syhwp.open(str(file_path))

    # 본문과 표 텍스트를 저장할 리스트
    elements = []

    # 2. 문서 요소를 순서대로 순회
    for block in doc.blocks:
        block_type = type(block).__name__

        # 3. 일반 본문 텍스트 추출
        if block_type == "Paragraph":
            paragraph_text = block.text or ""

            if paragraph_text:
                elements.append(paragraph_text)

        # 4. 표 텍스트 추출
        elif block_type == "Table":
            n_rows = block.n_rows
            n_cols = block.n_cols

            if n_rows == 0 or n_cols == 0:
                continue

            # 셀 내용을 저장할 2차원 리스트 생성
            rows = [[""] * n_cols for _ in range(n_rows)]

            # 5. 셀의 시작 위치에만 텍스트 저장
            # 병합 영역 전체에 복사하지 않아 중복 방지
            for cell in block.cells:
                row_idx = cell.row
                col_idx = cell.col

                if not (0 <= row_idx < n_rows and 0 <= col_idx < n_cols):
                    continue

                cell_text = (cell.text or "").replace("\n", " ").strip()

                rows[row_idx][col_idx] = cell_text

            # 6. 단일 열 표 처리
            if n_cols == 1:
                for row in rows:
                    if row[0]:
                        elements.append(row[0])

                continue

            # 7. 한 행으로 된 2열 표 처리
            if n_rows == 1 and n_cols == 2:
                left, right = rows[0]

                if left and right:
                    elements.append(f"{left}: {right}")
                else:
                    elements.extend(value for value in rows[0] if value)

                continue

            # 8. 일반적인 표: 첫 행이 모두 채워진 경우
            first_row = rows[0]

            has_headers = n_rows > 1 and all(header for header in first_row)

            if has_headers:
                headers = first_row
                data_rows = rows[1:]

                for row in data_rows:
                    cells = []

                    for i, value in enumerate(row):
                        if not value:
                            continue

                        header = headers[i]

                        cells.append(f"{header}: {value}")

                    if cells:
                        elements.append(" | ".join(cells))

            # 9. 병합 셀 등으로 첫 행에 빈 셀이 있는 표
            else:
                row_idx = 0

                while row_idx < n_rows:
                    current_row = rows[row_idx]

                    # 현재 행에서 값이 있는 열 위치 확인
                    current_cols = [i for i, value in enumerate(current_row) if value]

                    # 다음 행이 존재하면 두 행의 구조 비교
                    if row_idx + 1 < n_rows:
                        next_row = rows[row_idx + 1]

                        next_cols = [i for i, value in enumerate(next_row) if value]

                        # 두 행의 값이 같은 열에 위치하고
                        # 항목이 2개 이상이면 헤더-데이터 관계로 판단
                        if len(current_cols) >= 2 and current_cols == next_cols:
                            cells = []

                            for col_idx in current_cols:
                                header = current_row[col_idx]
                                value = next_row[col_idx]

                                cells.append(f"{header}: {value}")

                            elements.append(" | ".join(cells))

                            # 두 행을 처리했으므로 다음 두 행으로 이동
                            row_idx += 2
                            continue

                    # 관계를 확신할 수 없다면 원본 값만 보존
                    values = [value for value in current_row if value]

                    if values:
                        elements.append(" | ".join(values))

                    row_idx += 1

        # 10. 수식 추출
        elif block_type == "Equation":
            equation_text = block.script or ""

            if equation_text:
                elements.append(f"[수식: {equation_text}]")

        # 11. 이미지 표시
        elif block_type == "Image":
            elements.append("[그림]")

    # 12. 전체 본문과 표 연결
    full_text = "\n".join(elements)

    return full_text
