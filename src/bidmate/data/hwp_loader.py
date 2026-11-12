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

import syhwp


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
    elements = []

    # 2. 문서 요소를 순서대로 순회
    for block in doc.blocks:
        block_type = type(block).__name__

        # 3. 일반 문단 추출
        if block_type == "Paragraph":
            paragraph_text = block.text or ""

            if paragraph_text.strip():
                elements.append(paragraph_text)

        # 4. 표 추출
        elif block_type == "Table":
            n_rows = block.n_rows
            n_cols = block.n_cols

            if n_rows == 0 or n_cols == 0:
                continue

            # 5. 표의 행과 열 구조 생성
            rows = [[""] * n_cols for _ in range(n_rows)]

            first_row_merged = False
            merged_cells = []

            # 6. 원본 셀 데이터 배치
            for cell in block.cells:
                row_idx = cell.row
                col_idx = cell.col

                if not (0 <= row_idx < n_rows and 0 <= col_idx < n_cols):
                    continue

                cell_text = " ".join((cell.text or "").split())
                cell_text = cell_text.replace("|", r"\|")

                row_span = max(1, int(getattr(cell, "row_span", 1) or 1))
                col_span = max(1, int(getattr(cell, "col_span", 1) or 1))

                # 첫 번째 행의 병합 셀 여부 확인
                if row_idx == 0 and (row_span > 1 or col_span > 1):
                    first_row_merged = True

                # 셀의 시작 위치에 원본 텍스트 저장
                rows[row_idx][col_idx] = cell_text

                # 세로 병합 셀 정보 별도 저장
                if row_span > 1 and cell_text:
                    merged_cells.append((row_idx, col_idx, row_span, cell_text))

            # 7. 세로 병합 셀 처리
            # 주변 열에 데이터가 있는 행에만 병합 셀 값 적용
            for row_idx, col_idx, row_span, cell_text in merged_cells:
                for next_row in range(
                    row_idx + 1,
                    min(row_idx + row_span, n_rows),
                ):
                    has_other_data = any(
                        value.strip() for i, value in enumerate(rows[next_row]) if i != col_idx
                    )

                    if has_other_data and not rows[next_row][col_idx]:
                        rows[next_row][col_idx] = cell_text

            # 8. 내용이 전혀 없는 표는 제외
            if not any(any(cell.strip() for cell in row) for row in rows):
                continue

            # 9. 헤더 판단
            first_row = rows[0]

            header_keywords = {
                "구분",
                "항목",
                "분류",
                "번호",
                "순번",
                "추진과제",
            }

            has_headers = (
                n_rows > 1
                and not first_row_merged
                and all(first_row)
                and first_row[0] in header_keywords
            )

            if has_headers:
                headers = first_row
                data_rows = rows[1:]
            else:
                # 헤더 판단이 어려우면 모든 행을 데이터로 보존
                headers = [""] * n_cols
                data_rows = rows

            # 10. 내용이 없는 데이터 행 제거
            data_rows = [row for row in data_rows if any(cell.strip() for cell in row)]

            # 헤더와 데이터 모두 비어 있으면 제외
            if not data_rows and not any(headers):
                continue

            # 11. Markdown 표 생성
            markdown_lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * n_cols) + " |",
            ]

            for row in data_rows:
                markdown_lines.append("| " + " | ".join(row) + " |")

            # 표 앞뒤에 줄바꿈을 넣어 본문과 구분
            elements.append("\n".join(markdown_lines))

        # 12. 수식 추출
        elif block_type == "Equation":
            equation_text = block.script or ""

            if equation_text:
                elements.append(f"[수식: {equation_text}]")

        # 13. 이미지 표시
        elif block_type == "Image":
            elements.append("[그림]")

    # 14. 본문과 표를 문서 순서대로 결합
    return "\n".join(elements)
