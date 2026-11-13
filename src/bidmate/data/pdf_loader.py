"""PDF 문서에서 본문과 표의 텍스트를 추출한다."""

from pathlib import Path

import pymupdf


def extract_pdf_text(file_path: Path) -> str:
    """
    PDF 파일에서 전체 텍스트를 추출한다.

    - 일반 본문은 텍스트 라인 단위로 추출
    - 표는 행과 열을 분석하여 '항목: 값 | 항목: 값' 형태로 변환
    - 헤더가 없는 셀은 임시 열 이름 없이 원본 텍스트 유지
    - 본문과 표를 페이지 내 좌표 기준으로 정렬

    Args:
        file_path (Path): PDF 파일 경로

    Returns:
        str: 추출된 전체 텍스트
    """

    page_texts = []

    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            elements = []
            table_areas = []

            # 1. 페이지 내 표 탐지
            tables = page.find_tables()

            for table in tables.tables:
                table_areas.append(pymupdf.Rect(table.bbox))

                # 2. 표의 모든 행과 셀을 추출
                rows = []

                for row in table.rows:
                    cell_texts = []

                    for cell in row.cells:
                        if cell is None:
                            cell_texts.append("")
                            continue

                        cell_text = page.get_textbox(pymupdf.Rect(cell)).strip()

                        # Markdown 표의 한 셀은 한 줄로 표현
                        cell_text = " ".join(cell_text.split())

                        # Markdown 구분 기호와 충돌하지 않도록 처리
                        cell_text = cell_text.replace("|", r"\|")

                        cell_texts.append(cell_text)

                    rows.append(cell_texts)

                if not rows:
                    continue

                # 열 개수를 통일
                n_cols = max(len(row) for row in rows)

                if n_cols == 0:
                    continue

                rows = [row + [""] * (n_cols - len(row)) for row in rows]

                # 3. 헤더 판단
                headers = [""] * n_cols
                data_rows = rows

                if table.header.external:
                    # 표 바깥에 별도 헤더가 있는 경우
                    external_headers = []

                    for cell in table.header.cells:
                        if cell is None:
                            external_headers.append("")
                            continue

                        header_text = page.get_textbox(pymupdf.Rect(cell)).strip()

                        header_text = " ".join(header_text.split()).replace("|", r"\|")

                        external_headers.append(header_text)

                    external_headers = external_headers[:n_cols] + [""] * max(
                        0, n_cols - len(external_headers)
                    )

                    if any(external_headers):
                        headers = external_headers

                        # 실제 사용한 외부 헤더 영역만 제외
                        table_areas.append(pymupdf.Rect(table.header.bbox))

                else:
                    first_row = rows[0]

                    # 첫 행이 실제 헤더인지 판단하기 위한
                    # 임시 휴리스틱:
                    # - 데이터 행이 하나 이상 존재
                    # - 첫 행에 빈칸이 없음
                    # - 각 셀의 텍스트가 비교적 짧음
                    is_header = (
                        len(rows) > 1
                        and all(first_row)
                        and all(len(cell) <= 40 for cell in first_row)
                    )

                    if is_header:
                        headers = first_row
                        data_rows = rows[1:]

                # 4. Markdown 표 생성
                markdown_lines = []

                markdown_lines.append("| " + " | ".join(headers) + " |")

                markdown_lines.append("| " + " | ".join(["---"] * n_cols) + " |")

                for row in data_rows:
                    markdown_lines.append("| " + " | ".join(row) + " |")

                # 5. 표를 페이지 내 위치와 함께 저장
                elements.append(
                    (
                        table.bbox[1],
                        table.bbox[0],
                        "\n" + "\n".join(markdown_lines) + "\n",
                    )
                )

            # 6. 표 영역을 제외한 일반 본문 추출
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_rect = pymupdf.Rect(line["bbox"])

                    center = (
                        (line_rect.x0 + line_rect.x1) / 2,
                        (line_rect.y0 + line_rect.y1) / 2,
                    )

                    # 이미 표로 추출한 영역의 중복 처리 방지
                    if any(area.contains(center) for area in table_areas):
                        continue

                    line_text = "".join(span["text"] for span in line["spans"])

                    if line_text.strip():
                        elements.append(
                            (
                                line_rect.y0,
                                line_rect.x0,
                                line_text,
                            )
                        )

            # 7. 본문과 표를 좌표 기준으로 정렬
            elements.sort(key=lambda item: (item[0], item[1]))

            page_texts.append("\n".join(element[2] for element in elements))

    # 8. 전체 페이지 결합
    return "\n".join(page_texts)
