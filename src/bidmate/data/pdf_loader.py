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

    # 1. PDF 파일 열기
    with pymupdf.open(file_path) as pdf:

        # 2. 페이지별 텍스트 추출
        for page in pdf:

            elements = []
            table_areas = []

            # 3. 페이지 내 표 탐지
            tables = page.find_tables()

            for table in tables.tables:

                # 표 영역 저장 (일반 본문과 중복 추출 방지)
                table_areas.append(pymupdf.Rect(table.bbox))

                # 4. 표의 행과 셀 텍스트 추출
                rows = []

                for row in table.rows:
                    cell_texts = []

                    for cell in row.cells:

                        if cell is None:
                            cell_texts.append(None)
                            continue

                        # 셀의 좌표를 이용해 텍스트 추출
                        cell_text = page.get_textbox(
                            pymupdf.Rect(cell)
                        )

                        cell_texts.append(cell_text.strip())

                    rows.append(cell_texts)

                if not rows:
                    continue

                # 5. 표의 헤더 확인
                if table.header.external:

                    # 표 바깥에 별도 헤더가 존재하는 경우
                    headers = []

                    for cell in table.header.cells:
                        if cell is not None:
                            header_text = page.get_textbox(
                                pymupdf.Rect(cell)
                            ).strip()
                        else:
                            header_text = ""

                        headers.append(header_text)

                    # 외부 헤더 영역도 중복 추출되지 않도록 제외
                    table_areas.append(
                        pymupdf.Rect(table.header.bbox)
                    )

                    data_rows = rows

                else:
                    # 첫 번째 행을 헤더로 사용
                    headers = rows[0]
                    data_rows = rows[1:]

                # 6. 헤더가 전혀 없으면 모든 행을 데이터로 처리
                if not any(headers):
                    headers = [""] * len(rows[0])
                    data_rows = rows

                # 7. 표의 각 행을 텍스트로 변환
                table_lines = []

                for row in data_rows:
                    cells = []

                    for i, value in enumerate(row):

                        # 비어 있는 셀 제외
                        if value is None or not str(value).strip():
                            continue

                        # 셀 내부 줄바꿈 및 연속 공백 정리
                        value = " ".join(str(value).split())

                        # 해당 열의 헤더 확인
                        if i < len(headers) and headers[i]:
                            header = " ".join(
                                str(headers[i]).split()
                            )
                            header = header.rstrip("：: ").strip()
                        else:
                            header = ""

                        # 헤더가 있는 경우 항목: 값 형식
                        if header:
                            cells.append(f"{header}: {value}")

                        # 헤더가 없는 경우 값만 출력
                        else:
                            cells.append(value)

                    # 한 행을 한 줄로 변환
                    if cells:
                        table_lines.append(" | ".join(cells))

                # 8. 표 텍스트 저장
                if table_lines:
                    elements.append(
                        (
                            table.bbox[1],
                            table.bbox[0],
                            "\n".join(table_lines),
                        )
                    )

            # 9. 표 이외의 일반 본문 추출
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:

                # 이미지 블록 등 텍스트가 아닌 요소 제외
                if "lines" not in block:
                    continue

                for line in block["lines"]:

                    line_rect = pymupdf.Rect(line["bbox"])

                    # 텍스트 라인 중심 좌표 계산
                    center = (
                        (line_rect.x0 + line_rect.x1) / 2,
                        (line_rect.y0 + line_rect.y1) / 2,
                    )

                    # 표 영역에 포함된 텍스트는 중복 추출 방지
                    if any(
                        area.contains(center)
                        for area in table_areas
                    ):
                        continue

                    # 같은 줄에 있는 텍스트 합치기
                    line_text = "".join(
                        span["text"]
                        for span in line["spans"]
                    )

                    if line_text.strip():
                        elements.append(
                            (
                                line_rect.y0,
                                line_rect.x0,
                                line_text,
                            )
                        )

            # 10. 페이지 내부 요소를 위치 기준으로 정렬
            elements.sort(
                key=lambda item: (item[0], item[1])
            )

            # 11. 페이지 텍스트 저장
            page_texts.append(
                "\n".join(
                    element[2]
                    for element in elements
                )
            )

    # 12. 전체 페이지의 텍스트 반환
    return "\n".join(page_texts)