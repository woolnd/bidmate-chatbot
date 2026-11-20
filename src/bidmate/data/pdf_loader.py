"""PDF 문서에서 본문과 표의 텍스트를 추출한다."""

import re
from pathlib import Path

import pymupdf

_LIST_START = re.compile(
    r"^(?:\d{1,3}(?:의\d+)?[.)]|[가나다라마바사아자차카타파하][.)]"
    r"|[①-⑳]|[○◯◦●❍※▪□]|[-–•]\s+)"
)


def _clean_line(text: str) -> str:
    """하나의 텍스트 줄에서 연속 공백만 정리한다."""
    return " ".join(text.split())


def _page_words(page) -> list:
    """한 페이지의 단어 좌표를 표 처리에 사용한다."""
    cache: dict = _page_words.__dict__.setdefault("cache", {})
    parent = id(page.parent)
    if cache and next(iter(cache))[0] != parent:
        cache.clear()
    key = (parent, page.number)
    if key not in cache:
        cache[key] = page.get_text("words")
    return cache[key]


def _repair_clipped_text(page, rect, text: str) -> str:
    """표의 셀 경계 때문에 앞부분이 잘린 텍스트를 복구"""
    box = pymupdf.Rect(rect)
    overlapping = []
    for word in _page_words(page):
        center_y = (word[1] + word[3]) / 2
        if not (box.y0 - 0.6 <= center_y <= box.y1 + 0.6):
            continue
        if word[2] > box.x0 and word[0] < box.x1:
            overlapping.append(word)
    if not overlapping:
        return text
    line_keys = {(word[5], word[6]) for word in overlapping}
    left = min(word[0] for word in overlapping)
    kept = list(overlapping)
    left_words = [
        word
        for word in _page_words(page)
        if (word[5], word[6]) in line_keys and word[2] <= left + 0.5
    ]
    for word in sorted(left_words, key=lambda item: item[0], reverse=True):
        if left - word[2] <= 8:
            kept.append(word)
            left = word[0]
        else:
            break
    repaired = _join_positioned_words(kept)
    token = _plain(text)
    repaired_token = _plain(repaired)
    if token and repaired_token.endswith(token) and 0 < len(repaired_token) - len(token) <= 12:
        return repaired
    return text


def _owned_cell_words(page, rect, row_band=None, other_bands=None) -> tuple[list, bool]:
    """칸 면적의 절반 넘게 들어온 단어만 그 칸의 글자로 본다.

    경계에 살짝 걸친 단어, 그리고 더 좁은 다른 행 안에 중심이 있는 단어는
    이 칸에 넣지 않는다. 두 번째 값은 그런 단어가 있었는지다.
    """
    box = pymupdf.Rect(rect)
    owned: list = []
    grazed = False
    row_height = None if row_band is None else row_band[1] - row_band[0]
    for word in _page_words(page):
        word_box = pymupdf.Rect(word[:4])
        word_area = word_box.width * word_box.height
        if word_area <= 0:
            continue
        center_y = (word[1] + word[3]) / 2
        if (
            row_height is not None
            and other_bands
            and any(y0 <= center_y < y1 and (y1 - y0) < row_height - 0.5 for y0, y1 in other_bands)
        ):
            grazed = True
            continue
        ix0 = max(word_box.x0, box.x0)
        iy0 = max(word_box.y0, box.y0)
        ix1 = min(word_box.x1, box.x1)
        iy1 = min(word_box.y1, box.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) / word_area >= 0.5:
            owned.append(word)
        else:
            grazed = True
    return owned, grazed


def _format_cell_lines(words: list) -> str:
    """같은 칸의 단어를 줄로 묶고, 목록이 아닌 줄은 한 문단으로 잇는다."""
    chunks = [_clean_line(line) for line in _join_positioned_words(words).splitlines()]
    chunks = [line for line in chunks if line]
    if not chunks:
        return ""
    lines = [chunks[0]]
    for item in chunks[1:]:
        if _LIST_START.match(item):
            lines.append(item)
        else:
            lines[-1] += " " + item
    return "\n".join(lines)


def _textbox_lines(page, rect) -> str:
    """단어 좌표가 없을 때만 칸 문자열을 직접 읽는다."""
    raw = page.get_textbox(pymupdf.Rect(rect))
    chunks = [_clean_line(line) for line in raw.splitlines()]
    chunks = [line for line in chunks if line]
    if not chunks:
        return ""
    lines = [chunks[0]]
    for item in chunks[1:]:
        if _LIST_START.match(item):
            lines.append(item)
        else:
            lines[-1] += " " + item
    return "\n".join(lines)


def _cell_text(page, rect, row_band=None, other_bands=None) -> str:
    """셀의 텍스트 추출을 총괄하고 필요하면 잘린 글자 복구"""
    if rect is None:
        return ""
    owned, grazed = _owned_cell_words(page, rect, row_band, other_bands)
    if owned:
        text = _format_cell_lines(owned)
    elif grazed:
        # 경계에만 걸친 글자는 이웃 칸의 것이다. 이 칸에 다시 넣지 않는다.
        return ""
    else:
        text = _textbox_lines(page, rect)
    if not text or "\n" in text:
        return text
    return _repair_clipped_text(page, rect, text)


def _plain(value: str) -> str:
    """추출 과정에서 공백이나 줄바꿈 차이 때문에 같은 내용을 다른 텍스트로 판단하지 않도록,
    비교용 문자열을 생성."""
    return re.sub(r"\s+", "", value or "")


def _is_value(value: str) -> bool:
    """데이터 코드, 숫자, 긴 설명 등 값에 가까운 셀인지 확인한다."""
    text = _plain(value)
    return bool(
        re.fullmatch(r"[+-]?\d[\d,.]*(?:%|원|개|명)?", text)
        or re.search(r"[A-Za-z]{2,}[-_/]\d", text)
        or len(text) >= 28
    )


def _short_label(value: str) -> bool:
    """항목 이름으로 사용할 만한 짧은 텍스트인지 확인한다."""
    text = _plain(value)
    return 1 <= len(text) <= 18 and not _is_value(value)


def _looks_like_header(rows: list[list[str]]) -> bool:
    """첫 행을 헤더로 확신할 때만 True를 반환한다."""
    start = next((index for index, row in enumerate(rows) if any(row)), None)
    if start is None or len(rows) - start < 3:
        return False
    first = rows[start]
    if len(first) < 2 or not all(first):
        return False
    if not all(_short_label(item) for item in first):
        return False
    body = [row for row in rows[start + 1 : start + 7] if any(row)]
    if len(body) < 2:
        return False
    contrasts = 0
    for column, head in enumerate(first):
        samples = [row[column] for row in body if column < len(row) and row[column]]
        if len(samples) < 2:
            continue
        value_ratio = sum(_is_value(sample) for sample in samples) / len(samples)
        if value_ratio >= 0.75 and not _is_value(head):
            contrasts += 1
    return contrasts >= 2


def _looks_like_key_value(rows: list[list[str]]) -> bool:
    """헤더 없는 항목-값 표만 식별한다. 불명확하면 Markdown으로 추출."""
    if len(rows) < 3 or len(rows[0]) not in (2, 3):
        return False
    if _looks_like_header(rows):
        return False

    # 다열 목록/집계표는 첫 열이 범주, 중간 열이 설명, 마지막 열이
    # 수량인 경우가 많다. 이 구조를 속성-값 폼으로 오분류하지 않는다.
    if len(rows[0]) == 3:
        sample = [row for row in rows if row[0] and row[1] and row[2]]
        if len(sample) >= 3:
            numeric_last = sum(
                bool(re.fullmatch(r"[+-]?\d[\d,.]*(?:%|원|개|명)?", _plain(row[2])))
                for row in sample
            )
            explanation_middle = sum(len(_plain(row[1])) >= 20 for row in sample)
            if numeric_last / len(sample) >= 0.6 and explanation_middle / len(sample) >= 0.6:
                return False

    meaningful = [row for row in rows if any(row)]
    paired = [row for row in meaningful if len([v for v in row if v]) >= 2]
    if len(paired) < 3 or len(paired) / len(meaningful) < 0.75:
        return False

    labels = [_plain(next((v for v in row if v), "")) for row in paired]
    if not all(_short_label(label) for label in labels):
        return False

    # 첫 열이 데이터 레코드마다 달라지는 목록표는 Key-Value가 아니다.
    # 상세 속성표는 요구사항번호/상세설명 등 명시적 속성명이나 반복 그룹명을 갖는다.
    property_terms = (
        "번호",
        "명칭",
        "요구사항",
        "상세설명",
        "산출정보",
        "관련요구사항",
        "정의",
        "사업명",
        "사업기간",
        "사업비",
        "담당자",
        "연락처",
    )
    has_property_label = sum(any(term in label for term in property_terms) for label in labels) >= 2
    repeated_label = len(set(labels)) < len(labels)
    if not (has_property_label or repeated_label):
        return False

    if _label_relation_is_unclear(rows):
        return False
    return any(_is_value(row[-1]) for row in paired)


def _label_relation_is_unclear(rows: list[list[str]]) -> bool:
    """한 칸짜리 짧은 라벨만 있는 행이 있으면 항목-값 관계가 불명확하다.

    병합으로 라벨과 값의 관계가 불분명한지 확인
    """
    for row in rows:
        filled = [value for value in row if value]
        if len(filled) == 1 and _short_label(filled[0]):
            return True
    return False


def _to_markdown(rows: list[list[str]], external: list[str] | None) -> str:
    """원본 데이터 행과 열을 보존한 Markdown 표를 만든다."""
    # PyMuPDF가 표 위에 빈 행을 감지하면 헤더 판별 이전에 제거한다.
    rows = list(rows)
    while len(rows) > 1 and not any(rows[0]):
        rows.pop(0)
    n_cols = len(rows[0])
    headers = [""] * n_cols
    body = rows
    if external and any(external):
        headers = external
    elif _looks_like_header(rows):
        headers, body = rows[0], rows[1:]

    def fmt(value: str) -> str:
        return value.replace("|", r"\|").replace("\n", "<br>")

    result = [
        "| " + " | ".join(fmt(v) for v in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in body)
    return "\n".join(result)


def _to_key_value(rows: list[list[str]]) -> str:
    """표를 key: value 형태로 변환"""
    lines = []
    for row in rows:
        values = [v for v in row if v]
        if not values:
            continue
        if len(values) == 1:
            # 원문에 없는 자리표시는 만들지 않는다. 칸에 있는 텍스트만 남긴다.
            lines.append(values[0])
            continue
        label_parts = [_clean_line(value) for value in values[:-1]]
        label = " > ".join(dict.fromkeys(label_parts))
        lines.append(f"{label}: {values[-1]}")
    return "\n".join(lines)


def _combine_continuation(parts: list[str]) -> str:
    """여러 텍스트 조각을 연결하되 번호 목록의 줄바꿈은 유지"""
    if not parts:
        return ""
    result = parts[0]
    for part in parts[1:]:
        if re.match(r"^\d{1,3}[.)]\s*", part) or _LIST_START.match(part):
            result += "\n" + part
        else:
            result += " " + part
    return result


def _spanned_property_cards(page, table, rows: list[list[str]]) -> str | None:
    """병합 때문에 여러 칸으로 쪼개진 요구사항 상세표를 분석해,
       관계가 유지되는 읽기 쉬운 텍스트로 재구성"""
    if not rows or len(rows[0]) != 4 or len(rows) < 5:
        return None
    if any(row[1] for row in rows):
        return None
    if len(table.rows) != len(rows):
        return None

    starts = []
    for i, (row, geo) in enumerate(zip(rows, table.rows)):
        rect0 = geo.cells[0]
        if rect0 is None or not row[0] or not row[3]:
            continue
        if geo.cells[1] is None and geo.cells[2] is None:
            bounds = pymupdf.Rect(rect0)
            value = pymupdf.Rect(geo.cells[3]) if geo.cells[3] else None
            if value and abs(bounds.x1 - value.x0) < 2:
                starts.append(i)

    if not starts or starts[0] != 0:
        return None

    outputs = []
    for start, end in zip(starts, starts[1:] + [len(rows)]):
        chunk = rows[start:end]
        if len(chunk) < 4:
            return None

        lines = [f"{chunk[0][0]}: {chunk[0][3]}"]
        pos = 1
        while pos < len(chunk):
            row = chunk[pos]
            if not any(row):
                pos += 1
                continue

            if row[0] and row[2] and row[3] and pos <= 2:
                lines.append(f"{_clean_line(row[0] + ' ' + row[2])}: {row[3]}")
                pos += 1
                continue
            break

        detail = chunk[pos:]
        if detail:
            primary_candidates = [r[0] for r in detail if r[0]]
            if not primary_candidates:
                return None
            primary = max(primary_candidates, key=lambda v: len(_plain(v)))

            if any(_plain(v) not in _plain(primary) for v in primary_candidates):
                return None

            group = []
            groups = []
            pending_label = ""
            for row in detail:
                secondary = _plain(row[2])
                value = row[3]

                if (
                    not secondary
                    and pending_label == "정의"
                    and value
                    and re.match(r"^\d{1,3}[.)]\s*", value)
                ):
                    if group:
                        groups.append((pending_label, group))
                    pending_label, group = "", [value]
                    continue
                if secondary == "정의":
                    if group:
                        groups.append((pending_label, group))
                    pending_label, group = "정의", [value] if value else []
                elif secondary and not pending_label:
                    pending_label = row[2]
                    if value:
                        group.append(value)
                elif (
                    secondary
                    and pending_label != "정의"
                    and len(_plain(pending_label)) <= 4
                    and len(secondary) <= 4
                ):
                    # '세부' + '내용'
                    pending_label = _clean_line(pending_label + " " + row[2])
                    if value:
                        group.append(value)
                elif secondary:
                    if group:
                        groups.append((pending_label, group))
                    pending_label, group = row[2], [value] if value else []
                elif value:
                    group.append(value)
            if group:
                groups.append((pending_label, group))
            if not groups or any(not label for label, _ in groups):
                return None
            for label, values in groups:
                lines.append(f"{primary} > {label}: {_combine_continuation(values)}")
        outputs.append("\n".join(lines))
    return "\n\n".join(outputs)


def _inferred_column_bounds(table, n_cols: int) -> list[tuple[float, float]] | None:
    """셀 좌표를 참고해 각 열의 경계를 추정"""
    anchors: list[list[tuple[float, float]]] = [[] for _ in range(n_cols)]
    for row in table.rows:
        for col, cell in enumerate(row.cells[:n_cols]):
            if cell is None:
                continue
            rect = pymupdf.Rect(cell)
            if rect.width > 1:
                anchors[col].append((rect.x0, rect.x1))

    if any(not values for values in anchors):
        return None

    for row in table.rows:
        if len(row.cells) < n_cols or any(cell is None for cell in row.cells[:n_cols]):
            continue
        rects = [pymupdf.Rect(cell) for cell in row.cells[:n_cols]]
        if all(abs(rects[i].x1 - rects[i + 1].x0) <= 2.5 for i in range(n_cols - 1)):
            return [(rect.x0, rect.x1) for rect in rects]

    bounds = []
    for values in anchors:
        xs = [a for a, _ in values]
        xe = [b for _, b in values]
        if max(xs) - min(xs) > 2.5 or max(xe) - min(xe) > 2.5:
            return None
        bounds.append((sum(xs) / len(xs), sum(xe) / len(xe)))
    if any(bounds[i][1] > bounds[i + 1][0] + 2.5 for i in range(n_cols - 1)):
        return None
    return bounds


def _recover_missing_cell_words(page, table, rows: list[list[str]]) -> list[list[str]]:
    """PyMuPDF가 None으로 반환한 셀에 원문 단어가 존재하면 복구"""
    if not rows or not any(cell is None for row in table.rows for cell in row.cells):
        return rows
    n_cols = len(rows[0])
    bounds = _inferred_column_bounds(table, n_cols)
    if bounds is None:
        return rows

    words = page.get_text("words")
    recovered = [row.copy() for row in rows]
    for row_i, geometry in enumerate(table.rows):
        if row_i >= len(recovered):
            break
        y0, y1 = geometry.bbox[1], geometry.bbox[3]
        if y1 <= y0:
            continue
        for col_i, cell in enumerate(geometry.cells[:n_cols]):
            if cell is not None or recovered[row_i][col_i]:
                continue
            x0, x1 = bounds[col_i]

            if any(
                other is not None
                and max(0.0, min(x1, pymupdf.Rect(other).x1) - max(x0, pymupdf.Rect(other).x0))
                > 0.25 * (x1 - x0)
                for k, other in enumerate(geometry.cells)
                if k != col_i
            ):
                continue
            selected = [
                w for w in words if x0 <= (w[0] + w[2]) / 2 < x1 and y0 <= (w[1] + w[3]) / 2 < y1
            ]
            if not selected:
                continue

            selected.sort(key=lambda w: (w[5], w[6], w[7]))
            lines: list[str] = []
            current_key = None
            current_words: list[str] = []
            for word in selected:
                key = (word[5], word[6])  # PDF block and line indices
                if current_key is not None and key != current_key:
                    lines.append(" ".join(current_words))
                    current_words = []
                current_key = key
                current_words.append(str(word[4]))
            if current_words:
                lines.append(" ".join(current_words))
            token = _plain("\n".join(lines))
            # 다른 칸과 글자가 같은 짧은 라벨만 건너뛴다. 부분 일치는 지우지 않는다.
            if token and lines and all(_short_label(line) for line in lines):
                already = {_plain(cell) for row in recovered for cell in row if cell}
                if token in already:
                    continue
            recovered[row_i][col_i] = "\n".join(lines)
    return recovered


def _join_positioned_words(words: list) -> str:
    """단어들의 위치 정보를 이용해 같은 줄의 단어를 합침"""
    ordered = sorted(words, key=lambda word: (word[5], word[6], word[7]))
    lines: list[str] = []
    current_key = None
    current_words: list[str] = []
    for word in ordered:
        key = (word[5], word[6])
        if current_key is not None and key != current_key:
            lines.append(" ".join(current_words))
            current_words = []
        current_key = key
        current_words.append(str(word[4]))
    if current_words:
        lines.append(" ".join(current_words))
    return "\n".join(lines)


def _place_unassigned_words(page, table, rows: list[list[str]]) -> list[list[str]]:
    """표 안에 있지만 어떤 셀에도 할당되지 않은 단어를 적절한 빈 셀에 배치

    이미 칸에 들어간 텍스트는 다시 넣지 않는다. 행·열이 정해지지 않는
    """
    if not rows:
        return rows
    n_cols = len(rows[0])
    n_rows = min(len(rows), len(table.rows))
    covered = [
        pymupdf.Rect(cell) for geometry in table.rows for cell in geometry.cells if cell is not None
    ]
    column_rects: list[list[pymupdf.Rect]] = [[] for _ in range(n_cols)]
    for geometry in table.rows:
        for index, cell in enumerate(geometry.cells[:n_cols]):
            if cell is not None:
                column_rects[index].append(pymupdf.Rect(cell))
    bbox = pymupdf.Rect(table.bbox)
    grouped: dict[tuple[int, int], list] = {}
    for word in page.get_text("words"):
        center_x = (word[0] + word[2]) / 2
        center_y = (word[1] + word[3]) / 2
        point = pymupdf.Point(center_x, center_y)
        if not bbox.contains(point) or any(rect.contains(point) for rect in covered):
            continue
        row_hits = []
        for row_index in range(n_rows):
            y0, y1 = table.rows[row_index].bbox[1], table.rows[row_index].bbox[3]
            if y0 <= center_y < y1 or (row_index == n_rows - 1 and y0 <= center_y <= y1):
                row_hits.append((y1 - y0, -y0, row_index))
        if not row_hits:
            continue
        row_index = min(row_hits)[2]
        geometry = table.rows[row_index]
        column_hits = []
        for column_index, rects in enumerate(column_rects):
            if column_index >= len(geometry.cells):
                continue
            for rect in rects:
                if rect.x0 - 0.8 <= center_x <= rect.x1 + 0.8:
                    missing = geometry.cells[column_index] is None
                    column_hits.append((0 if missing else 1, rect.width, column_index))
                    break
        if not column_hits:
            continue
        column_index = min(column_hits)[2]
        if rows[row_index][column_index]:
            continue
        grouped.setdefault((row_index, column_index), []).append(word)

    placed = [row.copy() for row in rows]
    existing = _plain("\n".join(cell for row in placed for cell in row))
    for (row_index, column_index), words in sorted(grouped.items()):
        text = _join_positioned_words(words)
        token = _plain(text)
        if not token or token in existing:
            continue
        placed[row_index][column_index] = text
        existing += token
    return placed


def _unplaced_line_elements(page, table, output: str) -> list[tuple[float, float, str]]:
    """셀에도 표 출력에도 없는, 표 상자 안의 단어 줄을 본문 요소로 돌려준다."""
    covered = [
        pymupdf.Rect(cell) for geometry in table.rows for cell in geometry.cells if cell is not None
    ]
    bbox = pymupdf.Rect(table.bbox)
    header = table.header
    if header.external:
        bbox.include_rect(pymupdf.Rect(header.bbox))
        for cell in header.cells:
            if cell is not None:
                covered.append(pymupdf.Rect(cell))
    loose = []
    for word in page.get_text("words"):
        center_x = (word[0] + word[2]) / 2
        center_y = (word[1] + word[3]) / 2
        point = pymupdf.Point(center_x, center_y)
        if bbox.contains(point) and not any(rect.contains(point) for rect in covered):
            loose.append(word)
    if not loose:
        return []
    flat = _plain(output)
    elements: list[tuple[float, float, str]] = []
    current_key = None
    current_words: list = []
    for word in sorted(loose, key=lambda item: (item[5], item[6], item[7])):
        key = (word[5], word[6])
        if current_key is not None and key != current_key:
            text = " ".join(str(item[4]) for item in current_words)
            if _plain(text) and _plain(text) not in flat:
                elements.append((current_words[0][1], current_words[0][0], text))
                flat += _plain(text)
            current_words = []
        current_key = key
        current_words.append(word)
    if current_words:
        text = " ".join(str(item[4]) for item in current_words)
        if _plain(text) and _plain(text) not in flat:
            elements.append((current_words[0][1], current_words[0][0], text))
    return elements


def _table_rows(page, table) -> list[list[str]]:
    """PyMuPDF가 탐지한 표의 각 행과 셀을 읽어 행렬로 만든다."""
    bands = [(row.bbox[1], row.bbox[3]) for row in table.rows]
    rows = []
    for index, row in enumerate(table.rows):
        others = [band for band_index, band in enumerate(bands) if band_index != index]
        rows.append([_cell_text(page, cell, bands[index], others) for cell in row.cells])
    if not rows:
        return []
    n_cols = max(map(len, rows))
    if n_cols == 0:
        return []
    rows = [row + [""] * (n_cols - len(row)) for row in rows]
    rows = _recover_missing_cell_words(page, table, rows)
    return _place_unassigned_words(page, table, rows)


def _external_headers(page, table, n_cols: int) -> list[str] | None:
    """PyMuPDF가 표 외부의 헤더로 탐지한 텍스트를 추출"""
    header = table.header
    if not header.external:
        return None
    result = [_cell_text(page, cell) for cell in header.cells]
    result = (result + [""] * n_cols)[:n_cols]
    return result if any(result) else None


def _table_rect(table) -> pymupdf.Rect:
    """표 상자의 면적을 계산할 사각형을 돌려준다."""
    return pymupdf.Rect(table.bbox)


def _select_tables(tables) -> list:
    """서로 겹치는 표 중 과도하게 큰 표나 중복 탐지된 표를 제외"""
    if len(tables) < 2:
        return list(tables)
    rects = [_table_rect(table) for table in tables]
    areas = [max(rect.width, 0) * max(rect.height, 0) for rect in rects]
    cell_counts = [
        sum(1 for row in table.rows for cell in row.cells if cell is not None) for table in tables
    ]
    drop: set[int] = set()
    for index, rect in enumerate(rects):
        if areas[index] <= 0:
            drop.add(index)
            continue
        for other, other_rect in enumerate(rects):
            if other == index or areas[other] <= 0:
                continue
            inter = pymupdf.Rect(rect)
            inter.intersect(other_rect)
            inter_area = max(inter.width, 0) * max(inter.height, 0)
            if inter_area / areas[other] < 0.75 or areas[other] >= areas[index] * 0.9:
                continue
            if (
                len(tables[other].rows) > len(tables[index].rows)
                or cell_counts[other] > cell_counts[index]
            ):
                drop.add(index)
                break
    for index, rect in enumerate(rects):
        if index in drop or areas[index] <= 0:
            continue
        for other in range(index):
            if other in drop or areas[other] <= 0:
                continue
            inter = pymupdf.Rect(rect)
            inter.intersect(rects[other])
            inter_area = max(inter.width, 0) * max(inter.height, 0)
            smaller = min(areas[index], areas[other])
            if smaller <= 0 or inter_area / smaller < 0.8:
                continue
            if cell_counts[index] < cell_counts[other] or (
                cell_counts[index] == cell_counts[other] and areas[index] > areas[other]
            ):
                drop.add(index)
            else:
                drop.add(other)
    return [table for index, table in enumerate(tables) if index not in drop]


def extract_pdf_text(file_path: Path) -> str:
    """PDF의 본문과 표를 페이지 내 좌표 순서대로 추출하여 반환한다."""
    page_texts: list[str] = []
    with pymupdf.open(str(file_path)) as pdf:
        for page in pdf:
            elements: list[tuple[float, float, str]] = []
            table_areas = []
            for table in _select_tables(page.find_tables().tables):
                rows = _table_rows(page, table)
                if not rows or not any(any(cell for cell in row) for row in rows):
                    continue
                n_cols = len(rows[0])
                external = _external_headers(page, table, n_cols)
                if external:
                    table_areas.append(pymupdf.Rect(table.header.bbox))

                restored = _spanned_property_cards(page, table, rows) if external is None else None
                if restored is not None:
                    output = restored
                elif n_cols == 1 and sum(bool(row[0]) for row in rows) == 1:
                    output = next(row[0] for row in rows if row[0])
                elif external is None and _looks_like_key_value(rows):
                    output = _to_key_value(rows)
                else:
                    output = _to_markdown(rows, external)

                table_areas.append(pymupdf.Rect(table.bbox))
                # 본문과 표의 경계를 유지해 다른 문장에 붙지 않도록 한다.
                elements.append((table.bbox[1], table.bbox[0], "\n" + output + "\n"))
                # 칸에 넣지 못한 표 안 단어는 표 영역에 가려 삭제되지 않게 남긴다.
                elements.extend(_unplaced_line_elements(page, table, output))

            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    rect = pymupdf.Rect(line["bbox"])
                    center = ((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
                    if any(area.contains(center) for area in table_areas):
                        continue
                    text = "".join(span["text"] for span in line["spans"])
                    if text.strip():
                        elements.append((rect.y0, rect.x0, text))

            elements.sort(key=lambda item: (item[0], item[1]))
            page_texts.append("\n".join(text for _, _, text in elements))
    return "\n".join(page_texts)
