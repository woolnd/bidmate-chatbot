"""PDF 문서에서 본문과 표의 텍스트를 추출한다."""

import re
from itertools import groupby
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


def _span_line_text(page) -> dict[tuple[int, int], str]:
    """블록·줄 번호로 원문 줄 문자열을 찾는다. 끝 공백은 유지한다."""
    cache: dict = _span_line_text.__dict__.setdefault("cache", {})
    parent = id(page.parent)
    if cache and next(iter(cache))[0] != parent:
        cache.clear()
    key = (parent, page.number)
    if key not in cache:
        mapping: dict[tuple[int, int], str] = {}
        for block_index, block in enumerate(page.get_text("dict")["blocks"]):
            if "lines" not in block:
                continue
            for line_index, line in enumerate(block["lines"]):
                mapping[(block_index, line_index)] = "".join(span["text"] for span in line["spans"])
        cache[key] = mapping
    return cache[key]


def _line_chunks(page, words: list) -> list[tuple]:
    """셀 안의 시각적 줄과, 그 줄이 공백으로 끝났는지를 돌려준다."""
    raw_lines = _span_line_text(page)
    ordered = sorted(words, key=lambda word: (word[5], word[6], word[7]))
    chunks = []
    for key, group in groupby(ordered, key=lambda word: (word[5], word[6])):
        line_words = list(group)
        raw = raw_lines.get(key, "")
        chunks.append(
            (
                _clean_line(" ".join(str(word[4]) for word in line_words)),
                min(word[0] for word in line_words),
                max(word[2] for word in line_words),
                min(word[1] for word in line_words),
                max(word[3] for word in line_words),
                raw.endswith((" ", "\u00a0")),
            )
        )
    chunks.sort(key=lambda chunk: (chunk[3], chunk[1]))
    return chunks


def _glue_broken_word(
    previous: str,
    nxt: str,
    previous_x1: float,
    next_x0: float,
    right_edge: float,
    left_edge: float,
    ended_with_space: bool,
) -> bool:
    """오른쪽 끝에서 잘려 다음 줄로 넘어간 한 글자만 붙인다.

    앞 줄이 공백으로 끝나면 단어 경계로 보고 붙이지 않는다.
    """
    if ended_with_space or not previous or not nxt:
        return False
    previous_token = previous.rsplit(" ", 1)[-1]
    next_token = nxt.split(" ", 1)[0]
    if not re.fullmatch(r"[가-힣A-Za-z]+", previous_token):
        return False
    if not re.fullmatch(r"[가-힣A-Za-z]", next_token):
        return False
    if right_edge - previous_x1 > 10:
        return False
    if next_x0 - left_edge > 22:
        return False
    return True


def _join_visual_lines(chunks: list, rect=None) -> str:
    """목록과 넓은 줄 간격은 나누고, 이어지는 줄만 한 문단으로 붙인다."""
    if not chunks:
        return ""
    heights = [chunk[4] - chunk[3] for chunk in chunks if chunk[4] > chunk[3]]
    line_height = sorted(heights)[len(heights) // 2] if heights else 12.0
    gaps = [
        chunks[index][3] - chunks[index - 1][4]
        for index in range(1, len(chunks))
        if chunks[index][3] >= chunks[index - 1][4]
    ]
    usual_gap = sorted(gaps)[len(gaps) // 2] if gaps else line_height * 0.35
    box = pymupdf.Rect(rect) if rect is not None else None
    right_edge = box.x1 if box is not None else max(chunk[2] for chunk in chunks)
    left_edge = box.x0 if box is not None else min(chunk[1] for chunk in chunks)
    lines = [chunks[0][0]]
    for index, chunk in enumerate(chunks[1:], 1):
        text, x0, _x1, y0, _y1, _ended = chunk
        if not text:
            continue
        previous = chunks[index - 1]
        gap = y0 - previous[4]
        if _LIST_START.match(text) or gap > max(usual_gap * 2.4, line_height * 0.9):
            lines.append(text)
            continue
        if _glue_broken_word(
            previous[0], text, previous[2], x0, right_edge, left_edge, previous[5]
        ):
            token = text.split(" ", 1)[0]
            rest = text[len(token) :].lstrip()
            lines[-1] += token + ((" " + rest) if rest else "")
        else:
            lines[-1] += " " + text
    return "\n".join(lines)


def _format_cell_lines(page, words: list, rect=None) -> str:
    """셀의 시각적 줄과 번호 목록을 보존해 텍스트로 변환한다."""
    return _join_visual_lines(_line_chunks(page, words), rect)


def _textbox_lines(page, rect) -> str:
    """단어 좌표가 없을 때 칸 문자열을 직접 읽는다."""
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
        text = _format_cell_lines(page, owned, rect)
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
        re.fullmatch(r"[+-]?\d[\d,.]*(?:%|[가-힣]{1,2})?", text)
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
                bool(re.fullmatch(r"[+-]?\d[\d,.]*(?:%|[가-힣]{1,2})?", _plain(row[2])))
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

    # 특정 속성명에 의존하지 않고 행의 값 분포로 구분한다.
    # 값이 적어도 두 행 이상에서 데이터(숫자·코드·긴 설명)로 보이는지 확인한다.
    # 단순한 2열 목록을 속성표로 오분류할 수 있으므로, 라벨 반복 또는
    # 값이 긴 설명인 행이 과반수인 경우에만 Key-Value로 변환한다.
    repeated_label = len(set(labels)) < len(labels)
    data_values = [_plain(row[-1]) for row in paired]
    value_like = sum(_is_value(value) for value in data_values)
    long_values = sum(len(value) >= 28 for value in data_values)
    if value_like < 2 or not (repeated_label or long_values > len(paired) / 2):
        return False

    if _label_relation_is_unclear(rows):
        return False
    return True


def _label_relation_is_unclear(rows: list[list[str]]) -> bool:
    """한 칸짜리 짧은 라벨만 있는 행이 있으면 항목-값 관계가 불명확하다.

    병합으로 라벨과 값의 관계가 불분명한지 확인
    """
    for row in rows:
        filled = [value for value in row if value]
        if len(filled) == 1 and _short_label(filled[0]):
            return True
    return False


def _expand_cell(value: str) -> str:
    """긴 셀을 목록/문단 경계에서 읽기 쉬운 여러 줄로 나눈다."""
    parts = [part.strip() for part in value.split("\n") if part.strip()]
    if not parts:
        return ""
    # 숫자 목록과 글머리표는 이미 _format_cell_lines에서 분리됨.
    # 동일 줄 안에 연속된 번호가 있는 경우에도 안전하게 나눈다.
    out = []
    for part in parts:
        # '1. ... 2. ...' 경계만 처리. 본문의 날짜/소수점은 건드리지 않는다.
        split = re.split(r"(?<=\S)\s+(?=\d{1,2}[.)]\s+)", part)
        out.extend(split)
    return "\n".join(out)


def _to_markdown(rows: list[list[str]], external: list[str] | None) -> str:
    """짧은 표는 Markdown, 긴 설명 셀은 행/열 관계가 보이는 블록으로 출력."""
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

    # 긴 텍스트를 <br>가 들어간 하나의 거대한 Markdown 행에 가두지 않는다.
    # 키워드 없이 문자 길이와 목록의 개행만으로 분기한다.
    long_cells = any(len(value) >= 180 or value.count("\n") >= 3 for row in body for value in row)
    if long_cells:
        result = []
        if any(headers):
            result.append(
                "표 열: "
                + " | ".join(f"{i + 1}={label}" for i, label in enumerate(headers) if label)
            )
        for index, row in enumerate(body, 1):
            if not any(row):
                continue
            result.append(f"행 {index}:")
            for col, value in enumerate(row):
                if not value:
                    continue
                column = headers[col] if col < len(headers) and headers[col] else f"열 {col + 1}"
                value = _expand_cell(value)
                result.append(f"  {column}: {value.replace(chr(10), chr(10) + '    ')}")
            result.append("")
        return "\n".join(result).strip()

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
        lines.append(f"{label}: {_expand_cell(values[-1])}")
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
    """병합된 4열 속성표를 셀 배치만으로 관계가 읽히도록 재구성한다.

    항목의 이름은 비교하지 않는다. 관계가 모호하면 원본 Markdown 표를 유지한다.
    """
    if not rows or len(rows[0]) != 4 or len(rows) < 5:
        return None
    if any(row[1] for row in rows) or len(table.rows) != len(rows):
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

            groups: list[tuple[str, list[str]]] = []
            pending_label = ""
            group: list[str] = []
            for row in detail:
                secondary = row[2]
                value = row[3]
                if secondary:
                    # 새로운 셀 텍스트는 새로운 항목으로 취급한다.
                    # 단어 길이나 특정 문자열만으로 라벨을 합치지 않는다.
                    if pending_label:
                        groups.append((pending_label, group))
                    pending_label = secondary
                    group = [value] if value else []
                elif value:
                    # 라벨이 없는 다음 행의 값은 앞 항목의 연속으로 취급한다.
                    if not pending_label:
                        return None
                    group.append(value)
            if pending_label:
                groups.append((pending_label, group))
            if not groups or any(not label or not values for label, values in groups):
                return None
            for label, values in groups:
                lines.append(f"{primary} > {label}: {_expand_cell(_combine_continuation(values))}")
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


def _unplaced_line_elements(
    page, table, output: str
) -> list[tuple[float, float, float, float, str]]:
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
    elements: list[tuple[float, float, float, float, str]] = []
    current_key = None
    current_words: list = []

    def emit(group: list) -> None:
        nonlocal flat
        text = " ".join(str(item[4]) for item in group)
        if _plain(text) and _plain(text) not in flat:
            elements.append(
                (
                    min(item[1] for item in group),
                    min(item[0] for item in group),
                    max(item[2] for item in group),
                    max(item[3] for item in group),
                    text,
                )
            )
            flat += _plain(text)

    for word in sorted(loose, key=lambda item: (item[5], item[6], item[7])):
        key = (word[5], word[6])
        if current_key is not None and key != current_key:
            emit(current_words)
            current_words = []
        current_key = key
        current_words.append(word)
    if current_words:
        emit(current_words)
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


def _column_gutters(
    spans: list[tuple[float, float, float, float]], page_width: float, page_height: float
) -> list[float]:
    """본문이 세로로 갈라지는 위치만 돌려준다. 한 단이면 빈 목록이다."""
    narrow = [
        span for span in spans if span[1] - span[0] <= page_width * 0.46 and span[1] > span[0]
    ]
    if len(narrow) < 8:
        return []
    if page_width > page_height * 1.15:
        centers = sorted((span[0] + span[1]) / 2 for span in narrow)
        best_gap = 0.0
        best_cut = None
        for previous, nxt in zip(centers, centers[1:]):
            gap = nxt - previous
            cut = (previous + nxt) / 2
            if gap > best_gap and page_width * 0.25 < cut < page_width * 0.75:
                best_gap = gap
                best_cut = cut
        if best_cut is not None and best_gap >= 40:
            return [best_cut]
        return []
    candidates = []
    for x_pos in range(int(page_width * 0.18), int(page_width * 0.82), 4):
        left = [span for span in narrow if span[1] <= x_pos + 2]
        right = [span for span in narrow if span[0] >= x_pos - 2]
        crossed = [span for span in narrow if span[0] + 4 < x_pos < span[1] - 4]
        if len(left) < 4 or len(right) < 4 or len(crossed) > max(2, int(len(narrow) * 0.12)):
            continue
        overlap = min(max(span[3] for span in left), max(span[3] for span in right)) - max(
            min(span[2] for span in left), min(span[2] for span in right)
        )
        if overlap >= 36:
            candidates.append(x_pos)
    if not candidates:
        return []
    groups = [[candidates[0]]]
    for x_pos in candidates[1:]:
        if x_pos - groups[-1][-1] <= 8:
            groups[-1].append(x_pos)
        else:
            groups.append([x_pos])
    gutters = []
    for group in groups:
        center = sum(group) / len(group)
        if not gutters or center - gutters[-1] >= page_width * 0.12:
            gutters.append(center)
    return gutters


def _column_index(x0: float, x1: float, gutters: list[float], page_width: float) -> int | None:
    """여러 단에 걸치면 None, 아니면 왼쪽부터 단 번호를 돌려준다."""
    if not gutters or x1 - x0 >= page_width * 0.62:
        return None
    center = (x0 + x1) / 2
    return sum(center > gutter for gutter in gutters)


def _body_continues(previous: tuple, item: tuple) -> bool:
    """같은 단에서 바로 아래 줄이면 한 문단의 연속으로 본다."""
    overlap = min(previous[2], item[2]) - max(previous[1], item[1])
    same_band = overlap > 0 or abs(previous[1] - item[1]) <= 24
    if not same_band:
        return False
    gap = item[0] - previous[3]
    height = max(previous[3] - previous[0], item[3] - item[0], 8)
    if gap < -2 or gap > height * 1.15:
        return False
    return not _LIST_START.match(item[4].strip())


def _join_body_line(accumulated: str, previous: tuple, item: tuple) -> str:
    """연속된 본문 줄에서 잘린 한 글자만 붙인다."""
    stripped = item[4].strip()
    right_edge = max(previous[2], item[2])
    left_edge = min(previous[1], item[1])
    if _glue_broken_word(
        previous[4].strip(),
        stripped,
        previous[2],
        item[1],
        right_edge,
        left_edge,
        previous[4].endswith((" ", "\u00a0")),
    ):
        token = stripped.split(" ", 1)[0]
        rest = stripped[len(token) :].lstrip()
        return accumulated.rstrip() + token + ((" " + rest) if rest else "")
    return accumulated.rstrip() + " " + stripped


def _arrange_page(elements: list[tuple], page_width: float, page_height: float) -> str:
    """단이 있으면 왼쪽 단을 끝까지 읽고, 이어서 다음 단을 읽는다.

    elements 항목은 (y0, x0, x1, y1, text, mergeable)이다.
    표처럼 순서를 나누면 안 되는 블록은 mergeable이 False다.
    """
    if not elements:
        return ""
    gutters = _column_gutters(
        [(item[1], item[2], item[0], item[3]) for item in elements if item[5]],
        page_width,
        page_height,
    )
    indexed = [(_column_index(item[1], item[2], gutters, page_width), item) for item in elements]
    indexed.sort(key=lambda pair: (pair[1][0], pair[1][1]))
    ordered: list[tuple] = []
    column_bucket: dict[int, list[tuple]] = {}

    def flush_columns() -> None:
        for column in sorted(column_bucket):
            ordered.extend(sorted(column_bucket[column], key=lambda item: (item[0], item[1])))
        column_bucket.clear()

    for column, item in indexed:
        if column is None:
            flush_columns()
            ordered.append(item)
        else:
            column_bucket.setdefault(column, []).append(item)
    flush_columns()

    parts: list[str] = []
    previous_line: tuple | None = None
    for item in ordered:
        if not item[4].strip():
            continue
        if not item[5] or previous_line is None or not _body_continues(previous_line, item):
            parts.append(item[4].strip())
            previous_line = item if item[5] else None
            continue
        parts[-1] = _join_body_line(parts[-1], previous_line, item)
        previous_line = item
    return "\n".join(parts)


def extract_pdf_text(file_path: Path) -> str:
    """PDF의 본문과 표를 페이지 순서와 단 순서로 추출하여 반환한다."""
    page_texts: list[str] = []
    with pymupdf.open(str(file_path)) as pdf:
        for page in pdf:
            elements: list[tuple] = []
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
                box = pymupdf.Rect(table.bbox)
                elements.append((box.y0, box.x0, box.x1, box.y1, "\n" + output + "\n", False))
                for line in _unplaced_line_elements(page, table, output):
                    elements.append((line[0], line[1], line[2], line[3], line[4], True))

            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    rect = pymupdf.Rect(line["bbox"])
                    center = pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
                    if any(area.contains(center) for area in table_areas):
                        continue
                    text = "".join(span["text"] for span in line["spans"])
                    if text.strip():
                        elements.append((rect.y0, rect.x0, rect.x1, rect.y1, text, True))

            page_texts.append(_arrange_page(elements, page.rect.width, page.rect.height))
    return "\n".join(page_texts)
