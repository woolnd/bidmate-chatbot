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

import importlib.util
import re
import sys
from pathlib import Path

import syhwp
from syhwp import _hwp5

# 문장이 1., 가., ①, ❍ 같은 기호로 시작하면 목록으로 간주하고 줄바꿈 실행
_LIST_ITEM = re.compile(
    r"^(?:"
    r"\d{1,2}[.)]"
    r"|[가나다라마바사아자차카타파하][.)]"
    r"|[①-⑳]"
    r"|[ㅇ○◯◦●❍※▪□]"
    r")"
)


def _cell_text_keep_paragraphs(paragraphs: list) -> str:
    """HWP 표 셀의 내부 노드에서 텍스트·수식·그림 정보를 읽고,
    목록 기호가 있는 문단의 줄바꿈을 유지하면서 하나의 문자열로 반환
    """
    chunks: list[str] = []
    for paragraph in paragraphs:
        parts: list[str] = []

        def collect(node) -> None:
            """문단의 노드를 재귀적으로 순회하며 텍스트·수식·그림 표시를 수집한다.
               결과는 바깥 함수의 parts에 누적한다.
            """
            if node.tag == _hwp5.HWPTAG_PARA_TEXT:
                parts.append(_hwp5.decode_para_text(node.payload))
            elif node.tag == _hwp5.HWPTAG_CTRL_HEADER and len(node.payload) >= 4:
                control_id = node.payload[:4]
                if control_id == _hwp5._CTRL_ID_EQUATION:
                    script = _hwp5._equation_script(node)
                    parts.append(f"[수식: {script}]" if script else "[수식]")
                elif control_id == _hwp5._CTRL_ID_GSO:
                    parts.append("[그림]")
            for child in node.children:
                collect(child)

        collect(paragraph)
        for line in "".join(parts).splitlines():
            collapsed = " ".join(line.split())
            if collapsed:
                chunks.append(collapsed)
    if not chunks:
        return ""
    lines = [chunks[0]]
    for chunk in chunks[1:]:
        if _LIST_ITEM.match(chunk):
            lines.append(chunk)
        else:
            lines[-1] = f"{lines[-1]} {chunk}"
    return "\n".join(lines)


def _load_paragraph_reader():
    """HWP 읽기 프로그램에 _cell_text_keep_paragraphs() 기능을 사용하게 만드는 함수"""
    source = Path(_hwp5.__file__)
    module_name = "syhwp._hwp5_paragraphs"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"HWP 파서를 불러오지 못했습니다: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
        sys.modules.pop(module_name, None)
    module._cell_text = _cell_text_keep_paragraphs
    if _hwp5._cell_text is _cell_text_keep_paragraphs:
        raise RuntimeError("syhwp 전역 셀 함수가 바뀌었습니다")
    return module


_paragraph_reader = _load_paragraph_reader()


def _open_hwp(file_path: Path):
    """파일 형식을 확인하고 알맞은 파서로 HWP 문서를 연다.
    HWP5 -> syhwp._hwp5_paragraphs, HWPX -> syhwp.open()
    """
    path = str(file_path)
    if syhwp.detect_format(path) == "hwp5":
        return _paragraph_reader.read_document_hwp5(path)
    return syhwp.open(path)


def extract_hwp_text(file_path: Path) -> str:
    """HWP 본문, 표, 수식, 그림 표시를 문서 순서로 반환한다.

    헤더는 병합과 칸의 형태로만 추정한다. 확신이 없으면 헤더를 만들지 않고
    모든 행과 열을 남긴다. 단어 목록은 쓰지 않는다.
    """
    return _render_document(file_path, None)


def _render_document(file_path: Path, records: list[dict] | None) -> str:
    """HWP 문서의 본문과 표를 읽습니다.
    표의 행·열 및 병합 구조를 분석합니다.
    표를 Markdown 또는 Key-Value 형식으로 변환합니다.
    본문과 변환된 표를 합쳐 하나의 문자열로 반환합니다.
    """

    def squash(text: str) -> str:
        """표 셀 값을 비교하기 쉽도록 모든 공백을 제거한다."""
        return re.sub(r"\s+", "", text or "")

    def is_number(text: str) -> bool:
        """셀 내용이 숫자·백분율 형식인지 판단한다."""
        return bool(re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?%?", text.strip()))

    def is_date(text: str) -> bool:
        """셀 내용이 단순 날짜 형식인지 판단한다."""
        return bool(
            re.fullmatch(
                r"\d{2,4}[-./년]\d{1,2}(?:[-./월]\d{1,2}일?)?",
                text.strip(),
            )
        )

    def is_identifier(text: str) -> bool:
        """일반 표 머리글보다 데이터 코드에 가까운 영숫자 식별자인지 판단한다."""
        # SFR-001, ATP10처럼 문자와 숫자가 함께 있는 데이터 코드.
        # M1, M18은 짧게 두어 일정 하위 헤더로 남긴다.
        compact = squash(text)
        if re.search(r"[A-Za-z]{2,}[-–_/].*\d", text or ""):
            return True
        return bool(
            len(compact) > 8 and re.search(r"\d", compact) and re.search(r"[A-Za-z가-힣]", compact)
        )

    def is_short_token(text: str) -> bool:
        """헤더 후보로 볼 수 있는 짧은 문자 토큰인지 판단한다."""
        compact = squash(text)
        if not (1 <= len(compact) <= 8):
            return False
        if is_number(text) or is_date(text) or is_identifier(text):
            return False
        return bool(re.search(r"[A-Za-z가-힣]", compact))

    def is_short_label(text: str) -> bool:
        """항목 이름이나 열 제목으로 쓸 수 있는 짧은 레이블인지 판단한다."""
        compact = squash(text)
        if not (1 <= len(compact) <= 18):
            return False
        if is_number(text) or is_date(text) or is_identifier(text):
            return False
        if re.search(r"[.。]", text or ""):
            return False
        return bool(re.search(r"[A-Za-z가-힣]", compact))

    def is_value_like(text: str) -> bool:
        """날짜·숫자·코드·긴 설명 등 값으로 볼 가능성이 높은 텍스트인지 판단한다."""
        compact = squash(text)
        if not compact:
            return False
        if is_number(text) or is_date(text) or is_identifier(text):
            return True
        if len(compact) >= 24:
            return True
        return bool(re.search(r"[가-힣]", compact) and len(compact) >= 20)

    def origins_on(row_index: int):
        """지정 행에서 새로 시작한 비어 있지 않은 원본 셀을 중복 없이 반환한다.
           병합 셀의 연장 부분은 제외한다.
        """
        found = []
        seen = set()
        for col in range(n_cols):
            cell_id = owners[row_index][col]
            if cell_id is None or cell_id in seen:
                continue
            source_row, source_col, row_span, col_span, value = cells[cell_id]
            if source_row != row_index or not value:
                continue
            seen.add(cell_id)
            found.append((source_col, row_span, col_span, value, cell_id))
        return found

    def row_is_subheader(row_index: int) -> bool:
        """해당 행의 원본 셀들이 짧은 레이블로 구성된 하위 헤더 후보인지 검사한다."""
        started = origins_on(row_index)
        if len(started) < 2:
            return False
        for _, _, _, value, _ in started:
            if is_value_like(value):
                return False
            if not (is_short_label(value) or is_short_token(value)):
                return False
        return True

    def row_is_card_shaped(row_index: int) -> bool:
        """해당 행이 왼쪽 항목과 오른쪽 값을 대응시키는 구조인지 판별한다."""
        # 왼쪽은 짧은 항목, 오른쪽은 그 행의 값. 물리 열 수와는 별개다.
        started = origins_on(row_index)
        if not (2 <= len(started) <= 3):
            return False
        left = started[0][3]
        if not (is_short_label(left) or is_short_token(left)):
            return False
        for _, _, _, value, _ in started[1:-1]:
            if not (is_short_label(value) or is_short_token(value)):
                return False
        right_col, _, right_span, right, _ = started[-1]
        if squash(right) == squash(left):
            return False
        # 값 칸이 행의 오른쪽 끝까지 닿지 않으면 빈칸 양식의 항목명이다.
        if right_col + right_span < n_cols:
            return False
        middle = squash(started[1][3])
        if len(started) == 3 and not (
            is_value_like(right) or is_identifier(right) or len(squash(right)) >= len(middle) + 4
        ):
            return False
        return True

    def children_under(row_index: int, col: int, col_span: int):
        """가로 병합 셀 아래 행에서 그 범위에 시작하는 하위 셀을 찾는다."""
        found = []
        seen = set()
        nxt = row_index + 1
        if nxt >= n_rows:
            return found
        for covered in range(col, col + col_span):
            cell_id = owners[nxt][covered]
            if cell_id is None or cell_id in seen:
                continue
            seen.add(cell_id)
            source_row, source_col, row_span, span, value = cells[cell_id]
            if source_row == nxt and value and col <= source_col < col + col_span:
                found.append((source_col, row_span, span, value, cell_id))
        return found

    def looks_like_grouped_header(start_row: int) -> bool:
        """병합된 상위 제목이 여러 하위 열 제목으로 나뉘는 다단 헤더인지 판단한다."""
        # 가로로 합쳐진 칸이 다음 행에서 둘 이상의 짧은 열 이름으로 나뉠 때만
        # 다단 헤더로 본다. 항목 칸이 합쳐진 요구사항 카드는 여기서 제외한다.
        if start_row + 1 >= n_rows:
            return False
        split_found = False
        for col, row_span, col_span, value, _ in origins_on(start_row):
            if not value or row_span != 1 or col_span < 2 or col_span >= n_cols:
                continue
            kids = children_under(start_row, col, col_span)
            if len(kids) < 2:
                continue
            if all(
                (is_short_label(kid) or is_short_token(kid)) and not is_value_like(kid)
                for *_, kid, _ in kids
            ):
                split_found = True
                break
        if not split_found or not row_is_subheader(start_row + 1):
            return False
        # 위 행이 항목|값이고, 아래 행도 그 값 열을 그대로 쓰면 다음 항목이다.
        parent = origins_on(start_row)
        child = origins_on(start_row + 1)
        if row_is_card_shaped(start_row) and parent and child and parent[-1][0] == child[-1][0]:
            value_col = parent[-1][0]
            lefts = [value for col, _, _, value, _ in child if col < value_col]
            if lefts and all(is_short_label(value) or is_short_token(value) for value in lefts):
                return False
        return True

    def body_row_indexes(start_row: int, limit: int = 8):
        """시작 행부터 제한된 범위 안에서 실제 셀이 시작하는 본문 행 인덱스를 반환한다."""
        indexes = []
        for row_index in range(start_row, min(n_rows, start_row + limit)):
            if origins_on(row_index):
                indexes.append(row_index)
        return indexes

    def column_samples(col: int, row_indexes):
        """지정 열의 본문 행들에서 새로 시작하는 비어 있지 않은 값을 모은다."""
        samples = []
        for row_index in row_indexes:
            cell_id = owners[row_index][col]
            if cell_id is None:
                continue
            source_row, _, _, _, value = cells[cell_id]
            if source_row == row_index and value:
                samples.append(value)
        return samples

    def has_column_contrast(head: str, samples: list[str]) -> bool:
        """열 제목과 아래 데이터 값들의 형식 차이가 충분히 뚜렷한지 판단한다."""
        if not samples:
            return False
        kinds = []
        for value in samples:
            if is_number(value):
                kinds.append("number")
            elif is_date(value):
                kinds.append("date")
            elif is_identifier(value):
                kinds.append("code")
            elif re.fullmatch(r"[A-Za-z][A-Za-z\s/()._-]*", value.strip()):
                kinds.append("latin")
            elif re.search(r"[가-힣]", value):
                kinds.append("korean")
            else:
                kinds.append("other")
        common = max(set(kinds), key=kinds.count)
        if kinds.count(common) / len(kinds) < 0.75:
            return False
        if not re.search(r"[A-Za-z가-힣]", head):
            return False
        if common in {"number", "date", "code"}:
            return True
        head_latin = bool(re.fullmatch(r"[A-Za-z][A-Za-z\s/()._-]*", head.strip()))
        if common == "latin" and not head_latin:
            return True
        if common == "latin" and head_latin:
            median = sorted(len(squash(value)) for value in samples)[len(samples) // 2]
            return median >= len(squash(head)) + 4
        return False

    def looks_like_column_header(start_row: int) -> bool:
        """첫 행의 후보 레이블과 아래 행의 값 길이·형식을 비교해 단일 헤더 여부를 판단한다."""
        started = origins_on(start_row)
        if len(started) < 2:
            return False
        for _, _, _, value, _ in started:
            if is_value_like(value):
                return False
            if not (is_short_label(value) or is_short_token(value)):
                return False
        body = body_row_indexes(start_row + 1)
        if not body:
            return False
        strong = 0
        longer = 0
        considered = 0
        for col, _, _, head, _ in started:
            samples = column_samples(col, body)
            if not samples:
                continue
            considered += 1
            if has_column_contrast(head, samples):
                strong += 1
            median = sorted(len(squash(value)) for value in samples)[len(samples) // 2]
            if median >= len(squash(head)) + 8:
                longer += 1
        if considered == 0:
            return False
        return strong >= 1 or longer * 2 >= considered

    def peer_header_row(row_index: int) -> bool:
        """병합 없는 한 행의 셀들이 모두 짧은 제목 후보인지 확인한다."""
        started = origins_on(row_index)
        if len(started) < 2:
            return False
        if any(row_span != 1 or col_span != 1 for _, row_span, col_span, _, _ in started):
            return False
        values = [value for *_, value, _ in started]
        if any(is_identifier(value) or is_value_like(value) for value in values):
            return False
        return all(is_short_label(value) or is_short_token(value) for value in values)

    def looks_like_time_header(start_row: int) -> bool:
        """M, M+1, 1W 등 일정 시점을 나타내는 열 제목이 충분히 많은지 검사한다."""
        # M, M+1, 1W처럼 짧은 시점 이름이 여러 칸이면 일정 헤더로 본다.
        started = origins_on(start_row)
        if len(started) < 6:
            return False
        times = 0
        for _, _, _, value, _ in started:
            if is_value_like(value):
                return False
            if re.fullmatch(r"M(?:\+\d+)?|\d+W", squash(value), re.I):
                times += 1
        return times >= 4

    def plain_parallel_header(start_row: int) -> bool:
        """병합 없는 표의 첫 행이 실제 열 헤더인지 추정한다.
           첫 행 값이 아래 행에 반복되면 일반 데이터 행으로 간주해 헤더 승격을 방지한다.
        """
        # 병합도 식별자도 없는 표에서 열 헤더를 판별한다.
        if not peer_header_row(start_row):
            return False

        for _, _, row_span, col_span, value in cells:
            if row_span > 1 or col_span > 1:
                return False
            if value and is_identifier(value):
                return False

        body = body_row_indexes(start_row + 1)

        if len(body) < 2:
            return False

        # 첫 행의 값이 아래 행들에서도 반복되면
        # 열 헤더가 아니라 일반 데이터 행일 가능성이 높다.
        for col in range(n_cols):
            head_value = grid[start_row][col]

            if not head_value:
                continue

            samples = [grid[row][col] for row in body if grid[row][col]]

            if len(samples) < 3:
                continue

            repeated = sum(squash(value) == squash(head_value) for value in samples)

            if repeated >= 2 and repeated / len(samples) >= 0.75:
                return False

        return True

    def content_rows(start_row: int):
        """시작 위치부터 비어 있지 않은 데이터 행 인덱스를 반환한다."""
        return [row for row in range(start_row, n_rows) if any(grid[row])]

    def infer_anchor(start_row: int) -> int:
        """항목-값 형태의 표에서 값이 주로 놓이는 기준 열(anchor)을 추정한다."""
        counts: dict[int, int] = {}
        for row in content_rows(start_row):
            found = [(col, grid[row][col]) for col in range(n_cols) if grid[row][col]]
            if len(found) == 2:
                col = found[-1][0]
                counts[col] = counts.get(col, 0) + 1
        if counts:
            return sorted(counts, key=lambda col: (-counts[col], col))[0]
        for row in content_rows(start_row):
            found = [(col, grid[row][col]) for col in range(n_cols) if grid[row][col]]
            if len(found) >= 2:
                return found[-1][0]
        return max(n_cols - 1, 0)

    def row_supports_card(row_index: int) -> bool:
        """값이 비어 있는 행과 병합 구조를 포함해 항목-값 표의 행 패턴을 판단한다."""
        started = origins_on(row_index)
        if row_is_card_shaped(row_index):
            return True
        # 값은 비어 있고 항목만 있는 행도 카드의 한 줄이다.
        if len(started) == 1 and (is_short_label(started[0][3]) or is_short_token(started[0][3])):
            return True
        # 코드형 식별자가 있는 카드에서만, 정의처럼 짧은 값도 오른쪽 끝까지면 값으로 본다.
        has_code = any(re.search(r"[A-Za-z]{2,}[-–_/].*\d", value or "") for *_, value in cells)
        if len(started) == 3 and has_code:
            col, _, span, value, _ = started[-1]
            left_ok = all(
                is_short_label(text) or is_short_token(text) for _, _, _, text, _ in started[:-1]
            )
            if left_ok and col + span >= n_cols and squash(value) != squash(started[0][3]):
                return True
        return False

    def looks_like_key_value(start_row: int) -> bool:
        """표의 셀 배치·병합·값 특성을 검토해 Markdown 대신 항목-값 텍스트로 변환할지 결정한다."""
        # 물리 열이 많아도, 실제 칸이 항목과 값으로 놓여 있으면 카드로 본다.
        # 첫 행이 열 헤더이거나 숫자 측정 표이면 Markdown으로 남긴다.
        if n_cols < 2 or peer_header_row(start_row) or plain_parallel_header(start_row):
            return False
        content = content_rows(start_row)
        if len(content) < 3:
            return False
        paired = sum(row_is_card_shaped(row) for row in content)
        if paired < 2:
            return False
        matched = sum(row_supports_card(row) for row in content)
        if matched / len(content) < 0.75:
            return False
        # 식별자도 긴 값도 없으면 빈 양식으로 보고 Markdown에 둔다.
        # 제목 행으로 빠진 식별자도 표 전체에서 인정한다.
        if not any(value and (is_identifier(value) or is_value_like(value)) for *_, value in cells):
            return False
        # 세로로 묶인 항목이나 식별자가 없으면 일반 표로 남긴다.
        if not any(row_span > 1 and value for _, _, row_span, _, value in cells) and not any(
            value and is_identifier(value) for *_, value in cells
        ):
            return False
        anchor = infer_anchor(start_row)
        if sum(bool(grid[row][anchor]) for row in content) / len(content) < 0.5:
            return False
        for col in range(1, n_cols):
            values = [grid[row][col] for row in content if grid[row][col]]
            if len(values) >= 3 and sum(is_number(value) for value in values) / len(values) >= 0.6:
                return False
        wide = sum(1 for row in content if sum(bool(grid[row][col]) for col in range(n_cols)) >= 4)
        if wide and wide / len(content) >= 0.6:
            return False
        return True

    def looks_like_toc() -> bool:
        """2열 표가 페이지 번호 중심의 목차인지 판단한다."""
        # 점선 안내나 긴 제목이 있는 쪽 번호 목록만 목차로 본다.
        if n_cols != 2 or n_rows < 5:
            return False
        pairs = [
            (grid[row][0].strip(), grid[row][1].strip())
            for row in range(n_rows)
            if grid[row][0] or grid[row][1]
        ]
        if len(pairs) < 5:
            return False
        page_rows = [(left, right) for left, right in pairs if re.fullmatch(r"\d{1,3}", right)]
        if len(page_rows) / len(pairs) < 0.8:
            return False
        leaders = sum(bool(re.search(r"[·\.…]{2,}|-{2,}", left)) for left, _ in page_rows)
        long_left = sum(len(squash(left)) >= 8 for left, _ in page_rows)
        if leaders == 0 and long_left / len(page_rows) < 0.8:
            return False
        left0, right0 = pairs[0]
        if (is_short_label(left0) or is_short_token(left0)) and (
            is_short_label(right0) or is_short_token(right0)
        ):
            return False
        return True

    def header_path(col: int, depth: int, start_row: int) -> str:
        """단일 또는 다단 헤더의 각 열에 대응되는 제목 경로를 생성한다."""
        # 가로로 덮인 열에도 같은 헤더 문구를 반복한다.
        # Markdown은 병합을 표현하지 못하므로 열을 지우지 않는다.
        parts = []
        seen = set()
        for row in range(start_row, start_row + depth):
            cell_id = owners[row][col]
            if cell_id is None or cell_id in seen:
                continue
            source_row, _, _, _, value = cells[cell_id]
            seen.add(cell_id)
            if source_row == row and value and value not in parts:
                parts.append(value)
        return " / ".join(parts)

    def fill_rowspan() -> None:
        """세로 병합된 셀의 값을 아래 관련 행으로 전달해 항목-값 관계를 보존한다."""
        for row, col, row_span, col_span, value in cells:
            if row_span <= 1 or not value or col_span == n_cols:
                continue
            for next_row in range(row + 1, min(row + row_span, n_rows)):
                if not any(grid[next_row]):
                    continue
                if not grid[next_row][col]:
                    grid[next_row][col] = value

    def markdown_cell(value: str) -> str:
        """표 셀 내부 줄바꿈을 Markdown 테이블에서 안전한 <br> 표기로 바꾼다."""
        # 문단 경계는 칸 안에 남긴다. 표의 행이 끊기지 않게 br로 적는다.
        return (value or "").replace("\n", "<br>")

    def markdown(headers: list[str], rows: list[list[str]]) -> str:
        """헤더와 데이터 행을 원본 열 개수대로 Markdown 표 문자열로 직렬화한다."""
        lines = [
            "| " + " | ".join(markdown_cell(header) for header in headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
        return "\n".join(lines)

    def key_value_lines(start_row: int) -> list[str]:
        """표의 항목 열과 값 열을 연결해 읽기 쉬운 Key-Value 텍스트 행들을 만든다."""
        anchor = infer_anchor(start_row)
        lines = []
        for row in range(start_row, n_rows):
            found = [(col, grid[row][col]) for col in range(n_cols) if grid[row][col]]
            if not found:
                continue
            if len(found) == 1:
                lines.append(found[0][1])
                continue
            labels = []
            value = None
            extras = []
            for col, text in found:
                if col < anchor:
                    if text not in labels:
                        labels.append(text)
                elif col == anchor and value is None:
                    value = text
                else:
                    extras.append(text)
            if labels and value is not None:
                lines.append(f"{' > '.join(labels)}: {value}")
            elif value is not None:
                lines.append(value)
            elif labels:
                lines.append(" > ".join(labels))
            index = 0
            while index + 1 < len(extras):
                lines.append(f"{extras[index]}: {extras[index + 1]}")
                index += 2
            if index < len(extras):
                lines.append(extras[index])
        return lines

    def remember(mode: str, text: str, headers: list[str]) -> None:
        """검증용 records가 요청된 경우 표 유형, 원본 셀, 변환 결과를 기록한다."""
        if records is None:
            return
        records.append(
            {
                "mode": mode,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "headers": list(headers),
                "text": text,
                "cells": [value for *_, value in cells if value],
                "origins": [
                    {"row": row, "col": col, "row_span": row_span, "text": value}
                    for row, col, row_span, _, value in cells
                    if value
                ],
            }
        )

    doc = _open_hwp(file_path)
    elements: list[str] = []

    for block in doc.blocks:
        kind = type(block).__name__

        if kind == "Paragraph":
            paragraph = getattr(block, "text", "") or ""
            if paragraph.strip():
                elements.append(paragraph)
            continue

        if kind == "Equation":
            script = getattr(block, "script", "") or ""
            if script:
                elements.append(f"[수식: {script}]")
            continue

        if kind == "Image":
            elements.append("[그림]")
            continue

        if kind != "Table":
            continue

        n_rows = int(getattr(block, "n_rows", 0) or 0)
        n_cols = int(getattr(block, "n_cols", 0) or 0)
        if n_rows <= 0 or n_cols <= 0:
            continue

        grid = [[""] * n_cols for _ in range(n_rows)]
        owners = [[None] * n_cols for _ in range(n_rows)]
        cells = []

        for source in block.cells:
            row = int(source.row)
            col = int(source.col)
            if not (0 <= row < n_rows and 0 <= col < n_cols):
                continue
            row_span = max(1, min(int(getattr(source, "row_span", 1) or 1), n_rows - row))
            col_span = max(1, min(int(getattr(source, "col_span", 1) or 1), n_cols - col))
            raw_text = str(getattr(source, "text", "") or "")
            value = "\n".join(
                " ".join(line.split()).replace("|", r"\|")
                for line in raw_text.splitlines()
                if line.strip()
            )
            cell_id = len(cells)
            cells.append((row, col, row_span, col_span, value))
            grid[row][col] = value
            for covered_row in range(row, row + row_span):
                for covered_col in range(col, col + col_span):
                    if owners[covered_row][covered_col] is None:
                        owners[covered_row][covered_col] = cell_id

        if not any(value for *_, value in cells):
            continue

        if n_cols == 1 and sum(bool(value) for *_, value in cells) == 1:
            text = next(value for *_, value in cells if value)
            elements.append(text)
            remember("onecell", text, [])
            continue

        if looks_like_toc():
            lines = [
                f"{grid[row][0]} {grid[row][1]}".strip()
                for row in range(n_rows)
                if grid[row][0] or grid[row][1]
            ]
            elements.extend(lines)
            remember("toc", "\n".join(lines), [])
            continue

        fill_rowspan()

        title_parts = []
        start = 0
        first_cells = [cell for cell in cells if cell[0] == 0 and cell[4]]
        if n_rows >= 3 and len(first_cells) == 1:
            _, col, _, col_span, first_text = first_cells[0]
            second_count = len([cell for cell in cells if cell[0] == 1 and cell[4]])
            if col == 0 and col_span == n_cols and second_count >= 2:
                title_parts.append(first_text)
                start = 1

        header_depth = 0
        card = False
        if looks_like_grouped_header(start):
            header_depth = 2
        elif looks_like_key_value(start):
            card = True
        elif (
            looks_like_time_header(start)
            or plain_parallel_header(start)
            or looks_like_column_header(start)
        ):
            header_depth = 1

        if card and header_depth == 0:
            output = "\n".join(part for part in [*title_parts, *key_value_lines(start)] if part)
            if output:
                elements.append(output)
                remember("kv", output, [])
            continue

        # 원본 열 수를 그대로 둔다. 본문이 비어도 헤더 열을 삭제하지 않는다.
        headers = [
            header_path(col, header_depth, start) if header_depth else "" for col in range(n_cols)
        ]
        data_rows = []
        for row in range(start + header_depth, n_rows):
            values = [grid[row][col] for col in range(n_cols)]
            if any(values):
                data_rows.append(values)

        pieces = list(title_parts)
        if data_rows or any(headers):
            pieces.append(markdown(headers, data_rows))
        output = "\n".join(part for part in pieces if part)
        if output:
            elements.append(output)
            mode = "multi" if header_depth == 2 else "single" if header_depth == 1 else "plain"
            remember(mode, output, headers)

    # 추출 결과에 남은 UTF-16 surrogate 쌍을 유효한 Unicode 문자로 결합한다.
    # 단독 surrogate는 UTF-8 저장이 가능하도록 U+FFFD로 대체한다.
    text = "\n".join(elements)
    return text.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
