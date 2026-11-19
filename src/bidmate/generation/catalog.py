"""사업 전체 질문용 목록표 만들기 (에이전트 catalog 길, #60). 담당: 재웅

"예산이 가장 큰 사업", "대학교 사업 모두"처럼 사업 전체를 봐야 하는 질문은
조각 몇 개로는 답할 수 없다. 그래서 목록표(data_list.csv) 100줄을 글 하나로 만들어
4단계(answer_generator)에 조각 1개처럼 넘긴다.

받는 것: 없음 (목록표 경로는 config.PROJECT_LIST_PATH)
넘기는 것: Document 1개 (page_content = 목록표 글, metadata = 조각 이름표 4칸)
"""

import csv
from functools import lru_cache

from langchain_core.documents import Document

from bidmate.config import PROJECT_LIST_PATH

# 빈칸(예산·마감일)에 넣는 말. 원문에도 없음을 확인했다 (11/18)
MISSING_VALUE_TEXT = "문서에 없음 (발주 기관 문의 필요)"

# 4단계 출처 칸에 보이는 이름 (목록표 조각의 사업명)
CATALOG_PROJECT_NAME = "사업 목록표(data_list)"


def _budget_of(row: dict[str, str]) -> int:
    """목록표 한 줄의 예산(원)을 숫자로 돌려준다.

    받는 것: 목록표 한 줄 {"사업 금액": "130000000.0", ...} (csv는 값을 전부 문자열로 준다)
    넘기는 것: 예산(원). 빈칸("")이나 "0.0"이면 0 (정렬 때 맨 뒤로 간다)
    """
    value = row["사업 금액"]
    return int(float(value)) if value else 0


@lru_cache(maxsize=1)
def build_catalog_document() -> Document:
    """목록표 100줄을 4단계가 읽을 조각 1개로 만든다. 한 번만 만들고 기억해 둔다.

    받는 것: 없음
    넘기는 것: Document (page_content = 한 줄에 사업 하나, metadata = CHUNK_METADATA_KEYS 4칸)
    """
    # 1. 목록표 읽기 (config.PROJECT_LIST_PATH)
    #    쓰는 칸: 발주 기관 / 사업명 / 사업 금액 / 입찰 참여 마감일 / 사업 요약
    with open(PROJECT_LIST_PATH, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # 2. 예산이 큰 순서로 정렬 (예산 빈칸·0원은 맨 뒤)
    #    AI가 숫자 100개 중 가장 큰 값을 찾다 틀리지 않게, 맨 위만 보면 되게 한다
    sorted_rows = sorted(rows, key=_budget_of, reverse=True)

    # 3. 사업 하나를 한 줄로: "기관명 | 사업명 | 예산 | 마감일 | 사업 요약"
    #    예산은 "986,945,000원"처럼 쉼표, 빈칸·0원·마감일 빈칸은 MISSING_VALUE_TEXT
    lines = []
    for row in sorted_rows:
        budget = _budget_of(row)
        budget_text = f"{budget:,}원" if budget else MISSING_VALUE_TEXT
        deadline_text = row["입찰 참여 마감일"][:10] or MISSING_VALUE_TEXT  # "2024-09-09"
        summary_text = " ".join(row["사업 요약"].split())  # 줄바꿈·겹친 공백 → 띄어쓰기 하나
        agency_name, project_name = row["발주 기관"], row["사업명"]
        lines.append(
            f"{agency_name} | {project_name} | {budget_text} | {deadline_text} | {summary_text}"
        )

    # 4. 맨 앞에 설명 한 줄 + 칸 이름 줄을 붙이고 줄들을 잇는다
    #    예: "공공기관 RFP 사업 목록 (100개, 예산이 큰 순서)"
    header_lines = [
        f"공공기관 RFP 사업 목록 ({len(lines)}개, 예산이 큰 순서)",
        "기관명 | 사업명 | 예산 | 입찰 마감일 | 사업 요약",
    ]
    catalog_text = "\n".join(header_lines + lines)

    # 5. 조각 1개로 돌려준다
    #    이름표 4칸은 다른 조각과 같은 키로 맞춘다 (schemas.CHUNK_METADATA_KEYS)
    return Document(
        page_content=catalog_text,
        metadata={
            "chunk_id": "catalog",
            "project_id": "catalog",
            "agency_name": "",
            "project_name": CATALOG_PROJECT_NAME,
        },
    )
