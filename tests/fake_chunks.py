"""지어낸 조각 10개. 진짜 데이터가 오기 전에 검색 파트가 저장·찾기를 먼저 만들 때 쓴다.

내용은 모두 지어낸 것이라 Git에 올려도 된다 (원본 RFP 아님).
3장 "조각 규칙"과 똑같은 모양이다: 이름표 4칸 + "[기관명 / 사업명]" 머리말 + 내용.
사업 3개: 가나다공단 이러닝(4조각), 라마대학교 학사 시스템(3조각), 바사재단 누리집(3조각).
사업마다 기본 정보 조각(-info) 1개가 들어 있다.

실행해 보기 (레포 맨 위 폴더에서 `uv run python`을 켜고 한 줄씩 입력):
    >>> from tests.fake_chunks import make_fake_chunks
    >>> chunks = make_fake_chunks()
    >>> len(chunks)                 # 10
    >>> print(chunks[0].page_content)
    >>> chunks[0].metadata
"""

from langchain_core.documents import Document

# (project_id = 원본 파일 이름, 기관명, 사업명)
ELEARNING = ("가나다공단_이러닝.hwp", "가나다공단", "통합 이러닝 플랫폼 구축")
ACADEMIC = ("라마대학교_학사시스템.pdf", "라마대학교", "차세대 학사 정보 시스템 구축")
HOMEPAGE = ("바사재단_누리집.hwp", "바사재단", "누리집 개편 및 유지관리")

# (사업, 조각 번호 끝, 내용)
FAKE_CONTENTS = [
    (
        ELEARNING,
        "info",
        "기본 정보: 발주 기관 가나다공단 | 예산 150,000,000원 | 입찰 마감일 2024-05-31",
    ),
    (
        ELEARNING,
        "0001",
        "사업 목적: 직원 교육을 하나의 온라인 학습 공간에서 받을 수 있도록 "
        "흩어진 이러닝 시스템을 통합한다. 사업 기간은 계약일로부터 8개월이다.",
    ),
    (
        ELEARNING,
        "0002",
        "기능 요구사항: 강의 수강, 진도 관리, 수료증 발급 기능을 제공한다. "
        "모바일에서도 수강할 수 있어야 하며 웹 접근성 인증을 받아야 한다.",
    ),
    (
        ELEARNING,
        "0003",
        "구분: 1차년도 | 금액: 75,000,000원\n구분: 2차년도 | 금액: 75,000,000원\n"
        "제안서는 나라장터 전자 제출로만 받는다.",
    ),
    (
        ACADEMIC,
        "info",
        "기본 정보: 발주 기관 라마대학교 | 예산 820,000,000원 | 입찰 마감일 2024-07-15",
    ),
    (
        ACADEMIC,
        "0001",
        "사업 범위: 수강 신청, 성적 관리, 졸업 심사 기능을 새 시스템으로 옮긴다. "
        "기존 학사 데이터 10년 치를 손실 없이 이관해야 한다.",
    ),
    (
        ACADEMIC,
        "0002",
        "참가 자격: 최근 3년 안에 대학 학사 시스템 구축 실적이 있는 업체. "
        "공동수급은 2개 업체까지 허용한다.",
    ),
    (
        HOMEPAGE,
        "info",
        "기본 정보: 발주 기관 바사재단 | 예산 48,000,000원 | 입찰 마감일 2024-03-20",
    ),
    (
        HOMEPAGE,
        "0001",
        "요구사항: 누리집 화면을 새로 디자인하고 게시판, 공지사항, 자료실을 개편한다. "
        "개편 후 12개월 동안 유지관리 인력 1명이 상주해야 한다.",
    ),
    (
        HOMEPAGE,
        "0002",
        "보안 요구사항: 개인정보는 암호화해 저장하고, 관리자 화면은 내부망에서만 접속할 수 있다.",
    ),
]


def make_fake_chunks() -> list[Document]:
    """지어낸 조각 10개를 Document 목록으로 돌려준다. 모양은 진짜 조각과 같다."""
    chunks: list[Document] = []
    for (project_id, agency_name, project_name), chunk_number, content in FAKE_CONTENTS:
        chunks.append(
            Document(
                page_content=f"[{agency_name} / {project_name}]\n{content}",
                metadata={
                    "chunk_id": f"{project_id}-{chunk_number}",
                    "project_id": project_id,
                    "agency_name": agency_name,
                    "project_name": project_name,
                },
            )
        )
    return chunks
