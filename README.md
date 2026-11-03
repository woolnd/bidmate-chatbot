# bidmate-chatbot

공공기관 RFP 문서를 검색해서 질문에 출처와 함께 답하는 RAG 챗봇입니다.
코드잇 스프린트 AI 중급 프로젝트 (2026.11.09 ~ 11.27)

> 역할, 일정, 협업 규칙은 노션 기획서를 봅니다. 이 README는 코드 실행 방법만 다룹니다.

## 처음 한 번 설정

```bash
# 1. uv 설치 (이미 있으면 건너뜀)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 레포 받기
git clone <레포 주소>
cd bidmate-chatbot

# 3. 패키지 설치 (Python 3.12도 자동으로 설치됨)
uv sync

# 4. 비밀값 파일 만들고 OPENAI_API_KEY 채우기
cp .env.example .env
```

원본 RFP 파일과 `data_list.csv`는 구글 드라이브에서 받아 `data/raw/` 에 넣습니다.
**이 폴더는 Git에 올라가지 않습니다. 절대 커밋하지 마세요.**

## 실행

```bash
uv run python scripts/build_index.py       # 문서를 잘라 Vector DB에 저장 (문서가 바뀔 때만)
uv run streamlit run app/streamlit_app.py  # 챗봇 화면
uv run pytest                              # 테스트
uv run ruff check . && uv run ruff format . # 코드 스타일 검사·정리
```

지금은 파일 구조와 역할 설명만 있습니다. 각자 자기 파일 맨 위 설명(받는 것 / 넘기는 것)을 보고 채웁니다.

## 내 파일 찾기

| 담당 | 파일 |
| --- | --- |
| 데이터-1 | `src/bidmate/data/loader.py` (hwp), `src/bidmate/data/chunker.py` |
| 데이터-2 | `src/bidmate/data/loader.py` (pdf, 텍스트 정리) |
| 데이터-3 | `src/bidmate/data/metadata.py`, `evaluation/` |
| 검색-1 | `src/bidmate/retrieval/project_finder.py` |
| 검색-2 | `src/bidmate/retrieval/vector_store.py` |
| 재웅 | `src/bidmate/generation/`, `src/bidmate/pipeline/`, `src/bidmate/service/`, `app/` |

파일을 열면 맨 위 설명과 `TODO(담당)` 아래에 할 일과 참고 키워드가 적혀 있습니다.

## 흐름

```
미리 한 번:  문서 읽기 → 조각 자르기 → 머리말 붙이기 → Chroma 저장

질문마다:   1. 질문 다듬기 → 2. 기관·사업 찾기 → 3. 문서 검색 → 4. 답변 만들기
```

단계 사이에 주고받는 형식은 `src/bidmate/schemas.py` 에 있습니다. **바꾸기 전에 PM에게 먼저 말해 주세요.**

## 작업 방식

1. `main`에서 새 브랜치를 만듭니다: `git switch -c feature/hwp-loader`
2. 작업 후 `uv run pytest` 와 `uv run ruff check .` 가 통과하는지 확인합니다.
3. Pull Request를 올리면 CI가 자동으로 검사합니다. 다른 사람 1명이 리뷰한 뒤 머지합니다.
4. 새 패키지가 필요하면 `uv add 패키지명` 으로 추가합니다 (`pyproject.toml` 과 `uv.lock` 도 함께 커밋).
