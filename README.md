# bidmate-chatbot

공공기관 RFP(제안요청서) 100개를 검색해서, 질문에 **출처와 함께** 답하는 RAG 챗봇입니다.
코드잇 스프린트 AI 중급 프로젝트 (2026.11.09 ~ 11.27)

> B2G 입찰 컨설팅 스타트업 '입찰메이트'의 엔지니어링 팀이라고 가정합니다.
> 컨설턴트가 RFP를 일일이 읽지 않고도 요구 조건, 발주 기관, 예산, 제출 방식을 빠르게 파악하도록 돕습니다.

## 흐름

<img width="3600" height="2380" alt="베이스라인_흐름도" src="https://github.com/user-attachments/assets/dc640768-a855-4e22-85cd-eae7655f7c60" />


- **[A] 미리 한 번**: 목록표와 원본 문서를 읽어 1000자 조각으로 자르고(200자 겹침), 조각마다 `[기관명 / 사업명]`을 붙여 Chroma에 저장합니다. 문서마다 예산·마감일을 담은 기본 정보 조각을 하나 더 만듭니다.
- **[B] 질문마다**: ① 질문 다듬기 → ② 어느 문서인지 고르기 → ③ 그 문서 안에서 관련 조각 5개 찾기 → ④ 찾은 조각만 보고 답변. 항상 같은 순서로 돕니다(LangGraph).
- **[C] 평가**: 평가 문제 50개를 챗봇에 묻고 채점 AI가 정답과 비교합니다. 챗봇 흐름과는 따로 돕니다.

단계 사이에 주고받는 형식은 `src/bidmate/schemas.py`에 있습니다. **바꾸기 전에 PM에게 먼저 말해 주세요.**

## 현재 점수 (11/13, 베이스라인)

| 항목 | 값 |
| --- | --- |
| 답변 정확도 | 74% (50문항) |
| 평균 답변 시간 | 3.8초 (목표 10초 이내) |
| 채점 AI와 사람 채점 일치 | 20 / 20 |

고도화는 파트마다 숫자 하나를 정해 한 번에 하나씩 고치고, 고치기 전후 숫자로 비교합니다.

## 팀

| 이름 | 역할 | 맡은 단계 | 협업일지 |
| --- | --- | --- | --- |
| 재웅 | PM, 개발 | ① 질문 다듬기, ④ 답변 만들기, 4단계 연결, 서버·화면, 평가 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab0801e9f63f5cb76913da0?source=copy_link) |
| 성원 | 데이터 | 목록표 읽기, 평가셋 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080b58f87ce8450658ae8?source=copy_link) |
| 인호 | 데이터 | 글자 뽑기 (hwp, pdf) | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080b082d7e42622a0a4e1?source=copy_link) |
| 석우 | 데이터 | 조각 자르기 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080c29ca0e4ce3bed6f43?source=copy_link) |
| 예원 | 검색 | 임베딩·저장, ③ 조각 찾기 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080d1b4e8c790b9698268?source=copy_link) |
| 혜린 | 검색 | ② 문서 고르기 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab08004a018fad852971891?v=4153d7dd9ab080098f2e000ca469b2aa&source=copy_link) |

## 스택

gpt-5-mini · text-embedding-3-small · Chroma · LangChain + LangGraph · FastAPI + HTML 화면 · GCP VM

## 처음 한 번 설정

```bash
# 1. uv 설치 (이미 있으면 건너뜀)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 레포 받기
git clone https://github.com/woolnd/bidmate-chatbot.git
cd bidmate-chatbot

# 3. 패키지 설치 (Python 3.12도 자동으로 설치됨)
uv sync

# 4. 비밀값 파일 만들고 OPENAI_API_KEY 채우기
cp .env.example .env
```

원본 RFP 파일과 `data_list.csv`는 구글 드라이브에서 받아 `data/raw/`에 넣습니다.
**이 폴더는 Git에 올라가지 않습니다. 절대 커밋하지 마세요.**

## 실행

```bash
uv run python scripts/build_index.py                       # 문서를 잘라 Chroma에 저장 (문서·자르는 방법이 바뀔 때만, 약 $0.15)
uv run uvicorn app.server:app --reload                     # 챗봇 서버 + 화면 (http://localhost:8000)
uv run python evaluation/run_evaluation.py --search-only   # 검색 점수만 (거의 0원)
uv run python evaluation/run_evaluation.py                 # 검색 + 답변 점수 (몇백 원)
uv run pytest                                              # 테스트 (OpenAI를 부르지 않음)
uv run ruff check . && uv run ruff format .                # 코드 스타일 검사·정리
```

평가셋(`evaluation/questions.csv`)은 정답이 들어 있어서 Git에 올리지 않고 드라이브로 나눕니다.

## 내 파일 찾기

| 단계 | 담당 | 파일 |
| --- | --- | --- |
| 목록표 읽기 | 성원 | `src/bidmate/data/metadata.py` |
| 글자 뽑기 | 인호 | `src/bidmate/data/hwp_loader.py`, `pdf_loader.py`, `loader.py` |
| 조각 자르기 | 석우 | `src/bidmate/data/chunker.py` |
| 임베딩·저장, ③ 조각 찾기 | 예원 | `src/bidmate/retrieval/vector_store.py` |
| ② 문서 고르기 | 혜린 | `src/bidmate/retrieval/project_finder.py` |
| ① 질문 다듬기, ④ 답변 | 재웅 | `src/bidmate/generation/` |
| 4단계 연결, 서버·화면, 평가 | 재웅 | `src/bidmate/pipeline/`, `app/`, `evaluation/` |

파일을 열면 맨 위에 역할과 받는 것 / 넘기는 것이 적혀 있습니다.

## 작업 방식

1. `main`에서 새 브랜치를 만듭니다: `git switch -c 이슈번호-feat-작업이름`
2. 작업 후 `uv run pytest`와 `uv run ruff check .`가 통과하는지 확인합니다.
3. Pull Request를 올리면 CI가 자동으로 검사합니다. 다른 사람 1명이 리뷰한 뒤 머지합니다.
4. 새 패키지가 필요하면 `uv add 패키지명`으로 추가합니다 (`pyproject.toml`과 `uv.lock`도 함께 커밋).
5. 고도화 PR에는 고치기 전후 숫자를 적습니다. 예) "정답 조각 Hit@5 68.6% → 80.0%"
