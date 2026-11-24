# bidmate-chatbot

공공기관 RFP(제안요청서) 100개를 검색해서, 질문에 **출처와 함께** 답하는 RAG 챗봇과 그 위에 얹은 에이전트입니다.
코드잇 스프린트 AI 중급 프로젝트 (2026.11.09 ~ 11.27)

> B2G 입찰 컨설팅 스타트업 '입찰메이트'의 엔지니어링 팀이라고 가정합니다.
> 컨설턴트가 RFP를 일일이 읽지 않고도 요구 조건, 발주 기관, 예산, 제출 방식을 빠르게 파악하도록 돕습니다.

## 흐름

<img width="3600" height="2380" alt="베이스라인_흐름도" src="https://github.com/user-attachments/assets/dc640768-a855-4e22-85cd-eae7655f7c60" />

- **[A] 미리 한 번**: 목록표와 원본 문서(hwp 96개, pdf 4개)를 읽습니다. 표는 헤더가 있으면 마크다운 표로, 헤더 없이 항목과 값이 짝지어진 표는 `항목: 값` 줄로 뽑습니다. 1000자 조각으로 자르고(200자 겹침), 표 중간에서 잘린 조각에는 표 헤더를 다시 붙입니다. 조각마다 `[기관명 / 사업명]`을 붙여 Chroma에 저장하고, 문서마다 예산·마감일을 담은 기본 정보 조각을 하나 더 만듭니다.
- **[B] 질문마다**: ① 질문 다듬기 → ② 어느 사업인지 고르기 → ③ 그 사업 문서에서 관련 조각 5개 찾기 → ④ 찾은 조각만 보고 답변 (LangGraph).
  - ③은 사업마다 임베딩 후보 20개에 BM25(키워드 검색) 후보 5개를 더하고, 리랭커가 그중 5개를 고릅니다.
- **[C] 평가**: 평가 문제 75개를 챗봇에 묻고 채점 AI가 모범 답과 비교합니다. 챗봇 흐름과는 따로 돕니다.

### 에이전트: 질문을 보고 길을 고릅니다

① 질문 다듬기 AI가 질문 종류(`route`)를 고르면, 그래프(코드)가 알맞은 길로 보냅니다. AI는 판단만 하고 길은 코드가 고릅니다.

```
rewrite ─┬─ rfp     → ② 사업 고르기 → ③ 조각 5개 → ④ 답변 ─┬─ 근거 있음 → 끝
         │                                                 └─ 근거 없음 → 조각 15개로 한 번 더 찾아 다시 답변 → 끝
         ├─ catalog → 목록표(data_list) 100줄 전체를 보고 답변 → 끝        (예산 가장 큰 사업, 대학교 사업 모두)
         ├─ similar → ② 사업 고르기 → 비슷한 사업 (조각 임베딩 평균) → 끝   (사업을 못 고르면 되묻기)
         └─ chat    → 고정 인사 → 끝                                       (안녕, 고마워, 응)
```

- 답변 아래 **다음 질문 추천** 버튼 2개, **좋아요/별로예요** 버튼(로컬 `data/logs/feedback/`에 저장)
- 비슷한 사업과 인사는 AI 답변 호출이 없어 0원·1~3초입니다

단계 사이에 주고받는 형식은 `src/bidmate/schemas.py`에 있습니다. **바꾸기 전에 PM에게 먼저 말해 주세요.**

## 최종 점수

같은 평가셋(75문항)으로 GCP VM(L4)에서 측정했습니다.

| 이름 | 무엇을 보나 | 베이스라인 (11/16 코드) | 최종 |
| --- | --- | --- | --- |
| **답변 정확도** | 챗봇 답이 모범 답과 맞나 (채점 AI) | 61.3% | **84.0%** |
| **응답 시간** | 질문 1개 평균 | 4.2초 | 9.6초 |
| **검색 정확도** (실제 흐름) | ④ 답변 단계에 정답 값이 든 조각을 건네줬나 | 68.3% | 90.7% |
| **검색 정확도** (정답 사업 지정) | ③ 조각 찾기만의 실력 | 73.2% | 93.0% |

- 최종 설정을 두 번 측정하면 답변 정확도가 84.0%, 80.0%로 3문항 안팎 흔들립니다. 검색 정확도는 측정할 때마다 같습니다.
- 검색 정확도는 후속 질문, 없는 내용 질문, 정답 값을 하나로 정할 수 없는 질문을 빼고 잽니다(최종 43문항, 베이스라인 41문항).
- 채점 AI와 사람 채점 일치: 20 / 20
- 모든 기록은 `data/evaluation/scores.csv`에 쌓입니다(Git에 안 올라감).

부분별로 고도화 전후를 같은 75문항으로 비교한 결과입니다.

| 부분 | 고도화 전 | 고도화 후 |
| --- | --- | --- |
| 사업 고르기 | 61.3% | 65.3% |
| 검색 (리랭커 + BM25) | 72.0% | 84.0% |
| 문서 데이터 처리 (표 마크다운, 헤더 다시 붙이기) | 82.7% | 84.0% (흔들림 범위 안) |
| 에이전트 (갈림길 + 다시 찾기) | 81.3% | 84.0% (흔들림 범위 안, 목록 전체 질문 5개를 새로 맞힘) |

## 팀

| 이름 | 역할 | 맡은 단계 | 협업일지 |
| --- | --- | --- | --- |
| 재웅 | PM, 개발 | ① 질문 다듬기, ④ 답변 만들기, 4단계 연결, 에이전트, 서버·화면, 평가 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab0801e9f63f5cb76913da0?source=copy_link) |
| 성원 | 데이터 | 목록표 읽기, 평가셋 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080b58f87ce8450658ae8?source=copy_link) |
| 인호 | 데이터 | 글자·표 뽑기 (hwp, pdf) | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080b082d7e42622a0a4e1?source=copy_link) |
| 석우 | 데이터 | 조각 자르기 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080c29ca0e4ce3bed6f43?source=copy_link) |
| 예원 | 검색 | 임베딩·저장, ③ 조각 찾기 (리랭커·BM25) | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080d1b4e8c790b9698268?source=copy_link) |
| 혜린 | 검색 | ② 사업 고르기 | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab08004a018fad852971891?v=4153d7dd9ab080098f2e000ca469b2aa&source=copy_link) |

## 스택

gpt-5-mini · text-embedding-3-small · Chroma · bge-reranker-v2-m3 (리랭커) · BM25 (rank-bm25 + kiwipiepy) · LangChain + LangGraph · FastAPI + HTML 화면 · GCP VM (L4)

시나리오 A(답변 모델만 로컬): Ollama qwen3.5:9b. `.env`의 `ANSWER_PROVIDER=ollama`로 바꿉니다.

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

# 5. (GPU가 있는 컴퓨터) 리랭커·BM25 패키지 설치 후 .env에 USE_RERANKER=true, USE_BM25=true
uv sync --extra rerank
```

원본 RFP 파일과 `data_list.csv`는 구글 드라이브에서 받아 `data/raw/`에 넣습니다.
**이 폴더는 Git에 올라가지 않습니다. 절대 커밋하지 마세요.**

리랭커는 GPU 없이 돌리면 질문마다 10초 이상 느려집니다. 맥에서는 꺼 두는 것을 권장합니다.

## 실행

### 로컬 (내 컴퓨터)

```bash
# 문서 저장 (처음 한 번, 문서·자르는 방법이 바뀔 때만. 약 $0.15)
uv run python scripts/build_index.py

# 챗봇 서버 + 화면 켜기 → 브라우저에서 http://localhost:8000 (API 문서: /docs)
uv run uvicorn app.server:app --reload

# 테스트와 코드 스타일 (OpenAI를 부르지 않음)
uv run pytest
uv run ruff check . && uv run ruff format .
```

맥에서는 `.env`의 `USE_RERANKER`, `USE_BM25`를 꺼 두는 것을 권장합니다(리랭커가 GPU 없이 느림).

### GCP VM (GPU, 리랭커·BM25 켬)

```bash
# 최신 코드 받기 (패키지가 바뀌었으면 uv sync도)
cd ~/bidmate-chatbot
git pull
uv sync --extra rerank

# 문서 저장 (조각 만드는 코드가 바뀌었을 때만)
uv run python scripts/build_index.py

# 서버 켜기: 외부에서 접속하도록 0.0.0.0, SSH 창을 닫아도 계속 돌게 nohup
nohup uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 > ~/server.log 2>&1 &
tail -f ~/server.log          # 로그 보기 (Ctrl+C로 보기만 멈춤)

# 서버 끄기
pkill -f "uvicorn app.server:app"
```

접속 주소는 `http://VM외부IP:8000`입니다. 화면 비밀번호는 VM `.env`의 `APP_PASSWORD`에 넣습니다(비우면 비밀번호 없이 열림).
GCP 방화벽에서 8000번 포트가 열려 있어야 합니다.

### 평가와 실험

평가셋(`evaluation/questions.csv`)은 정답이 들어 있어서 Git에 올리지 않고 드라이브로 나눕니다.

```bash
# 검색 점수만 (거의 0원)
uv run python evaluation/run_evaluation.py evaluation/questions.csv --search-only

# 검색 + 답변 점수 (몇백 원). 결과는 data/evaluation/scores.csv에 한 줄씩 쌓임
uv run python evaluation/run_evaluation.py evaluation/questions.csv --memo="바꾼 것"

# 기능 하나만 끄고 비교 (.env는 그대로 두고 명령어 앞에 붙임)
USE_BM25=false uv run python evaluation/run_evaluation.py evaluation/questions.csv --memo="BM25 끔"
USE_RERANKER=false USE_BM25=false uv run python evaluation/run_evaluation.py evaluation/questions.csv --memo="리랭커·BM25 끔"

# ② 사업 고르기 정확도
uv run python -m evaluation.score_project_finder evaluation/questions.csv

# BM25 단어 나누기 방식 비교 (리랭커 점수까지, GPU VM에서)
uv run python -m evaluation.experiment_bm25 --rerank
```

평가는 75문항에 20분쯤 걸립니다. VM에서 오래 돌릴 때는 서버처럼 앞에 `nohup`, 뒤에 `> 로그파일 2>&1 &`를 붙이면 SSH 창을 닫아도 계속 돕니다.
`data/evaluation/answers.csv`(질문별 결과)는 실행할 때마다 덮어써지니, 비교할 결과는 이름을 바꿔 복사해 둡니다.

## 내 파일 찾기

| 단계 | 담당 | 파일 |
| --- | --- | --- |
| 목록표 읽기 | 성원 | `src/bidmate/data/metadata.py` |
| 글자·표 뽑기 | 인호 | `src/bidmate/data/hwp_loader.py`, `pdf_loader.py`, `loader.py` |
| 조각 자르기 | 석우 | `src/bidmate/data/chunker.py` |
| 임베딩·저장, ③ 조각 찾기 | 예원 | `src/bidmate/retrieval/vector_store.py`, `reranker.py`, `bm25.py` |
| ② 사업 고르기 | 혜린 | `src/bidmate/retrieval/project_finder.py` |
| ① 질문 다듬기, ④ 답변 | 재웅 | `src/bidmate/generation/` |
| 4단계 연결, 에이전트 갈림길, 서버·화면, 평가 | 재웅 | `src/bidmate/pipeline/`, `app/`, `evaluation/` |
| 에이전트: 사업 전체(catalog) | 재웅 | `src/bidmate/generation/catalog.py` |
| 에이전트: 비슷한 사업(similar) | 재웅 | `src/bidmate/retrieval/similar_projects.py` |

파일을 열면 맨 위에 역할과 받는 것 / 넘기는 것이 적혀 있습니다.

## 작업 방식

1. `main`에서 새 브랜치를 만듭니다: `git switch -c 이슈번호-feat-작업이름`
2. 작업 후 `uv run pytest`와 `uv run ruff check .`가 통과하는지 확인합니다.
3. Pull Request를 올리면 CI가 자동으로 검사합니다. 다른 사람 1명이 리뷰한 뒤 머지합니다.
4. 새 패키지가 필요하면 `uv add 패키지명`으로 추가합니다 (`pyproject.toml`과 `uv.lock`도 함께 커밋).
5. 고도화 PR에는 고치기 전후 숫자를 적습니다. 예) "검색 정확도 76.7% → 90.7%"
