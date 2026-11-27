# bidmate-chatbot

공공기관 RFP 100개(HWP 96, PDF 4)를 검색해 **출처와 함께** 답하는 RAG 챗봇 + 질문 종류별로 길을 고르는 에이전트.
코드잇 AI 중급 프로젝트 3팀 (2026.11.09 ~ 11.27)

## 구조

<img width="3600" height="2380" alt="흐름도" src="https://github.com/user-attachments/assets/dc640768-a855-4e22-85cd-eae7655f7c60" />

**문서 저장 (1회)**
`data_list.csv` + 원본 → 글자·표 추출 → 1,000자 조각(200자 겹침, 잘린 표에 헤더 재부착) → `[기관명 / 사업명]` 머리말 → 임베딩 → Chroma

**질문 처리 (LangGraph)**
```
rewrite ─┬─ rfp     → find_projects → search → answer ─┬─ 근거 있음 → END
         │                                              └─ 근거 없음 → retry(k=15) → END
         ├─ catalog → 목록표 100줄 → answer → END
         ├─ similar → find_projects → 사업 임베딩 평균 비교 (0개·여러 개면 되묻기) → END
         └─ chat    → 고정 인사 → END
```
- search: 사업별 임베딩 20 + BM25 5 → 리랭커(bge-reranker-v2-m3) → 5개
- AI 호출: 질문당 최대 3회 (rewrite 1 + answer 1 + retry 1)
- 답변과 함께 다음 질문 추천 2개, 좋아요/별로예요 피드백(`data/logs/feedback/`, 로컬 저장)
- 단계 간 데이터 형식: `src/bidmate/schemas.py`

## 결과

평가셋 75문항, GCP VM(L4)

| 지표 | 베이스라인 | 최종 |
| --- | --- | --- |
| 답변 정확도 (채점 AI) | 61.3% | **84.0%** |
| 검색 정확도 (실제 흐름) | 68.3% | 90.7% |
| 검색 정확도 (정답 사업 지정) | 73.2% | 93.0% |
| 평균 응답 시간 | 4.2초 | 9.6초 |

- 채점 AI ↔ 사람 채점 일치 20/20
- 답변 정확도는 측정마다 ±3문항 흔들림 (84.0% / 80.0%). 검색 정확도는 고정

| 개선 | 결과 |
| --- | --- |
| 리랭커 + BM25 | 답변 정확도 72.0% → 84.0% |
| 사업 고르기 (흔한 단어 제외, "A와 B" 분리) | 잘못 고른 질문 8 → 1 |
| 표 처리 (마크다운·`항목: 값`, 헤더 재부착) | 정답 값이 잘린 질문 2 → 0 |
| 에이전트 (catalog 등) | 목록 전체 질문 5개 새로 정답 |

## 한계

- 조건이 복잡한 목록 질문("10억 이상" 등)에서 일부 누락 → 조건 필터를 코드로 옮길 예정
- 표 전체를 세야 하는 개수 질문 미해결
- 평가셋이 원문 확인형 질문 위주

## 기술 스택

| 구분 | 사용 |
| --- | --- |
| LLM | gpt-5-mini (질문 다듬기·답변·채점), Ollama qwen3.5:9b (시나리오 A 답변) |
| 임베딩 | OpenAI text-embedding-3-small |
| 벡터 DB | Chroma |
| 검색 | BM25 (rank-bm25 + kiwipiepy), 리랭커 bge-reranker-v2-m3 |
| 파이프라인 | LangChain, LangGraph |
| 문서 추출 | syhwp (HWP), PyMuPDF (PDF) |
| 서버 · 화면 | FastAPI, HTML |
| 인프라 | GCP VM (NVIDIA L4) |
| 개발 도구 | uv, pytest, ruff, GitHub Actions |

시나리오 A는 `.env`의 `ANSWER_PROVIDER=ollama`로 전환합니다.

## 실행

```bash
uv sync                      # GPU 환경은 uv sync --extra rerank
cp .env.example .env         # OPENAI_API_KEY 입력
# 원본 RFP와 data_list.csv를 data/raw/에 배치 (Git 제외, 커밋 금지)

uv run python scripts/build_index.py        # 인덱싱 (약 $0.15)
uv run uvicorn app.server:app --reload      # http://localhost:8000
```

- 맥(GPU 없음)은 `.env`에서 `USE_RERANKER=false`, `USE_BM25=false` 권장
- VM: `nohup uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 > ~/server.log 2>&1 &` (8000 포트 개방, 비밀번호는 `APP_PASSWORD`)

## 평가

```bash
uv run python evaluation/run_evaluation.py evaluation/questions.csv --memo="변경 내용"
uv run python evaluation/run_evaluation.py evaluation/questions.csv --search-only   # 검색만
USE_BM25=false uv run python evaluation/run_evaluation.py evaluation/questions.csv  # 기능 끄고 비교
uv run python -m evaluation.score_project_finder evaluation/questions.csv         # 사업 고르기
```

- 평가셋은 정답 포함이라 Git 제외 (드라이브 공유)
- 결과: `data/evaluation/scores.csv` (누적), `answers.csv` (매번 덮어씀)

## 개발

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

- 테스트·CI는 OpenAI 호출 없음 (가짜 모델)
- 브랜치 `이슈번호-feat-작업명` → PR → CI 통과 + 리뷰 1명 → 머지
- 고도화 PR에는 전후 수치 기재

## 팀

| 이름 | 역할 | 담당 | 주요 파일 | 협업일지 |
| --- | --- | --- | --- | --- |
| 엄재웅 | PM · 개발 | ①④ 단계, LangGraph, 에이전트, 서버, 평가 | `generation/`, `pipeline/`, `app/` | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab0801e9f63f5cb76913da0?source=copy_link) |
| 윤성원 | 데이터 | 목록표, 평가셋 | `data/metadata.py` | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080b58f87ce8450658ae8?source=copy_link) |
| 강인호 | 데이터 | 글자·표 추출 | `data/hwp_loader.py`, `pdf_loader.py` | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080b082d7e42622a0a4e1?source=copy_link) |
| 이석우 | 데이터 | 조각 자르기 | `data/chunker.py` | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080c29ca0e4ce3bed6f43?source=copy_link) |
| 김예원 | 검색 | 임베딩·저장, ③ 조각 찾기 | `retrieval/vector_store.py`, `reranker.py`, `bm25.py` | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab080d1b4e8c790b9698268?source=copy_link) |
| 김혜린 | 검색 | ② 사업 고르기 | `retrieval/project_finder.py` | [노션](https://attractive-reason-70a.notion.site/40d3d7dd9ab08004a018fad852971891?v=4153d7dd9ab080098f2e000ca469b2aa&source=copy_link) |
