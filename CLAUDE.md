# CLAUDE.md

이 파일은 AI 코딩 도구(Claude Code 등)가 이 레포에서 작업할 때 먼저 읽는 안내서다.
사람도 처음 레포를 볼 때 읽으면 된다.

## 프로젝트 한 줄 요약

공공기관 RFP 문서 100개(hwp, pdf)를 검색해서, 질문에 출처와 함께 답하는 사내 RAG 챗봇.
B2G 입찰 컨설팅 스타트업 '입찰메이트'의 엔지니어링 팀이라고 가정한다.

## 자주 쓰는 명령어

```bash
uv sync                                    # 패키지 설치 (처음 한 번, 패키지가 바뀌었을 때)
uv run python scripts/build_index.py       # 문서를 읽고 잘라서 Vector DB에 저장 (미리 한 번)
uv run streamlit run app/streamlit_app.py  # 챗봇 화면 실행
uv run pytest                              # 테스트
uv run ruff check . && uv run ruff format . # 코드 스타일 검사와 자동 정리
```

## 시스템 흐름

미리 한 번: 목록표 읽기 → 문서 읽기 → 조각으로 자르기(문서마다 기본 정보 조각 1개 추가) → 조각 앞에 기관명·사업명 붙이기 → Chroma에 저장

질문이 올 때마다 아래 4단계를 항상 같은 순서로 지나간다. AI가 스스로 도구를 고르는 에이전트 방식은 쓰지 않는다.

| 단계 | 파일 | 담당 |
| --- | --- | --- |
| 1. 질문 다듬기 | `src/bidmate/generation/question_rewriter.py` | 재웅 |
| 2. 어느 문서인지 고르기 | `src/bidmate/retrieval/project_finder.py` | 검색 파트 |
| 3. 문서에서 관련 부분 찾기 | `src/bidmate/retrieval/vector_store.py` | 검색 파트 |
| 4. 답변 만들기 | `src/bidmate/generation/answer_generator.py` | 재웅 |

4단계 연결은 `src/bidmate/pipeline/graph.py`(LangGraph)에서 한다.

## 폴더 구조

```
src/bidmate/
  schemas.py          # 파트 간 데이터 약속. 바꾸기 전에 PM에게 먼저 말한다
  config.py           # 설정값 (모델 이름, 경로, k 값 등)
  data/               # 데이터 파트: 문서 읽기, 목록표, 조각 자르기
  retrieval/          # 검색 파트: 문서 고르기, 저장·검색
  generation/         # 생성: 질문 다듬기, 프롬프트, 답변
  pipeline/           # 4단계 연결
  service/            # 로그, 응답 캐시
app/                  # Streamlit 화면
scripts/              # 한 번씩 돌리는 스크립트
evaluation/           # 평가셋 양식과 채점
tests/                # 테스트
data/                 # 원본·가공 데이터, Chroma 저장 폴더 (Git에 올라가지 않음)
```

## 반드시 지킬 것

- **원본 RFP 문서와 data_list.csv는 절대 커밋하지 않는다.** 비밀유지 대상이다. `data/` 폴더는 `.gitignore`에 들어 있다.
- **API 키는 `.env`에만 둔다.** 코드, 로그, 커밋 메시지, AI 대화에 키를 쓰지 않는다.
- **`schemas.py`의 형식은 파트 간 약속이다.** 필드를 바꾸거나 지우지 말고, 필요하면 PM과 상의한다.
- **테스트와 CI에서는 OpenAI API를 부르지 않는다.** 팀 사용량 한도가 $20이다. LLM과 임베딩은 가짜 객체로 바꿔서 테스트한다.

## 코드 작성 규칙

- 변수명·함수명은 줄이지 않는다. `doc` 말고 `document`, `q` 말고 `question`. 단, `i`, `k`(검색 개수)처럼 관례로 쓰는 이름은 그대로 쓴다.
- 주석과 docstring은 한국어로 쓴다. 함수마다 무엇을 받아서 무엇을 돌려주는지 적는다.
- 타입 힌트를 붙인다.
- 아직 만들지 않은 함수는 `TODO`를 남기고 가짜 값을 돌려준다. 그래야 전체 흐름이 항상 실행된다.
- 설정값(모델 이름, 경로, 숫자)은 코드에 직접 쓰지 말고 `config.py`에서 가져온다.

## 프롬프트 규칙 (비용과 속도)

- 고정 부분(시스템 프롬프트 → 답변 예시 → 출력 형식)을 앞에, 바뀌는 부분(대화 기록 → 찾은 조각 → 질문)을 뒤에 둔다. 앞부분이 같아야 OpenAI 프롬프트 캐싱이 걸린다.
- 고정 부분에 날짜, 사용자 이름 같은 바뀌는 값을 넣지 않는다.
- 답변은 `schemas.py`의 `AnswerResult` 형식으로 받는다(구조화된 출력).
- 시스템 프롬프트는 짧게 시작하고, 평가에서 틀린 경우가 나올 때만 규칙을 더한다.

## AI 도구에게 부탁하는 방식

팀원 대부분이 RAG를 처음 만든다. AI 도구는 이렇게 도와준다.

- 작업을 시작할 때 팀원이 자기 담당(예: "나는 검색 파트, 2단계 담당")을 먼저 AI에게 말한다.
- 코드를 바꾸기 전에 무엇을 왜 바꾸는지 먼저 설명한다.
- 자기 담당 파일 밖을 고쳐야 하면, 고치지 말고 먼저 알려 준다.
- 새 패키지가 필요하면 `uv add 패키지명`으로 추가하고, 왜 필요한지 설명한다.
- 코드를 쓴 뒤 `uv run pytest`와 `uv run ruff check .`를 돌려 확인한다.
