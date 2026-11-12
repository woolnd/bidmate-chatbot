# bidmate-chatbot

공공기관 RFP 문서를 검색해서 질문에 출처와 함께 답하는 RAG 챗봇입니다.
코드잇 스프린트 AI 중급 프로젝트 (2026.11.09 ~ 11.27)

> 역할, 일정, 협업 규칙은 노션 기획서를 봅니다. 이 README는 코드 실행 방법만 다룹니다.

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

원본 RFP 파일과 `data_list.csv`는 구글 드라이브에서 받아 `data/raw/` 에 넣습니다.
**이 폴더는 Git에 올라가지 않습니다. 절대 커밋하지 마세요.**

## 실행

```bash
uv run python scripts/build_index.py       # 문서를 잘라 Vector DB에 저장 (문서가 바뀔 때만, 구현 후)
uv run uvicorn app.server:app --reload    # 챗봇 서버 + 화면 (http://localhost:8000)
uv run pytest                              # 테스트
uv run ruff check . && uv run ruff format . # 코드 스타일 검사·정리
```

지금은 파일 구조와 역할 설명만 있습니다. 각자 자기 파일 맨 위 설명(받는 것 / 넘기는 것)을 보고 채웁니다.

## 내 파일 찾기

| 담당 | 파일 |
| --- | --- |
| 데이터 파트 | `src/bidmate/data/`, `evaluation/` (평가셋) |
| 검색 파트 | `src/bidmate/retrieval/` |
| 재웅 | `src/bidmate/generation/`, `src/bidmate/pipeline/`, `src/bidmate/service/`, `app/` |

파트 안에서 누가 어떤 파일을 맡을지는 파트끼리 정합니다. 파일을 열면 맨 위에 역할과 받는 것 / 넘기는 것이 적혀 있습니다.

## 흐름

```
미리 한 번:  문서 읽기 → 조각 자르기 → 머리말 붙이기 → Chroma 저장

질문마다:   1. 질문 다듬기 → 2. 어느 문서인지 고르기 → 3. 문서에서 관련 부분 찾기 → 4. 답변 만들기
```

단계 사이에 주고받는 형식은 `src/bidmate/schemas.py` 에 있습니다. **바꾸기 전에 PM에게 먼저 말해 주세요.**

## 작업 방식

1. `main`에서 새 브랜치를 만듭니다: `git switch -c feature/hwp-loader`
2. 작업 후 `uv run pytest` 와 `uv run ruff check .` 가 통과하는지 확인합니다.
3. Pull Request를 올리면 CI가 자동으로 검사합니다. 다른 사람 1명이 리뷰한 뒤 머지합니다.
4. 새 패키지가 필요하면 `uv add 패키지명` 으로 추가합니다 (`pyproject.toml` 과 `uv.lock` 도 함께 커밋).

## Search-B 규칙과 NER 비교

`scripts/compare_search_b_models.py`에서 같은 질문으로 변경 전후의 규칙과 NER 연결 방식을 비교한다.
NER는 기관명과 사업명의 검출 여부를 확인한 뒤 원래 질문을 규칙 선택기에 넘긴다.
모델 단독 문서 검색과는 구분한다.

실험용 Python 환경에 PyTorch, Transformers, Safetensors가 필요하다. 원본 목록, 질문 정답,
학습 문장과 모델 가중치는 로컬에 둔다. `reports/` 안의 결과와 모델 파일은 Git에 올리지 않는다.

설정 파일 예시:

```json
{
  "experiment_id": "search_b_compare_001",
  "csv": "data/raw/data_list.csv",
  "cases": "reports/search_b/cases.proposed.json",
  "baseline_rules": "reports/search_b/project_finder.baseline.py",
  "checkpoint": "reports/search_b/checkpoint",
  "weak_dataset": "reports/search_b/dataset.weak.json",
  "output": "reports/search_b/evaluation_001",
  "threshold": 0.3,
  "repeats": 3,
  "seed": 42,
  "cpu_threads": 4,
  "device": "mps"
}
```

경로와 장치는 로컬 환경에 맞춰 설정한 뒤 레포 루트에서 실행한다.

```bash
PYTHONPATH=src python -m scripts.compare_search_b_models --config reports/search_b/config.json
```

질문 파일의 필수 필드는 `case_id`, `question`, `expected_ids`, `category`, `split`이다.
`split`은 `development`만 허용한다. 같은 질문, 알 수 없는 문서 ID, 학습 문장과의 중복은 거부한다.
중복 검사는 Unicode, 대소문자, 공백을 정규화한 뒤 비교한다.

출력에는 설정, 입력 및 코드 해시, 실행 코드 원본, 질문별 예상/반환 ID, NER 구간과 전체 지표가 남는다.
기존 결과 폴더를 덮어쓰지 않으며, 실행 중 입력이나 코드가 바뀌면 결과 확정을 중단한다.
개발 질문의 완전 일치율은 최종 답변 정확도나 독립 최종 평가 점수와 구분해서 적는다.

학습 문장 확장과 NER 재학습은 아래 스크립트로 따로 실행한다.

```bash
PYTHONPATH=src python -m scripts.build_search_b_ner_training \
  --input reports/search_b/dataset.weak.json \
  --exclude-cases reports/search_b/cases.proposed.json \
  --output reports/search_b/dataset.augmented.json
PYTHONPATH=src python -m scripts.train_search_b_ner --config reports/search_b/train_config.json
```

문장 확장은 학습 기관에만 적용하고 검증 문장은 보존한다. 학습 설정에는 데이터 SHA-256,
기반 모델 revision, 장치, seed, 학습 파라미터와 출력 경로를 지정한다.
`class_weight_power` 기본값은 `0.5`, `lr_schedule` 기본값은 `constant`다.
학습 실행마다 실제 사용한 코드와 정규화 코드, 설정, 버전, epoch별 검증 수치를 로컬에 저장한다.
새 가중치는 개발 점수가 올랐다는 이유만으로 교체하지 않고 별도 검증 후 선택한다.
