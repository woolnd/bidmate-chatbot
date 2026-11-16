# Search-B 실험 다시 실행하기

[11월 16일 결과](../../docs/search-b-experiments-2026-11-16.md)에서 실험별 입력 범위, 전후 수치와 적용 여부를 볼 수 있습니다. 명령은 저장소 루트에서 실행합니다.

## 입력 준비

- `CATALOG`에 로컬 목록표 CSV, `QUESTIONS`에 권한이 있는 팀 평가 CSV를 지정합니다. 근거 채점에는 `answer_keywords` 열이 필요합니다.
- 원본 대조에는 `ORIGINAL_CATALOG`(이전 목록표), `ARCHIVE`(원본 ZIP), `RAW_DIR`(HWP/PDF 폴더)을 지정합니다. 원본의 파일명과 목록표의 문서 ID를 연결할 수 있어야 합니다.
- 아래 결과 폴더는 저장소 밖에 만듭니다. JSON과 Chroma 인덱스에는 질문, 문서 ID, 본문이 들어갑니다.
- 제목 변형의 기본 시드는 1611, 새 문서 비교 질문의 시드는 161105입니다. 평가셋이나 모델을 바꾸면 새 결과 폴더를 씁니다.

```sh
OUT="$(mktemp -d)"
uv run --locked python -m evaluation.search_b_experiments.compare_titles \
  "$CATALOG" "$QUESTIONS" "$OUT/titles.json"
uv run --locked python -m evaluation.search_b_experiments.build_corpus \
  "$CATALOG" "$RAW_DIR" "$OUT/corpus.json"
uv run --locked python -m evaluation.search_b_experiments.compare_chunks \
  "$CATALOG" "$QUESTIONS" "$OUT/corpus.json" "$OUT/chunks.json"
```

## 제목, 오타, 목록표

추가 패키지는 아래 명령을 실행할 때만 설치합니다.

```sh
EXPERIMENT_DEPS=evaluation/search_b_experiments/requirements.txt
uv run --locked --with-requirements "$EXPERIMENT_DEPS" \
  python -m evaluation.search_b_experiments.text_trials \
  "$CATALOG" "$QUESTIONS" "$OUT/titles.json" "$OUT/text-trials.json"
uv run --locked --with-requirements "$EXPERIMENT_DEPS" \
  python -m evaluation.search_b_experiments.record_link_trial \
  "$CATALOG" "$ORIGINAL_CATALOG" "$OUT/record-linkage.json"
uv run --locked python -m evaluation.search_b_experiments.review_metadata \
  "$CATALOG" "$ORIGINAL_CATALOG" "$ARCHIVE" "$OUT/corpus.json" "$OUT/metadata.json"
uv run --locked --with-requirements "$EXPERIMENT_DEPS" \
  python -m evaluation.search_b_experiments.verify_followups \
  "$CATALOG" "$QUESTIONS" "$ARCHIVE" "$OUT/corpus.json" "$OUT/chunks.json" \
  "$OUT/titles.json" "$OUT/followups.json"
```

`text_trials`에서 RapidFuzz 세 가지, BK-tree, MeCab을 비교합니다. `verify_followups`에서 한 글자 치환의 오선택·누락, 질문 항목으로 순서를 바꾼 BM25, 원본 해시와 목록표 빈칸을 다시 확인합니다.

## 근거 조각

```sh
uv run --locked python -m evaluation.search_b_experiments.compare_retrieval_libraries \
  "$OUT/corpus.json" "$OUT/chunks.json" "$RAW_DIR" "$OUT/fts-and-hwp.json"
uv run --locked python -m evaluation.search_b_experiments.score_evidence_units \
  "$OUT/corpus.json" "$OUT/chunks.json" "$CATALOG" "$RAW_DIR" "$OUT/evidence-units.json"
uv run --locked python -m evaluation.search_b_experiments.score_result_diversity \
  "$OUT/corpus.json" "$OUT/chunks.json" "$CATALOG" "$OUT/diversity.json"
```

로컬 벡터 검색과 모델 실험은 로컬 Ollama에서 `nomic-embed-text:latest`와 `qwen3:1.7b`를 사용할 수 있어야 합니다. 임베딩과 슬롯 추출 요청은 `127.0.0.1:11434`로 보냅니다. 처음 인덱스를 만들 때는 대상 원문의 조각을 임베딩하는 시간이 듭니다.

```sh
uv run --locked python -m evaluation.search_b_experiments.build_local_chroma \
  "$OUT/corpus.json" "$OUT/chunks.json" "$CATALOG" "$OUT/chroma"
uv run --locked python -m evaluation.search_b_experiments.score_chroma_replay \
  "$OUT/chroma" "$OUT/corpus.json" "$OUT/chunks.json" "$CATALOG" "$QUESTIONS" \
  "$OUT/chroma-results.json"
uv run --locked python -m evaluation.search_b_experiments.score_rank_fusion \
  "$OUT/chroma" "$OUT/corpus.json" "$OUT/chunks.json" "$CATALOG" "$OUT/rank-fusion.json"
uv run --locked python -m evaluation.search_b_experiments.dense_link_trial \
  "$OUT/dense-link.json"
uv run --locked python -m evaluation.search_b_experiments.slot_trial \
  "$CATALOG" "$QUESTIONS" "$OUT/slots.json"
```

개체명 인식 학습과 평가는 내부 목록표에서 만든 학습 문장과 로컬 체크포인트가 필요합니다. 공개 결과 문서에서 모델별 설정과 점수를 확인할 수 있습니다.

## PR에 남기는 내용

- 어떤 질문이 실패했고 어느 단계에서 달라졌는지 먼저 적습니다.
- 기준 커밋, 실행 명령, 같은 평가셋으로 잰 전후 수치와 분모를 함께 남깁니다. 성능을 이유로 보류할 때는 함께 잰 시간이나 새 오선택 수를 적습니다.
- 같은 작업의 후속 실험은 기존 PR에 보탭니다. 원문 대신 집계 결과와 재실행 코드를 커밋합니다.
