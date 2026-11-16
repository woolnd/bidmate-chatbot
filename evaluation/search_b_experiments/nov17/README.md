# 11/17 Search-B 실험 다시 실행하기

[반영한 문서 선택 코드와 회귀 테스트](https://github.com/woolnd/bidmate-chatbot/commit/cdf2f5e20290e3adef1884591ce691360d6933a3)는 복수 요청에서 빠진 문서를 다시 찾습니다. 이 폴더는 팀이 결과를 재검증하는 실험 코드입니다. 점수와 적용 판단은 [PR #41](https://github.com/woolnd/bidmate-chatbot/pull/41)에 있습니다.

팀 목록표와 평가 CSV를 각각 `CATALOG`, `QUESTIONS`에 지정합니다. `CASES`와 `PARAPHRASES`는 비공개 문항 JSON입니다. 각 행에는 `question`, `expected_ids`(목록표 파일명 배열), `agency_groups`, `category`가 들어갑니다. 같은 수치를 다시 재려면 같은 질문 파일과 기준 커밋을 씁니다. 폴더 밖에 결과를 저장합니다. 특히 생성 문항, 조각 인덱스 및 건별 오류 결과에는 비공개 내용이 들어갑니다.

```sh
OUT="$(mktemp -d)"
TRIAL=evaluation.search_b_experiments.nov17

uv run --locked python -m "$TRIAL.verify_selector" \
  --catalog "$CATALOG" --team "$QUESTIONS" --cases "$CASES" "$PARAPHRASES" \
  --output "$OUT/selector.json"
uv run --locked python -m "$TRIAL.audit_selection" \
  --catalog "$CATALOG" --questions "$QUESTIONS" --cases "$CASES" "$PARAPHRASES" \
  --output "$OUT/selection-audit.json"

uv run --locked --with scikit-learn python -m "$TRIAL.summary_recovery_trial" \
  --catalog "$CATALOG" --team "$QUESTIONS" --cases "$CASES" "$PARAPHRASES" \
  --output "$OUT/summary.json"
uv run --locked --with scikit-learn python -m "$TRIAL.audit_generated_labels" \
  --catalog "$CATALOG" --questions "$PARAPHRASES" --output "$OUT/labels.json"
```

생성 문항 점검 결과를 `audit_selection --label-audit "$OUT/labels.json"`에 전달하면 같은 설명을 다시 쓴 오답을 확인할 수 있습니다. 로컬 Ollama에 `qwen3:1.7b`가 있으면 `uv run --locked python -m "$TRIAL.generate_paraphrases" --catalog "$CATALOG" --output "$OUT/paraphrases.json"`으로 새 문항을 만들 수 있습니다. 생성된 정답은 원문 확인 후 쓸 수 있습니다. 점검 명령의 `--batch-size`는 생성 당시 묶음 크기에 맞춥니다. 11/17 점검 자료는 8건씩 만들었고, 새 생성은 기본 1건씩 받습니다.

## 임베딩과 분류기

로컬에 설치된 multilingual-e5-small 스냅샷을 `E5_SNAPSHOT`에 지정합니다. 모델 파일은 실행 중 내려받지 않습니다. 분류기는 기관 단위로 나눈 학습 문항을 쓰며 팀 공통 평가 문항을 학습에서 제외합니다.

```sh
uv run --locked --with torch --with transformers --with scikit-learn \
  python -m "$TRIAL.learned_link_trial" \
  --catalog "$CATALOG" --team "$QUESTIONS" --cases "$CASES" "$PARAPHRASES" \
  --model "$E5_SNAPSHOT" --output "$OUT/learned.json"
uv run --locked --with torch --with transformers --with scikit-learn \
  python -m "$TRIAL.scoped_recovery_trial" \
  --catalog "$CATALOG" --team "$QUESTIONS" --cases "$CASES" "$PARAPHRASES" \
  --model "$E5_SNAPSHOT" --prior "$OUT/learned.json" --output "$OUT/scoped.json"
```

## 팀 파이프라인과 문서 읽기

`CORPUS`와 `INDEX`에 상위 [실험 설명서](../README.md)대로 만든 코퍼스 JSON과 Chroma 폴더를 지정합니다. 같은 로컬 Nomic Chroma를 재사용하며 목록표·코퍼스 해시를 맞춰 봅니다. `ARCHIVE`는 원본 문서 ZIP입니다. 열린 팀 PR의 기준 코드가 로컬에 없을 때는 GitHub에서 참조를 먼저 받습니다.

```sh
git fetch origin refs/pull/39/head refs/pull/29/head
uv run --locked python -m "$TRIAL.audit_pipeline" \
  --catalog "$CATALOG" --questions "$QUESTIONS" --corpus "$CORPUS" --index "$INDEX" \
  --output "$OUT/pipeline-audit.json"
uv run --locked python -m "$TRIAL.audit_loader" \
  --archive "$ARCHIVE" --catalog "$CATALOG" --questions "$QUESTIONS" \
  --output "$OUT/loader-audit.json"
```

팀 평가의 문서 목록 완전 일치와 내용 기준 Hit@5는 별도로 읽습니다. 이 로컬 인덱스의 점수는 팀 공용 인덱스 점수와 직접 비교하지 않습니다.
