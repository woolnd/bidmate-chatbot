
"""BidMate RAG 챗봇 평가 프로그램.

담당:
- 재웅: 평가 실행 및 채점 코드
- 데이터 파트: 평가셋 작성 및 점수 정리

실행:
    uv run python evaluation/run_evaluation.py evaluation/questions.csv

주의:
- questions.csv에는 원본 RFP 정보가 포함되어 있으므로 Git에 올리지 않는다.
- results.csv에도 평가 질문과 답변이 포함되므로 Git에 올리지 않는다.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bidmate.pipeline.graph import run_chatbot
from bidmate.schemas import ChatHistory


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = [
    "question_id",
    "question_type",
    "question",
    "previous_question",
    "expected_answer",
    "expected_project_names",
]

VALID_QUESTION_TYPES = {
    "한 문서",
    "기본 정보",
    "여러 문서",
    "후속 질문",
    "없는 내용",
}

RESULT_COLUMNS = [
    "question_id",
    "question_type",
    "question",
    "previous_question",
    "expected_answer",
    "expected_project_names",
    "rewritten_question",
    "retrieved_project_names",
    "retrieved_chunk_count",
    "retrieval_hit",
    "generated_answer",
    "answer_source_project_names",
    "has_evidence",
    "answer_correct",
    "faithfulness",
    "error",
]


# --------------------------------------------------
# 2. 평가 결과 구조
# --------------------------------------------------

@dataclass
class EvaluationResult:
    question_id: str
    question_type: str
    question: str
    previous_question: str
    expected_answer: str
    expected_project_names: str

    rewritten_question: str = ""
    retrieved_project_names: str = ""
    retrieved_chunk_count: int = 0
    retrieval_hit: int | None = None

    generated_answer: str = ""
    answer_source_project_names: str = ""
    has_evidence: str = ""

    # 1주차에는 사람이 직접 채점한다.
    # 1 = 정답, 0 = 오답
    answer_correct: int | None = None

    # 2주차부터 AI 채점 기능을 연결한다.
    faithfulness: int | None = None

    error: str = ""


# --------------------------------------------------
# 3. 평가셋 읽기
# --------------------------------------------------

def load_questions(csv_path: Path) -> list[dict[str, str]]:
    """CSV 평가셋을 읽는다."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"평가셋 파일을 찾을 수 없습니다: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(
                "CSV 컬럼이 올바르지 않습니다.\n"
                f"필요한 컬럼: {REQUIRED_COLUMNS}\n"
                f"현재 컬럼: {reader.fieldnames}"
            )

        questions = list(reader)

    if not questions:
        raise ValueError("평가셋에 질문이 없습니다.")

    validate_questions(questions)

    return questions


# --------------------------------------------------
# 4. 평가셋 검증
# --------------------------------------------------

def validate_questions(questions: list[dict[str, str]]) -> None:
    """평가 문항의 필수값과 유형을 검증한다."""

    question_ids = set()

    for index, row in enumerate(questions, start=1):
        question_id = row["question_id"].strip()
        question_type = row["question_type"].strip()
        question = row["question"].strip()

        if not question_id:
            raise ValueError(
                f"{index}번째 문항의 question_id가 비어 있습니다."
            )

        if question_id in question_ids:
            raise ValueError(f"중복된 문항 번호: {question_id}")

        question_ids.add(question_id)

        if question_type not in VALID_QUESTION_TYPES:
            raise ValueError(
                f"{question_id}: 올바르지 않은 질문 유형입니다: "
                f"{question_type}"
            )

        if not question:
            raise ValueError(f"{question_id}: 질문이 비어 있습니다.")

        if not row["expected_answer"].strip():
            raise ValueError(
                f"{question_id}: expected_answer가 비어 있습니다."
            )

        if question_type == "후속 질문":
            if not row["previous_question"].strip():
                raise ValueError(
                    f"{question_id}: 이전 질문이 필요합니다."
                )
        elif row["previous_question"].strip():
            raise ValueError(
                f"{question_id}: 후속 질문이 아닌데 "
                "previous_question이 입력되어 있습니다."
            )

        if question_type == "없는 내용":
            if row["expected_project_names"].strip():
                raise ValueError(
                    f"{question_id}: 없는 내용 질문의 "
                    "expected_project_names는 비워야 합니다."
                )
        elif not row["expected_project_names"].strip():
            raise ValueError(
                f"{question_id}: 정답 사업명이 비어 있습니다."
            )

    print(f"[검증 완료] 총 {len(questions)}개 문항")


# --------------------------------------------------
# 5. 검색 결과에서 사업명 추출
# --------------------------------------------------

def get_retrieved_project_names(
    chunks: list[tuple[Any, float]],
) -> list[str]:
    """검색된 Document의 metadata에서 사업명을 가져온다."""

    project_names = []

    for document, score in chunks:
        metadata = document.metadata

        project_name = metadata.get("project_name")

        if project_name:
            project_names.append(str(project_name))

    # 중복 제거. 검색 순서는 유지한다.
    return list(dict.fromkeys(project_names))


# --------------------------------------------------
# 6. 검색 적중 여부 계산
# --------------------------------------------------

def calculate_retrieval_hit(
    expected_project_names: str,
    retrieved_project_names: list[str],
) -> int | None:
    """검색 결과에 정답 사업명이 하나라도 포함되면 1."""

    # 정답 문서가 없는 질문은 검색 적중률 계산에서 제외한다.
    if not expected_project_names.strip():
        return None

    expected_names = {
        name.strip()
        for name in expected_project_names.split("|")
        if name.strip()
    }

    retrieved_names = set(retrieved_project_names)

    return int(bool(expected_names & retrieved_names))


# --------------------------------------------------
# 7. 챗봇 실행 및 평가
# --------------------------------------------------

def evaluate_question(
    row: dict[str, str],
) -> EvaluationResult:
    """문항 하나를 실제 챗봇에 전달하고 검색 및 답변 결과를 수집한다."""

    result = EvaluationResult(
        question_id=row["question_id"].strip(),
        question_type=row["question_type"].strip(),
        question=row["question"].strip(),
        previous_question=row["previous_question"].strip(),
        expected_answer=row["expected_answer"].strip(),
        expected_project_names=row["expected_project_names"].strip(),
    )

    try:
        chat_history: ChatHistory = []

        # 후속 질문이면 이전 질문을 먼저 챗봇에 전달한다.
        # 이전 질문에 대한 실제 챗봇 답변을 대화 기록에 넣는다.
        if result.question_type == "후속 질문":
            previous_state = run_chatbot(
                result.previous_question,
                [],
            )

            previous_answer = previous_state[
                "answer_result"
            ].answer

            chat_history = [
                (result.previous_question, previous_answer)
            ]

        # 현재 질문 실행
        state = run_chatbot(
            result.question,
            chat_history,
        )

        # 1단계 결과
        result.rewritten_question = state.get(
            "rewritten_question",
            "",
        )

        # 3단계 검색 결과
        chunks = state.get("chunks", [])

        retrieved_names = get_retrieved_project_names(chunks)

        result.retrieved_project_names = " | ".join(
            retrieved_names
        )

        result.retrieved_chunk_count = len(chunks)

        result.retrieval_hit = calculate_retrieval_hit(
            result.expected_project_names,
            retrieved_names,
        )

        # 4단계 답변 결과
        answer_result = state["answer_result"]

        result.generated_answer = answer_result.answer

        result.answer_source_project_names = " | ".join(
            answer_result.source_project_names
        )

        result.has_evidence = str(
            answer_result.has_evidence
        )

    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"

    return result


# --------------------------------------------------
# 8. 전체 평가 실행
# --------------------------------------------------

def run_evaluation(
    questions: list[dict[str, str]],
) -> list[EvaluationResult]:
    """모든 평가 문항을 순서대로 실행한다."""

    results = []

    total = len(questions)

    print("\n챗봇 평가를 시작합니다.")
    print(f"전체 문항: {total}\n")

    for index, row in enumerate(questions, start=1):
        question_id = row["question_id"]

        print(
            f"[{index}/{total}] {question_id} "
            f"- {row['question']}"
        )

        result = evaluate_question(row)
        results.append(result)

        if result.error:
            print(f"  [오류] {result.error}")
        else:
            print(
                f"  검색 조각: {result.retrieved_chunk_count}개"
            )
            print(
                f"  검색 적중: {result.retrieval_hit}"
            )
            print(
                f"  답변: {result.generated_answer}"
            )

    return results


# --------------------------------------------------
# 9. 평가 결과 CSV 저장
# --------------------------------------------------

def save_results(
    results: list[EvaluationResult],
    output_path: Path,
) -> None:
    """문항별 평가 결과를 CSV로 저장한다."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                "question_id": result.question_id,
                "question_type": result.question_type,
                "question": result.question,
                "previous_question": result.previous_question,
                "expected_answer": result.expected_answer,
                "expected_project_names": result.expected_project_names,
                "rewritten_question": result.rewritten_question,
                "retrieved_project_names": result.retrieved_project_names,
                "retrieved_chunk_count": result.retrieved_chunk_count,
                "retrieval_hit": (
                    ""
                    if result.retrieval_hit is None
                    else result.retrieval_hit
                ),
                "generated_answer": result.generated_answer,
                "answer_source_project_names": result.answer_source_project_names,
                "has_evidence": result.has_evidence,
                "answer_correct": (
                    ""
                    if result.answer_correct is None
                    else result.answer_correct
                ),
                "faithfulness": (
                    ""
                    if result.faithfulness is None
                    else result.faithfulness
                ),
                "error": result.error,
            })

    print(f"\n[저장 완료] {output_path}")


# --------------------------------------------------
# 10. 전체 평가 점수 요약
# --------------------------------------------------

def print_summary(
    results: list[EvaluationResult],
) -> None:
    """검색 적중률과 사람 채점 결과를 요약한다."""

    total = len(results)

    retrieval_values = [
        result.retrieval_hit
        for result in results
        if result.retrieval_hit is not None
    ]

    answer_values = [
        result.answer_correct
        for result in results
        if result.answer_correct is not None
    ]

    faithfulness_values = [
        result.faithfulness
        for result in results
        if result.faithfulness is not None
    ]

    error_count = sum(
        bool(result.error)
        for result in results
    )

    print("\n" + "=" * 55)
    print("BidMate 평가 결과")
    print("=" * 55)

    print(f"전체 문항: {total}")
    print(f"실행 오류: {error_count}")

    if retrieval_values:
        hit_rate = (
            sum(retrieval_values)
            / len(retrieval_values)
            * 100
        )

        print(
            f"검색 적중률: {hit_rate:.2f}% "
            f"({sum(retrieval_values)}/{len(retrieval_values)})"
        )
    else:
        print("검색 적중률: 평가 가능한 문항 없음")

    if answer_values:
        accuracy = (
            sum(answer_values)
            / len(answer_values)
            * 100
        )
        print(f"답변 정확성: {accuracy:.2f}%")
    else:
        print("답변 정확성: 사람 채점 전")

    if faithfulness_values:
        faithfulness = (
            sum(faithfulness_values)
            / len(faithfulness_values)
            * 100
        )
        print(f"근거 충실도: {faithfulness:.2f}%")
    else:
        print("근거 충실도: 채점 기능 연결 전")

    print("=" * 55)


# --------------------------------------------------
# 11. 프로그램 시작점
# --------------------------------------------------

def main() -> None:
    """평가 프로그램을 실행한다."""

    if len(sys.argv) != 2:
        print(
            "사용법:\n"
            "uv run python evaluation/run_evaluation.py "
            "evaluation/questions.csv"
        )
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path

    try:
        # 평가셋 읽기 및 검증
        questions = load_questions(csv_path)

        # 실제 챗봇 평가
        results = run_evaluation(questions)

        # 결과 저장
        output_path = BASE_DIR / "evaluation" / "results.csv"
        save_results(results, output_path)

        # 점수 요약
        print_summary(results)

    except (FileNotFoundError, ValueError) as error:
        print(f"\n[평가 중단] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
