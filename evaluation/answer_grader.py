"""답변 정확도 채점. 담당: 재웅

평가셋 질문을 챗봇(run_chatbot)에 하나씩 묻고, 답이 맞았는지 센다.

채점 규칙:
    없는 내용 질문   has_evidence가 False ("찾은 문서에 없습니다")면 맞음 (AI 안 부름)
    나머지 질문      채점 AI가 챗봇 답을 모범 답(expected_answer)과 비교해 맞음/틀림
채점 AI는 답변용과 다른 채점 전용 프롬프트를 쓴다. 믿기 전에 20개는 사람 채점과 비교한다.
    채점 AI가 실패하면 한 번 다시 채점하고, 그래도 실패하면 "채점 실패"로 따로 세어 점수에서 뺀다
    (챗봇 답이 틀린 게 아니라 채점을 못 한 것이라서). 챗봇 자체가 실패하면 틀림으로 센다.

함께 내는 것:
    유형별 정확도     한 문서 / 기본 정보 / 여러 문서 / 후속 질문 / 없는 내용
    답변 시간         챗봇이 답한 시간만 (채점 AI 시간 제외). 평균, 가장 느린 질문, 10초 넘은 수
    질문별 결과       질문, 모범 답, 챗봇 답, 판정, 이유, 틀린 단계, 시간
                      → data/evaluation/answers.csv
                      (사람 채점 20개를 할 때 이 파일에 칸을 더해 적는다.
                      평가셋 내용이 들어 있어서 Git에 안 올라감)

실행: uv run python evaluation/answer_grader.py evaluation/questions.csv
평가셋(questions.csv)은 정답이 들어 있어서 Git에 올리지 않는다 (.gitignore).
질문마다 1·4단계 + 채점 AI를 부르므로 돈이 든다 (gpt-5-mini, 30개에 몇십 원).
"""

import csv
import sys
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from bidmate.config import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL_NAME,
    JUDGE_REASONING_EFFORT,
    OPENAI_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    RESPONSE_TIME_LIMIT_SECONDS,
)
from bidmate.pipeline.graph import get_projects, run_chatbot
from bidmate.schemas import AnswerResult, ChatState

NO_CONTENT_TYPE = "없는 내용"
QUESTION_TYPES = {"한 문서", "여러 문서", "후속 질문", NO_CONTENT_TYPE, "기본 정보"}
ANSWER_ROWS_PATH = PROJECT_ROOT / "data" / "evaluation" / "answers.csv"  # 질문별 결과
JUDGE_ATTEMPTS = 2  # 채점 AI 실패 시 한 번 더 (토큰 부족·시간 초과는 다시 하면 되는 경우가 많다)
LIST_SEPARATOR = "|"  # 평가셋에서 여러 값을 한 칸에 적을 때 나누는 기호

# 고정 부분이라 맨 앞에 둔다 (프롬프트 캐싱). 바뀌는 값(질문·답)은 사람 메시지로 따로 넣는다
JUDGE_SYSTEM_PROMPT = """너는 RFP 질의응답 채점자다.
질문, 모범 답, 챗봇 답을 보고 챗봇 답이 맞았는지 판단한다.

맞음:
- 모범 답의 핵심 내용(금액, 날짜, 기관명, 숫자, 요구 조건)이 챗봇 답에 모두 있고 틀린 곳이 없다
- 표현이나 단위가 달라도 뜻이 같으면 맞다 (예: 1억 2천만 원 = 120,000,000원)
- 모범 답에 없는 내용을 더 말해도, 모범 답과 어긋나지 않으면 맞다

틀림:
- 핵심 내용이 하나라도 빠졌거나 다르다
- 모범 답과 어긋나는 내용이 있다
- 답이 있는 질문인데 "문서에 없다"고 답했다

이유는 한 문장으로 쓴다."""

Chatbot = Callable[[str, list[tuple[str, str]]], ChatState]


class JudgeResult(BaseModel):
    """채점 AI가 채우는 양식."""

    is_correct: bool = Field(description="챗봇 답이 모범 답과 맞으면 true")
    reason: str = Field(description="맞음/틀림 이유 한 문장")


def make_judge_model() -> Runnable:
    """채점 AI를 만든다. invoke(메시지)하면 JudgeResult를 돌려준다."""
    model = ChatOpenAI(
        model=JUDGE_MODEL_NAME,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    return model.with_structured_output(JudgeResult)


def load_questions(path: Path) -> list[dict]:
    """평가셋 csv를 읽는다.

    받는 것: csv 경로 (칸은 questions_template.csv와 같다)
    넘기는 것: 줄마다 dict. expected_project_names는 | 로 나눈 목록
    평가셋에 잘못 적은 곳이 있으면 ValueError (돈 쓰기 전에 멈춘다)
    """
    # 1. utf-8-sig: 엑셀로 저장한 csv 맨 앞의 보이지 않는 글자(BOM)를 지운다
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # 2. 정답 문서는 여러 개일 수 있어서 목록으로 (빈 칸은 빈 목록)
    for row in rows:
        text = row.get("expected_project_names") or ""
        row["expected_project_names"] = [
            name.strip() for name in text.split(LIST_SEPARATOR) if name.strip()
        ]

    # 3. 잘못 적은 곳 찾기 (조용히 틀린 점수가 나오지 않게, 한 번에 모두 알려 준다)
    problems = find_problems(rows)
    if problems:
        raise ValueError("평가셋을 고쳐 주세요:\n" + "\n".join(problems))
    return rows


def find_problems(rows: list[dict]) -> list[str]:
    """평가셋에서 잘못 적은 곳을 찾는다.

    받는 것: csv를 읽은 줄 목록
    넘기는 것: 문제 설명 목록 (없으면 빈 목록)
    """
    problems = []
    seen_question_ids = set()
    for line_number, row in enumerate(rows, start=2):  # 1줄은 칸 이름이라 2줄부터
        question_id = (row.get("question_id") or "").strip()
        question_type = (row.get("question_type") or "").strip()

        # 1. 번호가 겹치면 틀린 질문을 셀 때 하나로 합쳐진다
        if question_id in seen_question_ids:
            problems.append(f"{line_number}줄 {question_id}: 질문 번호가 겹친다")
        seen_question_ids.add(question_id)

        # 2. 유형이 다섯 가지와 글자까지 같아야 없는 내용 질문을 제대로 가른다
        if question_type not in QUESTION_TYPES:
            problems.append(f"{line_number}줄 {question_id}: 모르는 유형 '{question_type}'")

        # 3. 질문은 꼭 있어야 하고, 없는 내용이 아니면 모범 답도 있어야 채점할 수 있다
        if not (row.get("question") or "").strip():
            problems.append(f"{line_number}줄 {question_id}: 질문이 비어 있다")
        if question_type != NO_CONTENT_TYPE and not (row.get("expected_answer") or "").strip():
            problems.append(f"{line_number}줄 {question_id}: 모범 답이 비어 있다")
    return problems


def ask_chatbot(question_row: dict, chatbot: Chatbot = run_chatbot) -> AnswerResult:
    """평가셋 한 줄을 챗봇에 묻는다. 후속 질문이면 앞 질문을 먼저 물어 대화 기록을 만든다.

    받는 것: 평가셋 한 줄, 챗봇 함수 (테스트에서는 가짜를 넣는다)
    넘기는 것: 마지막 질문의 AnswerResult
    """
    # 1. 후속 질문: 앞 질문의 실제 답으로 대화 기록을 만든다 (사용자가 겪는 상황과 같게)
    chat_history: list[tuple[str, str]] = []
    previous_question = (question_row.get("previous_question") or "").strip()
    if previous_question:
        previous_state = chatbot(previous_question, [])
        chat_history = [(previous_question, previous_state["answer_result"].answer)]

    # 2. 채점할 질문
    state = chatbot(question_row["question"], chat_history)
    return state["answer_result"]


def judge_answer(
    question_row: dict, answer_result: AnswerResult, judge_model: Runnable
) -> JudgeResult:
    """답 하나가 맞았는지 판단한다.

    받는 것: 평가셋 한 줄, 챗봇 답, 채점 AI (테스트에서는 가짜)
    넘기는 것: JudgeResult (맞음 여부 + 이유)
    """
    # 1. 없는 내용 질문: 근거 없다고 답해야 맞다. AI를 부를 필요가 없다
    if question_row["question_type"] == NO_CONTENT_TYPE:
        if answer_result.has_evidence:
            return JudgeResult(is_correct=False, reason="없는 내용인데 답을 지어냈다")
        return JudgeResult(is_correct=True, reason="없는 내용이라고 답했다")

    # 2. 나머지: 채점 AI가 모범 답과 비교한다
    messages = [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            f"질문: {question_row['question']}\n"
            f"모범 답: {question_row['expected_answer']}\n"
            f"챗봇 답: {answer_result.answer}",
        ),
    ]
    return judge_model.invoke(messages)


def judge_with_retry(
    question_row: dict, answer_result: AnswerResult, judge_model: Runnable
) -> JudgeResult | None:
    """채점 AI가 실패하면 JUDGE_ATTEMPTS번까지 다시 채점한다. 끝내 실패하면 None (채점 실패).

    받는 것: 평가셋 한 줄, 챗봇 답, 채점 AI
    넘기는 것: JudgeResult, 또는 채점을 못 했으면 None
    """
    for _ in range(JUDGE_ATTEMPTS):
        try:
            return judge_answer(question_row, answer_result, judge_model)
        except Exception:
            continue
    return None


def find_failed_stage(question_row: dict, state: ChatState) -> str:
    """틀린 질문이 어느 단계에서 처음 어긋났는지 코드로 판정한다 (점수가 아니라 진단용, AI 안 부름).

    받는 것: 평가셋 한 줄, 챗봇이 돌려준 상태 (고른 문서, 찾은 조각)
    넘기는 것: "2단계" / "3단계" / "4단계" / "3·4단계" (키워드가 없어 둘을 못 가를 때)
    """
    # 1. 없는 내용 질문을 틀렸으면 답을 지어낸 것 (4단계)
    if question_row["question_type"] == NO_CONTENT_TYPE:
        return "4단계"

    # 2. 2단계: 고른 문서가 정답 문서와 다르면
    expected_names = question_row["expected_project_names"]
    name_by_id = {project.project_id: project.project_name for project in get_projects()}
    found_names = {name_by_id.get(project_id) for project_id in state.get("project_ids", [])}
    if found_names != set(expected_names):
        return "2단계"

    # 3. 3단계: 찾은 조각에 정답 문서마다 그 문서 키워드가 없으면 (키워드는 사업명 순서대로 | )
    keywords_text = (question_row.get("answer_keywords") or "").strip()
    if not keywords_text:
        return "3·4단계"
    for name, keyword in zip(expected_names, keywords_text.split(LIST_SEPARATOR)):
        if not any(
            document.metadata.get("project_name") == name
            and keyword.strip() in document.page_content
            for document, _ in state.get("chunks", [])
        ):
            return "3단계"

    # 4. 조각에 답이 있었는데 틀렸으면 4단계
    return "4단계"


def score(
    questions: list[dict],
    chatbot: Chatbot = run_chatbot,
    judge_model: Runnable | None = None,
) -> dict:
    """평가셋 전체의 답변 정확도. 모든 채점 파일이 같은 모양으로 돌려준다.

    받는 것: load_questions()가 돌려준 목록, 챗봇 함수, 채점 AI (없으면 만든다)
    넘기는 것: {
        "score": 맞은 수 ÷ 채점한 수 (채점 실패는 뺀다, 0.0~1.0),
        "counted": 채점한 질문 수 (채점 실패 제외),
        "wrong_question_ids": 틀린 질문 번호, "wrong_reasons": {질문 번호: 틀린 이유},
        "judge_failed_question_ids": 채점 AI가 끝내 실패한 질문 번호,
        "by_type": {유형: (맞은 수, 채점한 수)},
        "average_seconds" / "slowest_seconds": 챗봇 답변 시간 (채점 시간 제외),
        "over_limit_count": 답변 시간이 RESPONSE_TIME_LIMIT_SECONDS를 넘은 질문 수,
        "rows": 질문별 결과 (save_rows로 저장),
    }
    """
    # 1. 채점 AI는 실제로 돌릴 때만 만든다 (테스트는 가짜를 넣는다)
    judge_model = judge_model or make_judge_model()

    # 2. 챗봇이 답한 시간만 재려고 감싼다.
    #    후속 질문은 앞 질문도 묻는데, 마지막(채점할) 질문 시간만 쓴다
    #    틀린 단계를 판정하려고 마지막 상태(고른 문서, 찾은 조각)도 같이 남긴다
    call_seconds: list[float] = []
    call_states: list[ChatState] = []

    def timed_chatbot(question: str, chat_history: list[tuple[str, str]]) -> ChatState:
        start_time = time.perf_counter()
        state = chatbot(question, chat_history)
        call_seconds.append(time.perf_counter() - start_time)
        call_states.append(state)
        return state

    # 3. 질문마다 묻고 채점 (한 질문이 실패해도 나머지는 계속한다)
    rows = []
    for question_row in questions:
        row = {
            "question_id": question_row["question_id"],
            "question_type": question_row["question_type"],
            "question": question_row["question"],
            "expected_answer": question_row.get("expected_answer", ""),
            "chatbot_answer": "",
            "has_evidence": "",
            "verdict": "",  # 맞음 / 틀림 / 채점 실패
            "reason": "",
            "failed_stage": "",  # 틀렸을 때만: 2단계 / 3단계 / 4단계 / 3·4단계
            "seconds": "",
        }
        rows.append(row)

        # 3-1. 챗봇에 묻기. 챗봇이 실패하면(API 시간 초과 등) 틀림으로 센다
        try:
            answer_result = ask_chatbot(question_row, timed_chatbot)
        except Exception as error:
            row.update(verdict="틀림", reason=f"챗봇 에러: {error}")
            continue
        row.update(
            chatbot_answer=answer_result.answer,
            has_evidence=answer_result.has_evidence,
            seconds=round(call_seconds[-1], 2),
        )

        # 3-2. 채점. 채점 AI가 끝내 실패하면 "채점 실패"로 따로 둔다
        judge_result = judge_with_retry(question_row, answer_result, judge_model)
        if judge_result is None:
            row.update(verdict="채점 실패", reason="채점 AI가 두 번 모두 실패")
        else:
            row.update(
                verdict="맞음" if judge_result.is_correct else "틀림",
                reason=judge_result.reason,
            )
            if not judge_result.is_correct:
                row["failed_stage"] = find_failed_stage(question_row, call_states[-1])

    # 4. 점수: 채점 실패는 빼고 계산한다 (질문이 0개면 0점)
    graded_rows = [row for row in rows if row["verdict"] != "채점 실패"]
    wrong_rows = [row for row in graded_rows if row["verdict"] == "틀림"]
    counted = len(graded_rows)
    by_type: dict[str, tuple[int, int]] = {}
    for question_type in dict.fromkeys(row["question_type"] for row in graded_rows):
        typed_rows = [row for row in graded_rows if row["question_type"] == question_type]
        if typed_rows:
            correct = sum(row["verdict"] == "맞음" for row in typed_rows)
            by_type[question_type] = (correct, len(typed_rows))
    seconds = [row["seconds"] for row in rows if row["seconds"] != ""]
    return {
        "score": (counted - len(wrong_rows)) / counted if counted else 0.0,
        "counted": counted,
        "wrong_question_ids": [row["question_id"] for row in wrong_rows],
        "wrong_reasons": {row["question_id"]: row["reason"] for row in wrong_rows},
        "judge_failed_question_ids": [
            row["question_id"] for row in rows if row["verdict"] == "채점 실패"
        ],
        "by_type": by_type,
        "average_seconds": sum(seconds) / len(seconds) if seconds else 0.0,
        "slowest_seconds": max(seconds, default=0.0),
        "over_limit_count": sum(second > RESPONSE_TIME_LIMIT_SECONDS for second in seconds),
        "rows": rows,
    }


def save_rows(rows: list[dict], path: Path = ANSWER_ROWS_PATH) -> None:
    """질문별 결과를 csv로 저장한다. data/ 아래라 Git에 안 올라간다. 실행마다 덮어쓴다."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result: dict) -> None:
    """답변 채점 결과를 사람이 읽기 좋게 출력한다."""
    correct_count = result["counted"] - len(result["wrong_question_ids"])
    print(f"[답변] 정확도 {result['score']:.1%} ({correct_count}/{result['counted']})")
    for question_type, (correct, total) in result["by_type"].items():
        print(f"  {question_type}: {correct}/{total}")
    print(
        f"  답변 시간: 평균 {result['average_seconds']:.1f}초, "
        f"가장 느림 {result['slowest_seconds']:.1f}초, "
        f"{RESPONSE_TIME_LIMIT_SECONDS}초 넘음 {result['over_limit_count']}개"
    )
    if result["judge_failed_question_ids"]:
        print(f"  채점 실패(점수에서 뺌): {', '.join(result['judge_failed_question_ids'])}")
    for question_id, reason in result["wrong_reasons"].items():
        print(f"  틀림 {question_id}: {reason}")


if __name__ == "__main__":
    questions_path = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluation/questions.csv")
    result = score(load_questions(questions_path))
    print_result(result)
    save_rows(result["rows"])
    print(f"질문별 결과: {ANSWER_ROWS_PATH}")
