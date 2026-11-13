"""답변 정확도 채점자 테스트. OpenAI를 부르지 않는다 (챗봇·채점 AI 모두 가짜)."""

import pytest
from langchain_core.runnables import RunnableLambda

from bidmate.schemas import AnswerResult
from evaluation.answer_grader import JudgeResult, ask_chatbot, judge_answer, load_questions, score

BUDGET_QUESTION = {
    "question_id": "Q001",
    "question_type": "기본 정보",
    "question": "가나다공단 이러닝 사업 예산은?",
    "previous_question": "",
    "expected_answer": "예산은 1억 2천만 원이다.",
}
NO_CONTENT_QUESTION = {
    "question_id": "Q002",
    "question_type": "없는 내용",
    "question": "가나다공단 이러닝 사업 점심 메뉴는?",
    "previous_question": "",
    "expected_answer": "",
}


def make_answer(answer: str, has_evidence: bool = True) -> AnswerResult:
    return AnswerResult(
        answer=answer, source_project_names=[], has_evidence=has_evidence, is_rfp_question=True
    )


def make_fake_judge(is_correct: bool, received_messages: list | None = None):
    """정해 둔 결과를 돌려주는 가짜 채점 AI. 받은 메시지를 기록할 수 있다."""

    def fake_judge(messages):
        if received_messages is not None:
            received_messages.append(messages)
        return JudgeResult(is_correct=is_correct, reason="가짜 채점")

    return RunnableLambda(fake_judge)


def test_judge_sends_question_and_both_answers():
    # 채점 AI에게 질문·모범 답·챗봇 답이 모두 간다
    received_messages = []
    judge = make_fake_judge(True, received_messages)

    result = judge_answer(BUDGET_QUESTION, make_answer("1억2천만 원입니다."), judge)

    assert result.is_correct
    human_message = received_messages[0][-1][1]
    assert "가나다공단 이러닝 사업 예산은?" in human_message
    assert "예산은 1억 2천만 원이다." in human_message
    assert "1억2천만 원입니다." in human_message


def test_no_content_question_skips_judge():
    # 없는 내용 질문은 채점 AI를 부르지 않고 has_evidence로만 판단
    received_messages = []
    judge = make_fake_judge(True, received_messages)

    no_evidence = make_answer("찾은 문서에 없습니다.", has_evidence=False)
    made_up = make_answer("김치찌개입니다.", has_evidence=True)

    assert judge_answer(NO_CONTENT_QUESTION, no_evidence, judge).is_correct
    assert not judge_answer(NO_CONTENT_QUESTION, made_up, judge).is_correct
    assert received_messages == []


def test_follow_up_question_gets_previous_answer_as_history():
    # 후속 질문은 앞 질문을 먼저 묻고, 그 답을 대화 기록으로 넘긴다
    calls = []

    def fake_chatbot(question, chat_history):
        calls.append((question, chat_history))
        return {"answer_result": make_answer(f"{question}의 답")}

    follow_up = {
        **BUDGET_QUESTION,
        "question": "그럼 예산은?",
        "previous_question": "이러닝 사업 알려줘",
    }
    ask_chatbot(follow_up, fake_chatbot)

    assert calls == [
        ("이러닝 사업 알려줘", []),
        ("그럼 예산은?", [("이러닝 사업 알려줘", "이러닝 사업 알려줘의 답")]),
    ]


def test_score_counts_and_lists_wrong_questions():
    # 챗봇은 늘 근거 있다고 답함 → 예산 질문은 맞음(가짜 채점), 없는 내용 질문은 틀림
    def fake_chatbot(question, chat_history):
        return {"answer_result": make_answer("예산은 1억 2천만 원입니다.")}

    result = score([BUDGET_QUESTION, NO_CONTENT_QUESTION], fake_chatbot, make_fake_judge(True))

    assert result["score"] == 0.5
    assert result["counted"] == 2
    assert result["wrong_question_ids"] == ["Q002"]
    assert "Q002" in result["wrong_reasons"]


def test_load_questions_splits_project_names(tmp_path):
    # | 로 나눈 정답 문서는 목록, 빈 칸은 빈 목록
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "question_id,question_type,question,previous_question,expected_answer,"
        "expected_project_names\n"
        'Q001,여러 문서,두 사업 예산은?,,"1억, 2억",이러닝 | 포털\n'
        "Q002,없는 내용,점심은?,,,\n",
        encoding="utf-8",
    )

    questions = load_questions(csv_path)

    assert questions[0]["expected_project_names"] == ["이러닝", "포털"]
    assert questions[0]["expected_answer"] == "1억, 2억"
    assert questions[1]["expected_project_names"] == []


def test_one_failure_does_not_stop_scoring():
    # 한 질문에서 에러가 나도 나머지는 채점하고, 실패한 질문은 틀림으로 센다
    def flaky_chatbot(question, chat_history):
        if "예산" in question:
            raise TimeoutError("시간 초과")
        return {"answer_result": make_answer("찾은 문서에 없습니다.", has_evidence=False)}

    result = score([BUDGET_QUESTION, NO_CONTENT_QUESTION], flaky_chatbot, make_fake_judge(True))

    assert result["counted"] == 2
    assert result["wrong_question_ids"] == ["Q001"]
    assert "시간 초과" in result["wrong_reasons"]["Q001"]


def test_load_questions_reports_mistakes_before_spending(tmp_path):
    # 모르는 유형, 빈 모범 답, 겹친 번호를 한 번에 알려 준다
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "question_id,question_type,question,previous_question,expected_answer,"
        "expected_project_names\n"
        "Q001,없는내용,점심은?,,,\n"
        "Q002,기본 정보,예산은?,,,이러닝\n"
        "Q002,한 문서,마감은?,,5월 31일,이러닝\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_questions(csv_path)
    message = str(error.value)

    assert "모르는 유형 '없는내용'" in message
    assert "Q002: 모범 답이 비어 있다" in message
    assert "질문 번호가 겹친다" in message


def make_flaky_judge(failures: int):
    """처음 failures번은 에러를 내고, 그다음부터 맞음을 돌려주는 가짜 채점 AI."""
    calls = []

    def fake_judge(messages):
        calls.append(messages)
        if len(calls) <= failures:
            raise ValueError("length limit was reached")
        return JudgeResult(is_correct=True, reason="가짜 채점")

    return RunnableLambda(fake_judge)


def answer_with_budget(question, chat_history):
    return {"answer_result": make_answer("예산은 1억 2천만 원입니다.")}


def test_judge_failure_once_is_retried():
    # 채점 AI가 한 번 실패해도 다시 채점해서 맞음이 된다
    result = score([BUDGET_QUESTION], answer_with_budget, make_flaky_judge(failures=1))

    assert result["score"] == 1.0
    assert result["judge_failed_question_ids"] == []


def test_judge_failure_twice_is_not_counted_as_wrong():
    # 채점 AI가 두 번 다 실패하면 틀림이 아니라 "채점 실패"로 빼고 계산한다
    result = score(
        [BUDGET_QUESTION, NO_CONTENT_QUESTION], answer_with_budget, make_flaky_judge(failures=9)
    )

    assert result["judge_failed_question_ids"] == ["Q001"]
    assert result["counted"] == 1  # 없는 내용 질문만 채점됨 (AI 안 부름)
    assert result["wrong_question_ids"] == ["Q002"]  # 근거 있다고 답해서 틀림
    assert result["rows"][0]["verdict"] == "채점 실패"


def test_score_reports_types_seconds_and_rows():
    # 유형별 점수, 챗봇 답변 시간, 질문별 결과를 같이 돌려준다
    result = score(
        [BUDGET_QUESTION, NO_CONTENT_QUESTION], answer_with_budget, make_fake_judge(True)
    )

    assert result["by_type"] == {"기본 정보": (1, 1), "없는 내용": (0, 1)}
    assert result["over_limit_count"] == 0
    assert 0 <= result["average_seconds"] <= result["slowest_seconds"]
    first_row = result["rows"][0]
    assert first_row["chatbot_answer"] == "예산은 1억 2천만 원입니다."
    assert first_row["verdict"] == "맞음"
    assert first_row["expected_answer"] == "예산은 1억 2천만 원이다."


def test_find_failed_stage(monkeypatch):
    # 고른 문서가 다르면 2단계, 조각에 키워드가 없으면 3단계, 있는데 틀렸으면 4단계
    from langchain_core.documents import Document

    from bidmate.schemas import ProjectInfo
    from evaluation import answer_grader

    projects = [ProjectInfo("a.hwp", "가나다공단", "이러닝", None, None)]
    monkeypatch.setattr(answer_grader, "get_projects", lambda: projects)
    question_row = {
        **BUDGET_QUESTION,
        "expected_project_names": ["이러닝"],
        "answer_keywords": "1억 2천만",
    }

    def state(project_ids, text):
        chunk = Document(page_content=text, metadata={"project_name": "이러닝"})
        return {"project_ids": project_ids, "chunks": [(chunk, 0.1)]}

    assert answer_grader.find_failed_stage(question_row, state([], "표지")) == "2단계"
    assert answer_grader.find_failed_stage(question_row, state(["a.hwp"], "표지")) == "3단계"
    assert (
        answer_grader.find_failed_stage(question_row, state(["a.hwp"], "예산 1억 2천만")) == "4단계"
    )
