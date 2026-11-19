"""평가셋으로 챗봇 점수를 낸다. 담당: 재웅(채점 코드), 데이터 파트(평가셋, 점수 정리)

실행: uv run python evaluation/run_evaluation.py evaluation/questions.csv
평가셋 파일(questions.csv)은 원본 RFP 내용이 들어 있어서 Git에 올리지 않는다.
양식은 evaluation/questions_template.csv 를 복사해서 쓴다.

평가셋 칸:
    question_id             질문 번호 (예: Q001)
    question_type           한 문서 / 여러 문서 / 후속 질문 / 없는 내용 / 기본 정보
    question                질문
    previous_question       후속 질문일 때 바로 앞 질문 (아니면 비움)
    expected_answer         정답 (원본 RFP를 직접 보고 적는다)
    expected_project_names  정답 문서 사업명. 여러 개면 | 로 구분. 없는 내용 질문이면 비움
    answer_keywords         정답 값 (원문 표기 그대로, 여러 문서면 사업명 순서대로 | 로 구분)

기록하는 핵심 숫자 4개 (data/evaluation/scores.csv에 실행마다 한 줄씩 쌓는다):
    1. 답변_정확도      최종 실력. 챗봇 답이 정답과 맞나 (채점 AI, answer_grader.py)
    2. 평균_초 / 가장_느린_초
                        챗봇 답변 시간 (채점 시간 제외). 10초를 넘으면 안 된다
    3. 검색_실제흐름    내용 기준 Hit@5. 2단계가 고른 문서 안에서 찾은 조각 5개에
                        정답 값이 든 정답 문서 조각이 있나 (4단계에 답을 건네줬나)
    4. 검색_정답문서    내용 기준 Hit@5. 정답 문서를 그대로 넣고 잰 값 (3단계만의 실력)
    검색 점수는 후속 질문과 정답 값이 없는 질문을 빼고 잰다 (score_search.py).
    사업명 기준 Hit@5는 너그러워서 기록하지 않고, 질문별 결과 파일에만 남긴다.

한 번에 돌리는 순서: 평가셋 검사 → 검색 점수(score_search.py) → 답변 점수(answer_grader.py)
    → 점수 한 줄을 scores.csv에 쌓는다 (실험끼리 비교용, Git에 안 올라감)
    --search-only 를 붙이면 돈이 거의 안 드는 검색 점수만 낸다.
    --memo="무엇을 바꿨는지" 를 붙이면 그 줄의 메모 칸에 남는다 (예: --memo="PR #40 문서 고르기")
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import answer_grader
import score_search

SCORES_PATH = score_search.RESULT_FOLDER / "scores.csv"  # 실행마다 한 줄씩 쌓는 점수 기록
# scores.csv 칸. 바꾸면 기존 파일과 칸이 어긋나니, 바꿀 때는 파일을 새로 시작한다
SCORE_COLUMNS = (
    "시각",
    "질문 수",
    "답변_정확도",
    "평균_초",
    "가장_느린_초",
    "검색_실제흐름",
    "검색_정답문서",
    "메모",
)


def save_score_line(line: dict) -> None:
    """점수 한 줄을 scores.csv 끝에 붙인다. 파일이 없으면 칸 이름부터 쓴다.

    받는 것: SCORE_COLUMNS 칸을 가진 사전
    넘기는 것: 없음. 기존 파일의 칸이 SCORE_COLUMNS와 다르면 섞이지 않게 멈춘다
    """
    # 1. 기존 파일의 칸이 지금 칸과 다르면 줄이 어긋나므로 멈춘다 (옛 파일은 옮기고 새로 시작)
    if SCORES_PATH.exists():
        with open(SCORES_PATH, encoding="utf-8-sig", newline="") as csv_file:
            existing_columns = next(csv.reader(csv_file), [])
        if tuple(existing_columns) != SCORE_COLUMNS:
            raise ValueError(
                f"{SCORES_PATH}의 칸이 지금 기록 칸과 다릅니다. "
                "옛 파일 이름을 바꿔 두고 다시 실행하세요."
            )

    # 2. 한 줄 붙이기 (새 파일이면 칸 이름부터)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not SCORES_PATH.exists()
    with open(SCORES_PATH, "a", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(SCORE_COLUMNS))
        if is_new_file:
            writer.writeheader()
        writer.writerow(line)


def main(questions_path: Path, search_only: bool, memo: str) -> None:
    """평가셋 하나로 검색 점수와 답변 점수를 차례로 내고 핵심 숫자 4개를 기록한다."""
    # 1. 평가셋 검사: 잘못 적은 곳이 있으면 돈 쓰기 전에 여기서 멈춘다
    answer_grader.load_questions(questions_path)
    questions = score_search.load_questions(questions_path)

    # 2. 검색 점수 (실제 흐름 / 정답 문서를 그대로 넣고)
    pipeline_result = score_search.score(questions, use_expected_projects=False)
    search_only_result = score_search.score(questions, use_expected_projects=True)
    if pipeline_result["unknown_project_names"]:
        print("[경고] 목록표에 없는 정답 사업명:", pipeline_result["unknown_project_names"])
    score_search.print_result("검색 - 실제 흐름", pipeline_result)
    score_search.print_result("검색 - 정답 문서 안에서만", search_only_result)
    score_search.save_rows(
        pipeline_result["rows"], score_search.RESULT_FOLDER / "score_search_pipeline.csv"
    )

    line = {
        "시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "질문 수": len(questions),
        "답변_정확도": "",
        "평균_초": "",
        "가장_느린_초": "",
        "검색_실제흐름": round(pipeline_result["content_score"], 3),
        "검색_정답문서": round(search_only_result["content_score"], 3),
        "메모": memo,
    }

    # 3. 답변 점수 (질문마다 챗봇 답변 + 채점 AI → 돈이 든다)
    if not search_only:
        answer_result = answer_grader.score(answer_grader.load_questions(questions_path))
        print()
        answer_grader.print_result(answer_result)
        answer_grader.save_rows(answer_result["rows"])
        print(f"질문별 답변 결과: {answer_grader.ANSWER_ROWS_PATH}")
        line["답변_정확도"] = round(answer_result["score"], 3)
        line["평균_초"] = round(answer_result["average_seconds"], 1)
        line["가장_느린_초"] = round(answer_result["slowest_seconds"], 1)

    # 4. 점수 기록
    save_score_line(line)
    print(f"\n점수 기록: {SCORES_PATH}")


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    memo_text = next(
        (argument.split("=", 1)[1] for argument in sys.argv[1:] if argument.startswith("--memo=")),
        "",
    )
    main(
        Path(arguments[0] if arguments else "evaluation/questions.csv"),
        search_only="--search-only" in sys.argv,
        memo=memo_text,
    )
