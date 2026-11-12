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

점수 (검색 점수와 답변 점수를 따로 낸다):
    검색 적중률   찾아온 조각 k개 안에 정답 문서 조각이 하나라도 있는 질문 비율 (코드 채점)
    정확성        답변이 정답과 맞는가 (1주차 사람, 2주차부터 AI)
    근거 충실도   찾은 조각에 있는 내용만 말했는가 (2주차부터 AI)
    AI 채점은 답변용과 다른 채점 전용 프롬프트로 하고, 믿기 전에 20개를 사람 채점과 비교한다.

한 번에 돌리는 순서: 평가셋 검사 → 검색 점수(score_search.py) → 답변 점수(answer_grader.py)
    → 점수 한 줄을 data/evaluation/scores.csv에 쌓는다 (실험끼리 비교용, Git에 안 올라감)
    --search-only 를 붙이면 돈이 거의 안 드는 검색 점수만 낸다.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import answer_grader
import score_search

SCORES_PATH = score_search.RESULT_FOLDER / "scores.csv"  # 실행마다 한 줄씩 쌓는 점수 기록


def save_score_line(line: dict) -> None:
    """점수 한 줄을 scores.csv 끝에 붙인다. 파일이 없으면 칸 이름부터 쓴다."""
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not SCORES_PATH.exists()
    with open(SCORES_PATH, "a", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(line))
        if is_new_file:
            writer.writeheader()
        writer.writerow(line)


def main(questions_path: Path, search_only: bool) -> None:
    """평가셋 하나로 검색 점수와 답변 점수를 차례로 내고 기록한다."""
    # 1. 평가셋 검사: 잘못 적은 곳이 있으면 돈 쓰기 전에 여기서 멈춘다
    answer_grader.load_questions(questions_path)
    questions = score_search.load_questions(questions_path)

    # 2. 검색 점수 (실제 흐름 / 검색만)
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
        "검색_실제흐름": round(pipeline_result["score"], 3),
        "검색_정답문서만": round(search_only_result["score"], 3),
        "답변_정확도": "",
        "질문당_평균_초": "",
        "답변_틀린_질문": "",
    }

    # 3. 답변 점수 (질문마다 챗봇 답변 + 채점 AI → 돈이 든다)
    if not search_only:
        start_time = time.perf_counter()
        answer_result = answer_grader.score(answer_grader.load_questions(questions_path))
        seconds_per_question = (time.perf_counter() - start_time) / max(len(questions), 1)
        print(
            f"\n[답변] 정확도 {answer_result['score']:.1%}, "
            f"질문당 평균 {seconds_per_question:.1f}초"
        )
        for question_id, reason in answer_result["wrong_reasons"].items():
            print(f"  틀림 {question_id}: {reason}")
        line["답변_정확도"] = round(answer_result["score"], 3)
        line["질문당_평균_초"] = round(
            seconds_per_question, 1
        )  # 채점 AI 시간 포함이라 실제 응답보다 길다
        line["답변_틀린_질문"] = " ".join(answer_result["wrong_question_ids"])

    # 4. 점수 기록
    save_score_line(line)
    print(f"\n점수 기록: {SCORES_PATH}")


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    main(
        Path(arguments[0] if arguments else "evaluation/questions.csv"),
        search_only="--search-only" in sys.argv,
    )
