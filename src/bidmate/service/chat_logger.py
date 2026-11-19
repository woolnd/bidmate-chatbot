"""질문과 답변 기록을 파일로 남긴다. 담당: 재웅

로그는 2주차 개선의 근거가 된다. '싫어요'를 받은 질문은 평가셋에 추가한다.
한 줄에 기록 하나씩 JSON으로 쓴다(JSONL). 날짜별로 파일이 나뉜다.
로그에는 RFP 내용이 섞이므로 logs/ 폴더는 Git에 올리지 않는다.

남기는 것: 질문, 찾은 조각, 답변, 응답 시간, 토큰 사용량(캐시 포함), 좋아요/싫어요
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from bidmate.config import FEEDBACK_LOG_FOLDER


def log_feedback(
    question: str,
    answer: str,
    rating: Literal["up", "down"],
    folder: Path = FEEDBACK_LOG_FOLDER,
) -> Path:
    """좋아요/별로예요 하나를 그날 파일(날짜.jsonl)에 한 줄 덧붙인다.

    받는 것: 질문, 답변, 평가("up" 좋아요 / "down" 별로예요), 저장 폴더 (테스트에서 바꾼다)
    넘기는 것: 기록한 파일 경로
    """
    # 1. 날짜별 파일 (폴더가 없으면 만든다)
    now = datetime.now()
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"{now:%Y-%m-%d}.jsonl"

    # 2. 한 줄에 기록 하나 (ensure_ascii=False: 한글을 그대로 읽을 수 있게)
    record = {
        "time": now.isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "rating": rating,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path
