"""Generate unreviewed practice questions with the local Ollama model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from evaluation.search_b_experiments.nov17.cases import csv_rows, normalized

SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "question": {"type": "string"}},
                "required": ["id", "question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}
INSTRUCTION = (
    "각 입력의 summary에 나온 구체적인 작업을 바탕으로 한국어 입찰 검색 질문을 하나씩 써라. "
    "문서에 근거하는 서로 구별되는 작업 두 가지를 짧고 자연스러운 동사로 표현한다. "
    "사업명, 기관명, 고유 약어, 공고 번호와 연도를 질문에 쓰지 않는다. 원문 어순을 피하고 "
    "쉬운 우리말로 바꾼다. 예산 수치나 실제 날짜는 묻기만 하고 추측하지 않는다. id를 보존한다."
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def compact(text: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", normalized(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1611)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    projects = csv_rows(args.catalog)
    client = build_opener(ProxyHandler({}), NoRedirect())
    output = []
    for offset in range(0, len(projects), args.batch_size):
        batch = {
            index: row
            for index, row in enumerate(projects[offset : offset + args.batch_size], start=offset)
        }
        records = [
            {"id": index, "title": row["사업명"], "summary": row["사업 요약"][:550]}
            for index, row in batch.items()
        ]
        payload = {
            "model": "qwen3:1.7b",
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "options": {
                "temperature": 0,
                "seed": args.seed,
                "num_predict": 950,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": json.dumps(records, ensure_ascii=False)},
            ],
        }
        request = Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with client.open(request, timeout=180) as response:
            questions = json.loads(json.load(response)["message"]["content"])["questions"]
        seen = set()
        for item in questions:
            index, question = item["id"], item["question"].strip()
            if index not in batch or index in seen or len(question) < 18:
                continue
            project = batch[index]
            if any(compact(project[k]) in compact(question) for k in ("사업명", "발주 기관")):
                continue
            seen.add(index)
            output.append(
                {
                    "question": question,
                    "expected_ids": [project["파일명"]],
                    "agency_groups": [project["발주 기관"]],
                    "category": "summary_paraphrase",
                    "label_status": "generated_not_human_reviewed",
                }
            )
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
