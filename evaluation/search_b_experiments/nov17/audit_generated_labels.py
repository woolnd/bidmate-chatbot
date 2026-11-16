"""Flag conflicting evidence without changing generated evaluation labels."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


def tokens(text):
    return re.findall(r"[가-힣a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())


def matching_terms(question, text):
    return sum(
        block.size
        for block in SequenceMatcher(
            None, question, tokens(text), autojunk=False
        ).get_matching_blocks()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "questions", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    with args.catalog.open(encoding="utf-8-sig", newline="") as source:
        projects = list(csv.DictReader(source))
    questions = json.loads(args.questions.read_text())
    doc_pos = {p["파일명"]: i for i, p in enumerate(projects)}
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    index = vectorizer.fit_transform([p["사업 요약"] for p in projects])
    values = (vectorizer.transform([q["question"] for q in questions]) @ index.T).toarray()
    fields = ("사업명", "사업 요약", "텍스트")
    findings = []
    for case, row in zip(questions, values, strict=True):
        if len(case["expected_ids"]) != 1 or case["expected_ids"][0] not in doc_pos:
            continue
        gold = doc_pos[case["expected_ids"][0]]
        alternative = int(row.argmax())
        if gold == alternative or row[alternative] < 0.40 or row[alternative] - row[gold] < 0.30:
            continue
        question = tokens(case["question"])
        hits = [
            [matching_terms(question, projects[i][field]) for field in fields]
            for i in (gold, alternative)
        ]
        if not all(a > b for b, a in zip(*hits, strict=True)):
            continue
        findings.append(
            {
                "question_fingerprint": hashlib.sha256(case["question"].encode()).hexdigest()[:12],
                "status": "needs_source_review",
                "summary_cosine": [round(float(row[i]), 4) for i in (gold, alternative)],
                "ordered_token_matches": hits,
                "same_generation_batch": gold // args.batch_size == alternative // args.batch_size,
            }
        )
    args.output.write_text(
        json.dumps(
            {"checked": len(questions), "fields": ["title", "summary", "body"], "flags": findings},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
