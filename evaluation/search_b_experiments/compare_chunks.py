from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document

from bidmate.config import TOP_K
from bidmate.retrieval.project_finder import select_project_scope
from evaluation import score_search
from evaluation.score_project_finder import _name_key
from evaluation.score_search import (
    FOLLOW_UP_QUESTION_TYPE,
    check_content_hit,
    get_keywords,
    keyword_in_document,
    match_keywords_to_projects,
    split_names,
)
from scripts.evaluate_project_finder import load_projects


def terms(text):
    for token in re.findall(r"[가-힣a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()):
        yield "w:" + token
        if len(token) > 2:
            yield from ("b:" + token[i : i + 2] for i in range(len(token) - 1))


class Index:
    def __init__(self, pieces, omit_header=False):
        self.pieces = pieces
        self.by_project = defaultdict(list)
        self.postings = defaultdict(list)
        lengths = []
        for i, piece in enumerate(pieces):
            self.by_project[piece.metadata["project_id"]].append(i)
            source = piece.page_content.split("\n", 1)[-1] if omit_header else piece.page_content
            counts = Counter(terms(source))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        average_length = statistics.mean(lengths) or 1
        self.norms = [1.2 * (0.25 + 0.75 * length / average_length) for length in lengths]

    def rank(self, text, project_ids):
        scores = {}
        allowed = (
            None
            if project_ids is None
            else {i for pid in project_ids for i in self.by_project[pid]}
        )
        for term in sorted(set(terms(text))):
            posting = self.postings.get(term, ())
            weight = math.log1p((len(self.pieces) - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, count in posting:
                if allowed is None or i in allowed:
                    scores[i] = scores.get(i, 0) + weight * 2.2 * count / (count + self.norms[i])
        matched = range(len(self.pieces)) if allowed is None else allowed
        return sorted(matched, key=lambda i: (-round(scores.get(i, 0), 8), i)), scores


def select(index, order, scores, project_ids, variant, k):
    if (
        variant in {"per_document_first", "without_header_per_document"}
        and project_ids
        and 1 < len(project_ids) <= k
    ):
        top_by_id = {}
        for i in order:
            top_by_id.setdefault(index.pieces[i].metadata["project_id"], i)
            if len(top_by_id) == len(project_ids):
                break
        reserved = set(top_by_id.values())
        order = [i for i in order if i in reserved] + [i for i in order if i not in reserved]
    return [(index.pieces[i], scores.get(i, 0)) for i in order[:k]]


def mask_exact_names(question, selected_projects):
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    names = {p.project_name for p in selected_projects} | {p.agency_name for p in selected_projects}
    for name in sorted(names, key=lambda s: (-len(s), s)):
        value = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
        normalized = normalized.replace(value, " ")
    return normalized if list(terms(normalized)) else question


def evidence_status(ids, texts):
    if not ids:
        return "keyword_missing"
    if any(
        values and not keyword_in_document(values, texts.get(pid, []))
        for pid, values in ids.items()
    ):
        return "absent_in_chunks"
    return "scorable"


def main():
    parser = argparse.ArgumentParser()
    for field in ("catalog", "questions", "corpus", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    projects = load_projects(args.catalog)
    pieces = [Document.model_validate(piece) for piece in corpus["chunks"]]
    names = defaultdict(list)
    for project in projects:
        names[_name_key(project.project_name)].append(project.project_id)
    by_id = {p.project_id: p for p in projects}
    per_doc = defaultdict(list)
    for piece in pieces:
        per_doc[piece.metadata["project_id"]].append(piece.page_content)
    plain = Index(pieces)
    no_header = Index(pieces, omit_header=True)
    variants = {
        "global_top_k": plain,
        "per_document_first": plain,
        "without_chunk_header": no_header,
        "mask_catalog_names": plain,
        "without_header_per_document": no_header,
    }
    rows = []
    with args.questions.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["question_type"] == FOLLOW_UP_QUESTION_TYPE:
                continue
            targets = [
                names[_name_key(name)] for name in split_names(row["expected_project_names"])
            ]
            if any(len(target) != 1 for target in targets):
                raise ValueError(f"Ambiguous or missing label: {row['question_id']}")
            expected = [target[0] for target in targets]
            if not expected:
                continue
            question = row["question"]
            found, catalog_miss = select_project_scope(question, projects)
            matched = match_keywords_to_projects(row, expected) if get_keywords(row) else {}
            origins = {
                pid: {
                    "keywords": keywords,
                    "body": bool(keywords)
                    and keyword_in_document(keywords, [corpus["text"].get(pid, "")]),
                    "body_chunk": bool(keywords)
                    and keyword_in_document(keywords, per_doc.get(pid, [])[1:]),
                    "info": bool(keywords)
                    and keyword_in_document(keywords, per_doc.get(pid, [])[:1]),
                }
                for pid, keywords in matched.items()
            }
            scorable = evidence_status(matched, per_doc)
            info = {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": question,
                "expected": expected,
                "selected": found,
                "evidence": origins,
                "scorable": scorable,
                "results": {},
            }
            for scope, ids in (
                ("oracle", expected),
                (
                    "actual",
                    found or ([] if catalog_miss else None),
                ),
            ):
                attempts = {}
                for variant, index in variants.items():
                    text = question
                    if variant == "mask_catalog_names" and ids:
                        text = mask_exact_names(question, [by_id[pid] for pid in ids])
                    start = time.perf_counter_ns()
                    ordered, scores = index.rank(text, ids)
                    result = select(index, ordered, scores, ids, variant, TOP_K)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    attempts[variant] = {
                        "document_hit": {p.metadata["project_id"] for p, _ in result}
                        >= set(expected),
                        "content_rank": check_content_hit(result, matched) if matched else None,
                        "milliseconds": elapsed,
                        "chunk_ids": [p.metadata["chunk_id"] for p, _ in result],
                    }
                info["results"][scope] = attempts
            rows.append(info)

    summary = {}
    eligibility = Counter(row["scorable"] for row in rows)
    print("eligibility", dict(eligibility), flush=True)
    for scope in ("oracle", "actual"):
        summary[scope] = {}
        for variant in variants:
            stats = {}
            for bucket, data in (
                ("all", rows),
                ("single", [r for r in rows if len(r["expected"]) == 1]),
                ("multiple", [r for r in rows if len(r["expected"]) > 1]),
            ):

                def base(r):
                    return r["results"][scope]["global_top_k"]

                def value(r):
                    return r["results"][scope][variant]

                scorable = [r for r in data if r["scorable"] == "scorable"]
                timings = sorted(value(r)["milliseconds"] for r in data)

                def hit(row):
                    return row["content_rank"] is not None

                stats[bucket] = {
                    "documents": [sum(value(r)["document_hit"] for r in data), len(data)],
                    "content": [sum(hit(value(r)) for r in scorable), len(scorable)],
                    "wins": sum(hit(value(r)) and not hit(base(r)) for r in scorable),
                    "losses": sum(hit(base(r)) and not hit(value(r)) for r in scorable),
                    "p95_ms": round(timings[math.ceil(0.95 * len(timings)) - 1], 2)
                    if timings
                    else None,
                }
            summary[scope][variant] = stats
            print(scope, variant, stats, flush=True)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    args.output.write_text(
        json.dumps(
            {
                "method": "offline sparse retrieval (word and character bigram BM25)",
                "k": TOP_K,
                "corpus": digest(args.corpus),
                "questions": digest(args.questions),
                "team_scorer": digest(Path(score_search.__file__)),
                "eligibility": eligibility,
                "metrics": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
