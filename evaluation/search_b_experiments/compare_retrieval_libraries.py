from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import syhwp
from langchain_core.documents import Document

from evaluation.score_search import check_content_hit, keyword_in_document
from evaluation.search_b_experiments.compare_chunks import select, terms

WORD = re.compile(r"[가-힣a-z0-9]+")


def text_key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def tokenized(text: str) -> str:
    return " ".join(term.replace(":", "_") for term in terms(text))


def phrase_query(text: str, mode: str) -> str:
    if mode == "word_bigram":
        phrases = set(tokenized(text).split())
    else:
        compact = "".join(WORD.findall(text_key(text)))
        phrases = {compact[i : i + 3] for i in range(len(compact) - 2)}
    return " OR ".join('"' + phrase + '"' for phrase in sorted(phrases))


class SQLiteSearch:
    def __init__(self, database: sqlite3.Connection, pieces: list[Document], mode: str):
        self.pieces = pieces
        self.database = database
        self.mode = mode
        self.by_project: dict[str, list[int]] = defaultdict(list)
        db_mode = "trigram" if mode == "native_trigram" else "unicode61 tokenchars _"
        database.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {mode} USING fts5("
            f"text, project_id UNINDEXED, tokenize='{db_mode}')"
        )
        rows = []
        for i, piece in enumerate(pieces):
            project = piece.metadata["project_id"]
            self.by_project[project].append(i)
            body = piece.page_content.split("\n", 1)[-1]
            body = (
                tokenized(body) if mode == "word_bigram" else "".join(WORD.findall(text_key(body)))
            )
            rows.append((i + 1, body, project))
        database.executemany(f"INSERT INTO {mode}(rowid, text, project_id) VALUES(?, ?, ?)", rows)

    def rank(self, question: str, project_ids: list[str], k: int):
        query = phrase_query(question, self.mode)
        if not project_ids:
            return []
        table = self.mode
        sql = (
            f"SELECT rowid, bm25({table}) FROM {table} WHERE {table} MATCH ? "
            f"AND project_id IN ({','.join('?' for _ in project_ids)}) "
            f"ORDER BY 2, 1"
        )
        scored = list(self.database.execute(sql, [query, *project_ids])) if query else []
        order = [rowid - 1 for rowid, _ in scored]
        scores = {rowid - 1: -score for rowid, score in scored}
        order.extend(sorted({i for p in project_ids for i in self.by_project[p]} - scores.keys()))
        return select(self, order, scores, project_ids, "per_document_first", k)


def measure(rows, backend, k):
    output = []
    for row in rows:
        keywords = {pid: evidence["keywords"] for pid, evidence in row["evidence"].items()}
        start = time.perf_counter_ns()
        chunks = backend.rank(row["question"], row["selected"], k)
        latency = (time.perf_counter_ns() - start) / 1e6
        output.append(
            {
                "question_id": row["question_id"],
                "documents": {chunk.metadata["project_id"] for chunk, _ in chunks}
                >= set(row["expected"]),
                "content_rank": check_content_hit(chunks, keywords) if keywords else None,
                "milliseconds": latency,
            }
        )
    return output


def audit_tables(rows, raw_dir: Path):
    failed = [
        row
        for row in rows
        if len(row["expected"]) > 1
        and row["results"]["actual"]["without_header_per_document"]["content_rank"] is None
    ]
    by_id = {}
    audit = []
    for row in failed:
        for project_id, evidence in row["evidence"].items():
            values = evidence["keywords"]
            if not values or Path(project_id).suffix.lower() != ".hwp":
                continue
            if project_id not in by_id:
                blocks = syhwp.open(str(raw_dir / project_id)).blocks
                by_id[project_id] = [
                    (
                        index,
                        cell.row,
                        cell.col,
                        cell.text,
                    )
                    for index, block in enumerate(blocks)
                    if type(block).__name__ == "Table"
                    for cell in block.cells
                    if cell.text
                ]
            cells = [
                {"block": block, "row": r, "column": c}
                for block, r, c, text in by_id[project_id]
                if keyword_in_document(values, [text])
            ]
            audit.append(
                {
                    "question_id": row["question_id"],
                    "found_in_body": evidence["body"],
                    "found_in_body_chunk": evidence["body_chunk"],
                    "cells": cells,
                }
            )
    return audit


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("corpus", "reference", "raw_dir", "output"):
        parser.add_argument(name, type=Path)
    cfg = parser.parse_args()
    reference = json.loads(cfg.reference.read_text())
    corpus = json.loads(cfg.corpus.read_text())
    assert reference["corpus"] == file_hash(cfg.corpus)
    rows = reference["rows"]
    pieces = [Document.model_validate(item) for item in corpus["chunks"]]
    methods = {}
    with closing(sqlite3.connect(":memory:")) as db:
        for mode in ("word_bigram", "native_trigram"):
            start = time.perf_counter()
            backend = SQLiteSearch(db, pieces, mode)
            build_seconds = time.perf_counter() - start
            predicted = measure(rows, backend, reference["k"])
            results = {}
            for group, items in (
                ("all", list(zip(rows, predicted, strict=True))),
                (
                    "multiple",
                    [
                        (row, result)
                        for row, result in zip(rows, predicted, strict=True)
                        if len(row["expected"]) > 1
                    ],
                ),
            ):
                scored = [(row, result) for row, result in items if row["scorable"] == "scorable"]
                compared = [
                    (
                        row["results"]["actual"]["without_header_per_document"]["content_rank"]
                        is not None,
                        result["content_rank"] is not None,
                    )
                    for row, result in scored
                ]
                timings = sorted(result["milliseconds"] for _, result in items)
                results[group] = {
                    "documents": [sum(result["documents"] for _, result in items), len(items)],
                    "evidence": [sum(after for _, after in compared), len(compared)],
                    "wins": sum(after and not before for before, after in compared),
                    "losses": sum(before and not after for before, after in compared),
                    "p95_ms": round(timings[math.ceil(0.95 * len(timings)) - 1], 2),
                }
            methods[mode] = {
                "scores": results,
                "build_seconds": round(build_seconds, 3),
                "questions": predicted,
            }
            print(mode, results, "build_seconds", round(build_seconds, 3), flush=True)
    table_audit = audit_tables(rows, cfg.raw_dir)
    print(
        "hwp keyword targets",
        len(table_audit),
        "existing chunks",
        sum(item["found_in_body_chunk"] for item in table_audit),
        "any table cell",
        sum(bool(item["cells"]) for item in table_audit),
        flush=True,
    )
    cfg.output.write_text(
        json.dumps(
            {
                "experiment": "bidmate_retrieval_libraries_20261116",
                "sqlite": sqlite3.sqlite_version,
                "syhwp": version("syhwp"),
                "input_sha256": {
                    "corpus": file_hash(cfg.corpus),
                    "reference": file_hash(cfg.reference),
                },
                "k": reference["k"],
                "methods": methods,
                "hwp_audit": table_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
