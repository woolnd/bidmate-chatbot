from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from unittest.mock import patch

from bidmate.pipeline import graph
from bidmate.retrieval.vector_store import make_project_filter, search_chunks
from evaluation.score_project_finder import _load_baseline, _sha256, score
from evaluation.score_search import check_content_hit, load_questions
from evaluation.search_b_experiments.build_local_chroma import (
    LocalEmbeddings,
    digests,
    open_database,
)
from scripts.evaluate_project_finder import load_projects


class CachedQueries(LocalEmbeddings):
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        if text not in self.cache:
            self.cache[text] = super().embed_query(text)
        return self.cache[text]


def measure(rows, graph_results, store, embeddings, k, oracle):
    for row in rows:
        expected = row["expected"]
        state = graph_results[row["question_id"]]
        project_ids = expected if oracle else state["project_ids"]
        catalog_miss = not oracle and state["catalog_miss"]
        question = row["question"]
        keywords = {pid: content["keywords"] for pid, content in row["evidence"].items()}
        if not catalog_miss:
            embeddings.embed_query(question)
        results = {}
        for label in ("global", "cover_documents"):
            embeddings.calls = 0
            start = time.perf_counter_ns()
            if catalog_miss:
                chunks = []
            elif label == "global":
                chunks = store.similarity_search_with_score(
                    question, k=k, filter=make_project_filter(project_ids)
                )
            else:
                chunks = search_chunks(question, project_ids, k, store)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert embeddings.calls == int(not catalog_miss)
            assert len(chunks) <= k
            ids = {doc.metadata["project_id"] for doc, _ in chunks}
            if project_ids:
                assert ids <= set(project_ids)
                if label == "cover_documents" and len(project_ids) <= k:
                    assert ids == set(project_ids)
            results[label] = {
                "all_documents": set(expected) <= ids,
                "rank": check_content_hit(chunks, keywords) if keywords else None,
                "ms": elapsed,
                "chunks": [doc.metadata["chunk_id"] for doc, _ in chunks],
            }
        yield {
            "id": row["question_id"],
            "multi": len(expected) > 1,
            "scorable": row["scorable"] == "scorable",
            "result": results,
        }


def summarize(rows):
    result = {}
    for label in ("global", "cover_documents"):
        result[label] = {}
        for group, data in (
            ("all", rows),
            ("single", [row for row in rows if not row["multi"]]),
            ("multiple", [row for row in rows if row["multi"]]),
        ):
            result_rows = [row["result"][label] for row in data]
            scorable = [row["result"][label] for row in data if row["scorable"]]
            timings = sorted(r["ms"] for r in result_rows)
            result[label][group] = {
                "documents": [sum(row["all_documents"] for row in result_rows), len(result_rows)],
                "evidence": [sum(row["rank"] is not None for row in scorable), len(scorable)],
                "wins": sum(
                    r["scorable"]
                    and r["result"][label]["rank"] is not None
                    and r["result"]["global"]["rank"] is None
                    for r in data
                ),
                "losses": sum(
                    r["scorable"]
                    and r["result"][label]["rank"] is None
                    and r["result"]["global"]["rank"] is not None
                    for r in data
                ),
                "p95_chroma_ms": round(timings[math.ceil(len(timings) * 0.95) - 1], 2),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    for field in ("database", "corpus", "reference", "catalog", "questions", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--baseline", type=Path, help="전체 검색을 허용했던 기준 문서 선택 코드")
    args = parser.parse_args()
    manifest = json.loads((args.database / "manifest.json").read_text())
    assert manifest["input_sha256"] == digests(args.corpus, args.reference, args.catalog)
    refs = json.loads(args.reference.read_text())
    embeddings = CachedQueries()
    store = open_database(args.database, embeddings)
    assert len(store.get(include=[])["ids"]) == manifest["chunks"]
    projects = load_projects(args.catalog)
    questions = load_questions(args.questions)
    team = score(questions, projects)
    graph_results = {}
    questions_by_id = {row["question_id"]: row for row in questions}
    search_calls = 0

    def search(question, project_ids, k):
        nonlocal search_calls
        search_calls += 1
        return search_chunks(question, project_ids, k, store)

    with (
        patch.object(graph, "get_projects", return_value=projects),
        patch.object(graph, "search_chunks", side_effect=search),
    ):
        for row in team["rows"]:
            item = questions_by_id[row["question_id"]]
            state = graph.find_projects_node({"rewritten_question": item["question"]})
            assert set(state["project_ids"]) == set(row["predicted_ids"])
            graph_results[row["question_id"]] = state
            if not row["predicted_ids"] and state.get("catalog_miss"):
                before = search_calls
                assert graph.search_node({"rewritten_question": item["question"], **state}) == {
                    "chunks": [],
                }
                assert search_calls == before
            elif 1 < len(row["predicted_ids"]) <= refs["k"]:
                before = embeddings.calls
                chunks = graph.search_node({"rewritten_question": item["question"], **state})[
                    "chunks"
                ]
                assert {doc.metadata["project_id"] for doc, _ in chunks} == set(
                    row["predicted_ids"]
                )
                assert embeddings.calls - before == 1
    actual = list(measure(refs["rows"], graph_results, store, embeddings, refs["k"], oracle=False))
    oracle = list(measure(refs["rows"], graph_results, store, embeddings, refs["k"], oracle=True))
    results = {"actual": summarize(actual), "oracle": summarize(oracle)}
    details = {"actual": actual, "oracle": oracle}
    if args.baseline:
        baseline = _load_baseline(args.baseline)
        baseline_scopes = {
            row["question_id"]: {
                "project_ids": baseline(row["question"], projects),
                "catalog_miss": False,
            }
            for row in refs["rows"]
        }
        details["baseline_selector"] = list(
            measure(refs["rows"], baseline_scopes, store, embeddings, refs["k"], oracle=False)
        )
        results["baseline_selector"] = summarize(details["baseline_selector"])
        print("baseline_selector", results["baseline_selector"], flush=True)
    print(
        "selection",
        team["correct"],
        "/",
        team["counted"],
        "graph comparisons",
        search_calls,
        flush=True,
    )
    print("actual", results["actual"], flush=True)
    print("oracle", results["oracle"], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "baseline_sha256": _sha256(args.baseline) if args.baseline else None,
                "tests": results,
                "selection": {
                    "correct": team["correct"],
                    "counted": team["counted"],
                    "graph_multi_comparisons": search_calls,
                },
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
