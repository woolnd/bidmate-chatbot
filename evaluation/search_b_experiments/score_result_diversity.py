from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from bidmate.config import TOP_K
from evaluation.score_search import check_content_hit
from evaluation.search_b_experiments.compare_chunks import Index, select
from evaluation.search_b_experiments.score_evidence_units import (
    file_hash,
    sample_new_pairs,
    team_query_rows,
)
from scripts.evaluate_project_finder import load_projects

SIMILARITY = 0.65
SHINGLE = 8
POOL = 30


@lru_cache(maxsize=4096)
def shingles(text):
    body = text.partition("\n")[2] or text
    text = re.sub(r"\W", "", body.casefold())
    return {text[i : i + SHINGLE] for i in range(len(text) - SHINGLE + 1)}


def remove_near_duplicates(index, order):
    retained = []
    deferred = []
    already = defaultdict(list)
    for i in order[:POOL]:
        piece = index.pieces[i]
        group = already[piece.metadata["project_id"]]
        feature = shingles(piece.page_content)
        if feature and any(
            len(feature & earlier) / len(feature | earlier) >= SIMILARITY for earlier in group
        ):
            deferred.append(i)
        else:
            retained.append(i)
            group.append(feature)
    return retained + order[POOL:] + deferred


def equal_document_rounds(index, order, project_ids, k):
    count = len(project_ids)
    quota = k // count if 1 < count <= k else 0
    if not quota:
        return order
    reserved = set()
    taken = defaultdict(int)
    for i in order:
        project_id = index.pieces[i].metadata["project_id"]
        if taken[project_id] < quota:
            reserved.add(i)
            taken[project_id] += 1
        if len(reserved) == count * quota:
            break
    return [i for i in order if i in reserved] + [i for i in order if i not in reserved]


def evaluate(questions, index):
    rows = []
    for question in questions:
        order, scores = index.rank(question["query"], question["ids"])
        base = select(index, order, scores, question["ids"], "per_document_first", TOP_K)
        distinct = select(
            index,
            remove_near_duplicates(index, order),
            scores,
            question["ids"],
            "per_document_first",
            TOP_K,
        )
        split = [
            (index.pieces[i], scores.get(i, 0))
            for i in equal_document_rounds(index, order, question["ids"], TOP_K)[:TOP_K]
        ]
        methods = {
            "per_document_one": base,
            "near_duplicate_downrank": distinct,
            "equal_document_slots": split,
        }
        rows.append(
            {
                "id": question["id"],
                "multi": len(question["ids"]) > 1,
                "results": {
                    key: check_content_hit(chunks, question["expected"])
                    for key, chunks in methods.items()
                },
            }
        )
    summary = {}
    for group, cases in (("all", rows), ("multi", [row for row in rows if row["multi"]])):
        summary[group] = {}
        for name in ("per_document_one", "near_duplicate_downrank", "equal_document_slots"):
            paired = [
                (row["results"]["per_document_one"] is not None, row["results"][name] is not None)
                for row in cases
            ]
            summary[group][name] = {
                "correct": sum(final for _, final in paired),
                "total": len(paired),
                "wins": sum(final and not base for base, final in paired),
                "losses": sum(base and not final for base, final in paired),
            }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    for field in ("corpus", "reference", "catalog", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    data = json.loads(args.corpus.read_text())
    reference = json.loads(args.reference.read_text())
    assert file_hash(args.corpus) == reference["corpus"]
    projects = load_projects(args.catalog)
    official = list(team_query_rows(reference))
    synthetic = list(
        sample_new_pairs(projects, {pid for r in reference["rows"] for pid in r["expected"]})
    )
    selected = {pid for r in official + synthetic for pid in r["ids"]}
    index = Index(
        [
            Document.model_validate(item)
            for item in data["chunks"]
            if item["metadata"]["project_id"] in selected
        ],
        omit_header=True,
    )
    result = {
        name: evaluate(rows, index) for name, rows in (("team", official), ("synthetic", synthetic))
    }
    output = {
        "experiment": "bidmate_result_diversity_20261116",
        "config": {"shingle_size": SHINGLE, "similarity": SIMILARITY, "pool": POOL, "k": TOP_K},
        "inputs": {
            key: file_hash(getattr(args, key)) for key in ("corpus", "reference", "catalog")
        },
        "summary": {key: value[0] for key, value in result.items()},
        "details": {key: value[1] for key, value in result.items()},
    }
    print(output["summary"], flush=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
