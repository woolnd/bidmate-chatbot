"""Score the integrated selector without loading an embedding model."""

import argparse
import json
from pathlib import Path
from statistics import quantiles
from time import perf_counter_ns

from bidmate.retrieval.project_finder import _find_project_ids, find_project_ids
from evaluation.score_project_finder import score
from evaluation.search_b_experiments.nov17.cases import (
    Case,
    contrast_cases,
    csv_rows,
    evaluate,
    load_cases,
    split_requests,
)
from scripts.evaluate_project_finder import load_projects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "team", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    team = csv_rows(args.team)
    cases, folds = load_cases(args.cases, projects, team, 1611)
    questions = {row["question_id"]: row["question"] for row in team}
    cases += list(contrast_cases(cases, projects, folds))
    cases += [
        Case(questions[row["question_id"]], frozenset(row["expected_ids"]), "team", "team")
        for row in score(team, projects)["rows"]
    ]
    selectors = {"current": _find_project_ids, "candidate": find_project_ids}
    metrics = {}
    for fold in ("validation", "test", "contrast", "team"):
        subset = [c for c in cases if c.split == fold]
        predictions = [
            {key: frozenset(fn(c.question, projects)) for key, fn in selectors.items()}
            for c in subset
        ]
        metrics[fold] = evaluate(subset, predictions)
    timing = {}
    for category, subset in (
        ("test", [c for c in cases if c.split == "test"]),
        (
            "two_requests",
            [
                c
                for c in cases
                if c.split in ("test", "contrast") and len(split_requests(c.question)) == 2
            ],
        ),
    ):
        elapsed = {key: [] for key in selectors}
        for repeat in range(4):
            order = tuple(selectors) if repeat % 2 == 0 else tuple(reversed(selectors))
            for c in subset:
                for key in order:
                    start = perf_counter_ns()
                    selectors[key](c.question, projects)
                    elapsed[key].append((perf_counter_ns() - start) / 1_000_000)
        timing[category] = {
            "questions": len(subset),
            "repeats": 4,
            "p95_ms": {k: round(quantiles(v, n=20)[18], 2) for k, v in elapsed.items()},
        }
    args.output.write_text(json.dumps({"metrics": metrics, "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
