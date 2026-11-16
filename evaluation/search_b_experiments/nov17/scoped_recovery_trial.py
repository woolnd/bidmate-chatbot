"""Compare scoped fallback and independent recovery of two requests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bidmate.retrieval import project_finder as finder
from evaluation.score_project_finder import score
from evaluation.search_b_experiments.nov17.cases import (
    Case,
    contrast_cases,
    csv_rows,
    evaluate,
    load_cases,
    normalized,
    split_requests,
)
from evaluation.search_b_experiments.nov17.learned_link_trial import (
    CANDIDATES,
    Encoder,
    pair_features,
    train_classifier,
)
from evaluation.search_b_experiments.nov17.revisions import from_git
from scripts.evaluate_project_finder import load_projects

SEED = 1711


def metadata_candidates(question, projects):
    text = normalized(question)
    agencies = finder._get_agency_matches(text, projects)
    if not agencies and (
        finder.ISSUER_REFERENCE.search(text) or finder._has_unlisted_issuer_prefix(text, projects)
    ):
        return []
    eligible = [p for p in projects if not agencies or p.agency_name in agencies]
    detail = finder.CONTENT_BOUNDARY.search(text)
    subject = text[: detail.start()] if detail else text
    eligible = finder._exclude_conflicting_editions(subject, eligible)
    indices = {p.project_id for p in eligible}
    return [i for i, p in enumerate(projects) if p.project_id in indices]


def compare(cases, current, rankings, projects, threshold):
    def best(question):
        ids, blocked = current[question]
        if len(ids) == 1:
            return ids
        if ids or blocked:
            return frozenset()
        position, confidence = rankings[question]
        return (
            frozenset({projects[position].project_id}) if confidence >= threshold else frozenset()
        )

    results = []
    for case in cases:
        q = case.question
        ids, _ = current[q]
        scoped = best(q) if not ids else ids
        result = {
            "current": ids,
            "scope_only": scoped,
            "parts_only": scoped,
            "scoped_parts": scoped,
            "verified_split": ids,
            "verified_split_scoped": scoped,
        }
        parts = split_requests(q)
        if len(parts) == 2 and len(ids) < 2:
            for key, picks in (
                ("parts_only", [current[part][0] for part in parts]),
                ("scoped_parts", [best(part) for part in parts]),
            ):
                confirmed = frozenset().union(*(pick for pick in picks if len(pick) == 1))
                if len(confirmed) == 2 or (not ids and len(confirmed) == 1):
                    result[key] = confirmed
            if ids:
                for key, picks in (
                    ("verified_split", [current[part][0] for part in parts]),
                    ("verified_split_scoped", [best(part) for part in parts]),
                ):
                    if all(len(pick) == 1 for pick in picks):
                        combined = picks[0] | picks[1]
                        if len(combined) == 2 and ids <= combined:
                            result[key] = combined
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "team", "model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path, nargs="+")
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--selector-revision", default="56f0cd9")
    args = parser.parse_args()
    select_scope = from_git(
        args.selector_revision, "src/bidmate/retrieval/project_finder.py"
    ).select_project_scope
    prior = json.loads(args.prior.read_text())
    threshold, _ = prior["thresholds"]["classifier"]
    torch.set_num_threads(4)
    torch.manual_seed(prior["seed"])
    team = csv_rows(args.team)
    projects = load_projects(args.catalog)
    cases, folds = load_cases(args.cases, projects, team, prior["seed"])
    official = score(team, projects)
    questions = {r["question_id"]: r["question"] for r in team}
    cases += list(contrast_cases(cases, projects, folds))
    cases += [
        Case(questions[row["question_id"]], frozenset(row["expected_ids"]), "team", "team")
        for row in official["rows"]
    ]
    texts = list(
        dict.fromkeys(
            [c.question for c in cases] + [p for c in cases for p in split_requests(c.question)]
        )
    )
    pos = {q: i for i, q in enumerate(texts)}
    raw = {r["파일명"]: r for r in csv_rows(args.catalog)}
    titles = [p.project_name for p in projects]
    profiles = [
        f"{p.project_name}. {p.agency_name}. {raw[p.project_id]['사업 요약']}" for p in projects
    ]
    encoder = Encoder(args.model, "cpu")
    title_vec = encoder.encode("passage: ", titles)
    profile_vec = encoder.encode("passage: ", profiles)
    feat = pair_features(texts, titles, encoder.encode("query: ", texts), title_vec, profile_vec)
    classifier, _ = train_classifier(
        cases, feat[[pos[c.question] for c in cases]], [p.project_id for p in projects], folds
    )
    current = {}
    rankings = {}
    for q, row in zip(texts, feat, strict=True):
        ids, blocked = select_scope(q, projects)
        current[q] = (frozenset(ids), blocked)
        index = metadata_candidates(q, projects)
        top = sorted(index, key=lambda i: (-row[i, 0], i))[:CANDIDATES]
        if top:
            values = classifier.predict_proba(row[top])[:, 1]
            first = int(np.argmax(values))
            rankings[q] = (top[first], float(values[first]))
        else:
            rankings[q] = (0, -1.0)
    metrics, deltas = {}, {}
    for fold in ("validation", "test", "contrast", "team"):
        subset = [c for c in cases if c.split == fold]
        pred = compare(subset, current, rankings, projects, threshold)
        metrics[fold] = evaluate(subset, pred)
        deltas[fold] = [
            {
                "category": case.category,
                "correct": {k: value == case.expected for k, value in results.items()},
                "missing": {k: len(case.expected - value) for k, value in results.items()},
                "extra": {k: len(value - case.expected) for k, value in results.items()},
                "partial_candidate_score": [
                    round(rankings[part][1], 4)
                    for part in split_requests(case.question)
                    if not current[part][0]
                ],
            }
            for case, results in zip(subset, pred, strict=True)
            if any(value != case.expected for value in results.values())
        ]
        print(fold, metrics[fold]["all"], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "prior_threshold": threshold,
                "model_revision": args.model.name,
                "seed": SEED,
                "metrics": metrics,
                "deltas": deltas,
                "case_counts": Counter(c.split for c in cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
