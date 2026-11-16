"""Score title search and a local character index for business summaries."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from bidmate.retrieval import project_finder as finder
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
    by_id = {p.project_id: p for p in projects}
    raw = {r["파일명"]: r["사업 요약"] for r in csv_rows(args.catalog)}
    team = csv_rows(args.team)
    cases, folds = load_cases(args.cases, projects, team, 1611)
    cases += list(contrast_cases(cases, projects, folds))
    questions = {row["question_id"]: row["question"] for row in team}
    cases += [
        Case(questions[r["question_id"]], frozenset(r["expected_ids"]), "team", "team")
        for r in score(team, projects)["rows"]
    ]
    cases = [c for c in cases if c.split != "train"]
    parts = {c.question: split_requests(c.question) for c in cases}
    texts = list(dict.fromkeys(v for q, ps in parts.items() for v in (q, *ps)))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    index = vectorizer.fit_transform([raw[p.project_id] for p in projects])
    sims = (vectorizer.transform(texts) @ index.T).toarray()
    best = {}
    strict = {}
    fallback_ok = {}
    named_missing = {}
    weak_title = {}
    title_terms = {p.project_id: finder._tokenize(p.project_name) for p in projects}
    for text, row in zip(texts, sims, strict=True):
        ids = finder._find_project_ids(text, projects, allow_agency_pool=False)
        strict[text] = frozenset(ids)
        agencies = finder._get_agency_matches(text, projects)
        explicit = finder.has_explicit_project_scope(text, projects)
        normalized = finder._normalize(text)
        numbered = finder.PROJECT_YEAR.search(normalized) or finder.PROJECT_ROUND.search(normalized)
        fallback_ok[text] = not (ids or agencies or explicit or numbered)
        named_missing[text] = (
            not ids and bool(agencies or explicit) and not finder._find_project_ids(text, projects)
        )
        top, second = np.argsort(-row, kind="stable")[:2]
        best[text] = (projects[top].project_id, float(row[top]), float(row[top] - row[second]))
        detail = finder.CONTENT_BOUNDARY.search(normalized)
        anchors = (
            finder._tokenize(normalized[: detail.start()]) - finder.ACTION_WORDS
            if detail
            else set()
        )
        hit = {
            pid: finder._match_project_tokens(anchors, terms) for pid, terms in title_terms.items()
        }
        weak_title[text] = any(
            sum(term in other for other in hit.values()) == 1
            for term in hit[projects[top].project_id]
        )
    current = {c.question: frozenset(finder.find_project_ids(c.question, projects)) for c in cases}

    def decide(case, cutoff, margin):
        q = case.question
        original = current[q]
        recovered = mixed = original
        ps = parts[q]

        def summary(text):
            project_id, similarity, lead = best[text]
            return (
                frozenset({project_id}) if similarity >= cutoff and lead >= margin else frozenset()
            )

        if len(ps) == 1 and not original and fallback_ok[q] and weak_title[q]:
            recovered = summary(q)
        if len(ps) == 2 and not finder.PROJECT_EXCLUSION.search(finder._normalize(q)):
            first, second = ps
            left, right = (
                strict[t] or (summary(t) if fallback_ok[t] and weak_title[t] else frozenset())
                for t in ps
            )
            if len(left) == len(right) == 1 and left != right and original <= (left | right):
                recovered = left | right
            for present, absent in ((first, second), (second, first)):
                if not named_missing[absent] or len(strict[present]) != 1:
                    continue
                pid = next(iter(strict[present]))
                title = "".join(finder._normalize(by_id[pid].project_name).split())
                text = "".join(finder._normalize(present).split())
                if (len(title) >= 4 and title in text) or strict[present] == summary(present):
                    mixed = strict[present]
        return {
            "current": original,
            "summary": recovered,
            "mixed": mixed,
            "both": mixed if mixed != original else recovered,
        }

    validation = [c for c in cases if c.split == "validation"]
    calibration = [c for c in validation if c.category == "summary_paraphrase"]
    grid = []
    for cutoff, margin in product(
        (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45), (0.05, 0.10, 0.15, 0.20)
    ):
        m = evaluate(validation, [decide(c, cutoff, margin) for c in validation])["all"]
        base, candidate = m["current"], m["summary"]
        if (
            candidate["losses"] == 0
            and candidate["questions_with_extra"] <= base["questions_with_extra"]
        ):
            predicted = [
                (doc in c.expected)
                for c in calibration
                for doc, value, lead in (best[c.question],)
                if value >= cutoff and lead >= margin
            ]
            quality = sum(1 if correct else -3 for correct in predicted)
            grid.append((candidate["exact"], quality, cutoff, margin))
    if not grid:
        raise RuntimeError("no safe validation operating point")
    _, _, cutoff, margin = max(grid)
    totals = {}
    for split in ("validation", "test", "contrast", "team"):
        cohort = [c for c in cases if c.split == split]
        totals[split] = evaluate(cohort, [decide(c, cutoff, margin) for c in cohort])
    args.output.write_text(
        json.dumps({"thresholds": [cutoff, margin], "metrics": totals}, indent=2)
    )


if __name__ == "__main__":
    main()
