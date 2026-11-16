from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from bidmate.retrieval import project_finder as search
from evaluation.score_project_finder import score
from scripts.evaluate_project_finder import load_projects

FIELDS = re.compile(r"\s+(?:사업\s+)?(?:예산|마감일|요구사항)(?:은|요)?[? ]*$")
NONWORD = re.compile(r"[^가-힣a-z0-9]")
EDITION_CONFLICTS = (search.PROJECT_YEAR, search.PROJECT_ROUND)
THRESHOLDS = (0.60, 0.68, 0.76, 0.84, 0.92)
MARGINS = (0.06, 0.12, 0.2)


def compact(text: str) -> str:
    return NONWORD.sub("", search._normalize(text))


@lru_cache(maxsize=4096)
def grams(text: str, mode: str) -> frozenset[str]:
    text = compact(text)
    if mode == "jamo":
        text = unicodedata.normalize("NFD", text)
    return frozenset(text[i : i + n] for n in (2, 3) for i in range(len(text) - n + 1))


def ranked_candidates(question, projects, mode):
    normalized = " ".join(search._normalize(question).split())
    matched = search._get_agency_matches(normalized, projects)
    if len(matched) != 1 or search.PROJECT_EXCLUSION.search(normalized):
        return []
    agency = " ".join(search._normalize(matched[0]).split())
    if not normalized.startswith(agency + " "):
        return []
    subject = FIELDS.sub("", normalized[len(agency) :].strip())
    if not subject or subject == normalized[len(agency) :].strip():
        return []
    if search.PROJECT_SEPARATOR.search(subject) or len(compact(subject)) < 5:
        return []
    inputs = grams(subject, mode)
    comparisons = []
    for project in projects:
        if project.agency_name != matched[0]:
            continue
        title = search._normalize(project.project_name)
        if any(
            set(pattern.findall(subject)) != set(pattern.findall(title))
            for pattern in EDITION_CONFLICTS
        ):
            continue
        target = grams(title, mode)
        if inputs and target:
            comparisons.append(
                (2 * len(inputs & target) / (len(inputs) + len(target)), project.project_id)
            )
    return sorted(comparisons, key=lambda pair: (-pair[0], pair[1]))


def chosen(ranking, limit, margin):
    if not ranking:
        return set()
    top, pid = ranking[0]
    second = ranking[1][0] if len(ranking) > 1 else 0
    return {pid} if top >= limit and top - second >= margin else set()


def edits(title):
    words = list(re.finditer(r"[가-힣]{4,}", title))
    if not words:
        return []
    longest = max(words, key=lambda m: len(m.group()))
    word = longest.group()
    middle = len(word) // 2
    positions = [(i, i + 1) for i in range(len(word) - 1) if word[i] != word[i + 1]]
    swapped = min(positions, key=lambda p: abs(p[0] - middle), default=None)
    pieces = {"one_syllable_missing": word[:middle] + word[middle + 1 :]}
    if swapped:
        i, j = swapped
        pieces["adjacent_swap"] = word[:i] + word[j] + word[i] + word[j + 1 :]
    return [
        (name, title[: longest.start()] + broken + title[longest.end() :])
        for name, broken in pieces.items()
    ]


def suite(projects, seed):
    known = {(p.agency_name, compact(p.project_name)) for p in projects}
    assert len(known) == len(projects)
    out = []
    for p in projects:
        group_hash = hashlib.sha256(f"{seed}\0{p.agency_name}".encode()).digest()[0]
        split = "dev" if group_hash < 101 else "test"
        variants = edits(p.project_name)
        if not variants:
            continue
        examples = [
            ("exact_title", p.agency_name, p.project_name, [p.project_id]),
            ("unlisted_issuer", "가상전환재단", p.project_name, []),
            ("embedded_title", p.agency_name, "신규" + p.project_name + "분리과제", []),
        ]
        year = search.PROJECT_YEAR.search(search._normalize(p.project_name))
        if year:
            replacement = search.PROJECT_YEAR.sub("2099년", p.project_name, count=1)
            examples.append(("wrong_year", p.agency_name, replacement, []))
        for kind, title in variants:
            if (p.agency_name, compact(title)) not in known:
                examples.append((kind, p.agency_name, title, [p.project_id]))
        for kind, agency, title, ids in examples:
            out.append((split, kind, f"{agency} {title} 사업 마감일?", sorted(ids), p.project_id))
    return out


def compare(entries, mode, limit, margin):
    counts = defaultdict(Counter)
    rows = []
    for row in entries:
        baseline = set(row["base"])
        result = baseline or chosen(row.get(mode, []), limit, margin)
        truth = set(row["expected"])
        previous = baseline == truth
        current = result == truth
        for key in ("all", row["kind"]):
            counts[key].update(
                {
                    "total": 1,
                    "correct": int(current),
                    "baseline": int(previous),
                    "wins": int(current and not previous),
                    "losses": int(previous and not current),
                    "spurious": len(result - truth),
                    "baseline_spurious": len(baseline - truth),
                    "missing": len(truth - result),
                }
            )
        if result != baseline:
            rows.append(
                {
                    "kind": row["kind"],
                    "question": row["question"],
                    "expected": row["expected"],
                    "before": row["base"],
                    "after": sorted(result),
                }
            )
    return dict(counts), rows


def collisions(projects, seed):
    data = []
    for p in projects:
        q = f"{p.agency_name} [{p.project_name}] 사업 예산"
        expected = {p.project_id}
        baseline = set(search.find_project_ids(q, projects))
        other = replace(p, project_id="fixture-other", agency_name="가상전환재단")
        same_prefix = replace(
            p, project_id="fixture-extension", project_name=p.project_name + " 기능 확장"
        )
        with_issuers = [*projects, other]
        with_extension = [*projects, same_prefix]
        comparisons = [
            ("different_issuer", with_issuers, expected),
            ("longer_title", with_extension, expected),
            ("flip_order", list(reversed(projects)), baseline),
        ]
        for kind, fixture, truth in comparisons:
            actual = set(search.find_project_ids(q, fixture))
            data.append(
                {
                    "kind": kind,
                    "question": q,
                    "expected": sorted(truth),
                    "actual": sorted(actual),
                    "baseline_correct": baseline == expected,
                    "correct": actual == truth,
                    "new_regression": baseline == expected and actual != truth,
                }
            )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1611)
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    with args.questions.open(encoding="utf-8-sig", newline="") as source:
        team = list(csv.DictReader(source))
    inputs = {
        "catalog": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "team": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "selector": hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest(),
    }
    rows = defaultdict(list)
    timings = []
    team_questions = {search._normalize(r["question"]) for r in team}
    for split, kind, question, expected, project_id in suite(projects, args.seed):
        if search._normalize(question) in team_questions:
            continue
        start = time.perf_counter()
        base = search.find_project_ids(question, projects)
        row = {
            "kind": kind,
            "question": question,
            "expected": expected,
            "base": base,
            "source_id": project_id,
        }
        if not base:
            for mode in ("syllable", "jamo"):
                row[mode] = ranked_candidates(question, projects, mode)
        timings.append(1000 * (time.perf_counter() - start))
        rows[split].append(row)

    results = {}
    for mode in ("syllable", "jamo"):
        options = []
        for limit in THRESHOLDS:
            for margin in MARGINS:
                metrics, _ = compare(rows["dev"], mode, limit, margin)
                m = metrics["all"]
                options.append((m["losses"] == 0, m["correct"], -m["spurious"], limit, margin))
        _, _, _, limit, margin = max(options)
        dev, _ = compare(rows["dev"], mode, limit, margin)
        test, changes = compare(rows["test"], mode, limit, margin)

        def team_selector(q, ps):
            return search.find_project_ids(q, ps) or sorted(
                chosen(ranked_candidates(q, ps, mode), limit, margin)
            )

        team_result = score(team, projects, selector=team_selector)
        results[mode] = {
            "limit": limit,
            "margin": margin,
            "dev": dev,
            "test": test,
            "team": {
                k: team_result[k]
                for k in ("correct", "counted", "extra_documents", "missing_documents")
            },
            "changes": changes,
        }
        print(
            mode,
            limit,
            margin,
            "development",
            dev["all"],
            "grouped test",
            test["all"],
            "team",
            team_result["correct"],
            "/",
            team_result["counted"],
            flush=True,
        )
    checks = collisions(projects, args.seed)
    categories = {}
    for kind in ("different_issuer", "longer_title", "flip_order"):
        same = [row for row in checks if row["kind"] == kind]
        categories[kind] = {
            "total": len(same),
            "correct": sum(row["correct"] for row in same),
            "new_regressions": sum(row["new_regression"] for row in same),
        }
    print("mutations", categories, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "inputs": inputs,
                "rows": rows,
                "results": results,
                "catalog_mutations": categories,
                "mutation_rows": checks,
                "median_all_variant_generation_ms": statistics.median(timings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
