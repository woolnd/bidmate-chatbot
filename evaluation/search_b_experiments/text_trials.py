from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import pybktree
from mecab import MeCab
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

from bidmate.retrieval import project_finder as search
from evaluation.score_project_finder import score
from scripts.evaluate_project_finder import load_projects

FIELD_END = re.compile(r"(?:\s+사업)?\s+(?:마감일|예산|요구사항)(?:은|요)?[? ]*$")
ALNUM = re.compile(r"[가-힣a-z0-9]+")
ORIGINAL_TOKENIZER = search._tokenize
DISTANCES = {
    "levenshtein": Levenshtein.distance,
    "damerau": DamerauLevenshtein.distance,
    "jaro_winkler": JaroWinkler.normalized_similarity,
}
EXPLICIT_EDITIONS = (search.PROJECT_YEAR, search.PROJECT_ROUND)


def compact(text):
    return "".join(ALNUM.findall(search._normalize(text)))


def editions(text):
    return tuple(tuple(sorted(set(pattern.findall(text)))) for pattern in EXPLICIT_EDITIONS)


def extract_subject(question, projects):
    normalized = " ".join(search._normalize(question).split())
    candidates = [
        agency
        for agency in {p.agency_name for p in projects}
        if normalized.startswith(" ".join(search._normalize(agency).split()) + " ")
    ]
    if not candidates or search.PROJECT_EXCLUSION.search(normalized):
        return None
    agency = max(candidates, key=lambda name: (len(name), name))
    subject = normalized[len(" ".join(search._normalize(agency).split())) + 1 :]
    head = FIELD_END.sub("", subject)
    if head == subject or len(compact(head)) < 7 or any(c in head for c in ("/", "&", "+")):
        return None
    return agency, head


def bounded_catalog(projects, agency, head):
    wanted = editions(head)
    return [
        p
        for p in projects
        if p.agency_name == agency and editions(search._normalize(p.project_name)) == wanted
    ]


def trial_rows(path):
    return json.loads(path.read_text())["rows"]


def ranking(question, projects, metric):
    context = extract_subject(question, projects)
    if context is None:
        return []
    agency, subject = context
    key = compact(subject)
    candidates = bounded_catalog(projects, agency, subject)
    values = []
    for project in candidates:
        title = compact(project.project_name)
        distance = DISTANCES[metric](key, title)
        values.append((distance if metric != "jaro_winkler" else 1 - distance, project.project_id))
    return sorted(values)


def top(rows, maximum):
    if not rows or rows[0][0] > maximum:
        return []
    if len(rows) > 1 and abs(rows[0][0] - rows[1][0]) < 1e-9:
        return []
    return [rows[0][1]]


def summary(records):
    total = len(records)
    before = sum(row["base"] == row["expected"] for row in records)
    after = sum(row["result"] == row["expected"] for row in records)
    return {
        "n": total,
        "base": before,
        "after": after,
        "wins": sum(r["base"] != r["expected"] and r["result"] == r["expected"] for r in records),
        "losses": sum(r["base"] == r["expected"] and r["result"] != r["expected"] for r in records),
        "extra_docs": sum(len(set(r["result"]) - set(r["expected"])) for r in records),
        "missed_docs": sum(len(set(r["expected"]) - set(r["result"])) for r in records),
    }


def assemble(rows, proposals):
    return [
        {key: row[key] for key in ("question", "kind", "base", "expected")}
        | {"result": sorted(proposed)}
        for row, proposed in zip(rows, proposals, strict=True)
    ]


def strict_single_edit_distractors(projects):
    titles = {compact(p.project_name) for p in projects}
    markers = {
        "운영": "운송",
        "개발": "개량",
        "교육": "체육",
        "정산": "결산",
        "구축": "증축",
        "관리": "감리",
    }
    result = []
    for project in projects:
        source = project.project_name
        for term, replacement in markers.items():
            if term not in source:
                continue
            name = source.replace(term, replacement, 1)
            if (
                compact(name) in titles
                or DamerauLevenshtein.distance(compact(name), compact(source)) != 1
            ):
                continue
            question = f"{project.agency_name} {name} 사업 마감일?"
            base = sorted(search.find_project_ids(question, projects))
            result.append(
                {"question": question, "kind": "판별용 다른 사업명", "base": base, "expected": []}
            )
            break
    return result


def run_fuzzy(split, hard, team, projects):
    result = {}
    for name in DISTANCES:
        values = {
            k: [ranking(r["question"], projects, name) if not r["base"] else [] for r in rows]
            for k, rows in {**split, "hard": hard}.items()
        }
        limits = (0.01, 0.02, 0.04, 0.06, 0.08) if name == "jaro_winkler" else (0, 1, 2)
        candidates = []
        for limit in limits:
            dev = assemble(
                split["dev"],
                (
                    r["base"] or top(order, limit)
                    for r, order in zip(split["dev"], values["dev"], strict=True)
                ),
            )
            s = summary(dev)
            candidates.append((s["losses"] == 0, s["after"], -s["extra_docs"], -limit))
        limit = -max(candidates)[-1]
        rows = {
            k: assemble(
                records,
                (
                    r["base"] or top(order, limit)
                    for r, order in zip(records, values[k], strict=True)
                ),
            )
            for k, records in {**split, "hard": hard}.items()
        }

        def candidate(question, docs):
            return search.find_project_ids(question, docs) or top(
                ranking(question, docs, name), limit
            )

        guard = score(team, projects, selector=candidate)
        hard_accepted = sum(bool(top(ranking(r["question"], projects, name), limit)) for r in hard)
        result[name] = {
            "maximum": limit,
            "standalone_accepts_hard": hard_accepted,
            "summary": {k: summary(v) for k, v in rows.items()},
            "team": guard["correct"],
            "rows": rows,
        }
        print(
            name,
            limit,
            result[name]["summary"],
            "team",
            guard["correct"],
            "standalone hard",
            hard_accepted,
            flush=True,
        )
    return result


def folded_cases(split, hard):
    return [r["question"] for r in split["dev"] + split["test"] + hard]


def p95_us(times):
    return round(sorted(times)[math.ceil(0.95 * len(times)) - 1] / 1000, 2)


def measure_tree(projects, questions):
    corpus = sorted({compact(p.project_name) for p in projects})
    topics = [
        compact(subject)
        for q in questions
        if (scope := extract_subject(q, projects)) is not None
        for subject in [scope[1]]
    ]
    results = {}
    for size in (100, 10000):
        titles = (
            corpus
            if size == 100
            else [
                f"{year:04d}{title}{(year * 73) % 101:03d}"
                for year in range(100)
                for title in corpus
            ]
        )
        queries = (
            topics
            if size == 100
            else [
                f"{year:04d}{topic}{(year * 73) % 101:03d}"
                for year, topic in enumerate(topics[:100])
            ]
        )
        times_tree, times_scan = [], []
        calls = 0
        tree_calls = 0

        def distance(first, second):
            nonlocal calls
            calls += 1
            return Levenshtein.distance(first, second)

        started = time.perf_counter_ns()
        tree = pybktree.BKTree(distance, titles)
        build_ms = (time.perf_counter_ns() - started) / 1e6
        for q in queries:
            started = time.perf_counter_ns()
            matches = {title for _, title in tree.find(q, 1)}
            times_tree.append(time.perf_counter_ns() - started)
            tree_calls += calls
            calls = 0
            started = time.perf_counter_ns()
            expected = {
                title for title in titles if Levenshtein.distance(q, title, score_cutoff=1) <= 1
            }
            times_scan.append(time.perf_counter_ns() - started)
            assert matches == expected
        results[str(size)] = {
            "catalog_titles": len(titles),
            "queries": len(queries),
            "identical_candidate_sets": len(queries),
            "build_ms": round(build_ms, 3),
            "tree_p95_us": p95_us(times_tree),
            "scan_p95_us": p95_us(times_scan),
            "tree_distance_calls": tree_calls,
            "scan_distance_calls": len(titles) * len(queries),
        }
    print("bk tree", results, flush=True)
    return results


def test_mecab(split, hard, team, projects):
    mecab = MeCab()
    tags = Counter()

    @lru_cache(maxsize=4096)
    def remove_particle(word):
        pieces = mecab.pos(word)
        tail = []
        for token, tag in reversed(pieces):
            if not tag.startswith("J"):
                break
            tail.append(token)
        suffix = "".join(reversed(tail))
        if suffix and word.endswith(suffix) and len(word) > len(suffix):
            tags.update(tag for _, tag in pieces if tag.startswith("J"))
            return word[: -len(suffix)]
        return word

    def tokenize(text):
        normalized = search._normalize(text)
        return ORIGINAL_TOKENIZER(ALNUM.sub(lambda m: remove_particle(m.group()), normalized))

    with patch.object(search, "_tokenize", tokenize):
        rows = {
            k: assemble(
                records, (search.find_project_ids(r["question"], projects) for r in records)
            )
            for k, records in {**split, "hard": hard}.items()
        }
        guard = score(team, projects)
    result = {
        "summary": {k: summary(v) for k, v in rows.items()},
        "team": guard["correct"],
        "tags": tags,
        "rows": rows,
    }
    print("mecab", result["summary"], "team", guard["correct"], "tags", dict(tags), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("catalog", "questions", "cases", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    split = trial_rows(args.cases)
    for group in split.values():
        for row in group:
            row["base"] = sorted(search.find_project_ids(row["question"], projects))
    hard = strict_single_edit_distractors(projects)
    with args.questions.open(encoding="utf-8-sig", newline="") as source:
        team = list(csv.DictReader(source))
    result = {
        "reference": score(team, projects)["correct"],
        "rapidfuzz": run_fuzzy(split, hard, team, projects),
        "bk_tree": measure_tree(projects, folded_cases(split, hard)),
        "mecab": test_mecab(split, hard, team, projects),
        "versions": {name: version(name) for name in ("rapidfuzz", "python-mecab-ko", "pybktree")},
        "input_hashes": {
            k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
            for k in ("catalog", "questions", "cases")
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
