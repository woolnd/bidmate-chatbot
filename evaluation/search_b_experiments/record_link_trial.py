from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
from splink.comparison_level_library import CustomLevel, ElseLevel
from splink.comparison_library import CustomComparison

from evaluation.search_b_experiments.text_trials import compact
from scripts.evaluate_project_finder import load_projects

SEED = 1611


def fold(agency):
    byte = hashlib.sha256(f"{SEED}\0{agency}".encode()).digest()[0]
    return "dev" if byte < 101 else "test"


def record(project, identifier):
    return {
        "unique_id": identifier,
        "agency": compact(project.agency_name),
        "title": compact(project.project_name),
        "budget": project.budget,
        "deadline": (project.deadline or "")[:10] or None,
    }


def masked_title(title):
    choices = [i for i, c in enumerate(title) if "가" <= c <= "힣" and i >= 2]
    if len(choices) < 3:
        return title
    i = choices[len(choices) // 2]
    return title[:i] + title[i + 1 :]


def comparisons(known_left, known_right, target):
    kinds = {
        "title": ("title_l = title_r", "levenshtein(title_l, title_r) <= 1", "else"),
        "budget": ("budget_l = budget_r", "else"),
        "deadline": ("deadline_l = deadline_r", "else"),
    }
    categories = {field: defaultdict(Counter) for field in kinds}
    comparable = Counter()
    for a in known_left:
        for b in known_right:
            if a["agency"] != b["agency"]:
                continue
            positive = target[a["unique_id"]] == b["unique_id"]
            comparable[positive] += 1
            for field in kinds:
                left_value, right_value = a[field], b[field]
                if left_value is None or right_value is None:
                    continue
                level = (
                    0
                    if left_value == right_value
                    else (
                        1
                        if field == "title" and Levenshtein.distance(left_value, right_value) <= 1
                        else len(kinds[field]) - 1
                    )
                )
                categories[field][positive][level] += 1
    assert comparable[True] > 0 and comparable[False] > 0
    values = []
    for field, conditions in kinds.items():
        null = CustomLevel(f"{field}_l is null or {field}_r is null").configure(is_null_level=True)
        levels = [null]
        for level, expression in enumerate(conditions):
            m_count = categories[field][True]
            u_count = categories[field][False]
            m = (m_count[level] + 1) / (sum(m_count.values()) + len(conditions))
            u = (u_count[level] + 1) / (sum(u_count.values()) + len(conditions))
            datum = ElseLevel() if expression == "else" else CustomLevel(expression)
            levels.append(datum.configure(m_probability=m, u_probability=u))
        values.append(CustomComparison(levels, output_column_name=field))
    return values, comparable[True] / sum(comparable.values())


def measure(rows, ranking, target, threshold):
    details = []
    for row in rows:
        case = row["unique_id"]
        options = ranking[case]
        chosen = None
        if options and options[0][0] >= threshold:
            if len(options) == 1 or options[0][0] - options[1][0] >= 0.03:
                chosen = options[0][1]
        details.append(
            {
                "case": case,
                "expected": target[case],
                "chosen": chosen,
                "correct": chosen == target[case],
            }
        )
    return {
        "hits": sum(r["correct"] for r in details),
        "n": len(details),
        "wrong": sum(r["chosen"] is not None and not r["correct"] for r in details),
        "undecided": sum(r["chosen"] is None for r in details),
        "details": details,
    }


def main():
    logging.getLogger("splink").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    for field in ("catalog", "original", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    old = {p.project_id: p for p in load_projects(args.original)}
    right = [record(p, f"p{i:03}") for i, p in enumerate(projects)]
    train_right = []
    groups = defaultdict(list)
    target = {}
    for i, p in enumerate(projects):
        previous = old.get(p.project_id)
        if not previous:
            continue
        agency_group = fold(p.agency_name)
        if agency_group == "dev":
            train_right.append(right[i])
        for kind in ("original", "typo_missing_one_field"):
            row = record(previous, f"q{i:03}-{kind}")
            if kind != "original":
                row["title"] = masked_title(row["title"])
                row["budget" if i % 2 == 0 else "deadline"] = None
            groups[agency_group].append(row)
            target[row["unique_id"]] = right[i]["unique_id"]
    seen_train = {r["agency"] for r in groups["dev"]}
    seen_test = {r["agency"] for r in groups["test"]}
    assert not seen_train & seen_test
    features, prior = comparisons(groups["dev"], train_right, target)
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=features,
        blocking_rules_to_generate_predictions=[block_on("agency")],
        unique_id_column_name="unique_id",
        retain_matching_columns=False,
        probability_two_random_records_match=prior,
    )
    linker = Linker(
        [pd.DataFrame(groups["dev"] + groups["test"]), pd.DataFrame(right)],
        settings,
        DuckDBAPI(),
        input_table_aliases=["queries", "catalog"],
        set_up_basic_logging=False,
    )
    predicted = linker.inference.predict().as_pandas_dataframe()
    by_query = defaultdict(list)
    for data in predicted.to_dict("records"):
        score, id_l, id_r = data["match_probability"], data["unique_id_l"], data["unique_id_r"]
        if id_l.startswith("p"):
            id_l, id_r = id_r, id_l
        by_query[id_l].append((score, id_r))
    for scores in by_query.values():
        scores.sort(key=lambda item: (-item[0], item[1]))
    choice = []
    for cut in (0.50, 0.65, 0.80, 0.90, 0.95, 0.99):
        stats = measure(groups["dev"], by_query, target, cut)
        choice.append((stats["wrong"] == 0, stats["hits"], -stats["wrong"], cut))
    threshold = max(choice)[-1]
    results = {}
    exact = defaultdict(list)
    for r in right:
        exact[(r["agency"], r["title"])].append(r["unique_id"])
    for split, rows in groups.items():
        output = {}
        for kind in ("original", "typo_missing_one_field"):
            segment = [r for r in rows if r["unique_id"].endswith(kind)]
            baseline = sum(
                exact[(r["agency"], r["title"])] == [target[r["unique_id"]]] for r in segment
            )
            output[kind] = {
                "exact_baseline": baseline,
                **measure(segment, by_query, target, threshold),
            }
        results[split] = output
    stats = {
        k: {a: {n: v for n, v in group.items() if n != "details"} for a, group in ss.items()}
        for k, ss in results.items()
    }
    print("record-linkage", threshold, stats, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "experiment": "record_linkage",
                "seed": SEED,
                "threshold": threshold,
                "train_agencies": len(seen_train),
                "test_agencies": len(seen_test),
                "versions": {lib: version(lib) for lib in ("splink", "duckdb")},
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
