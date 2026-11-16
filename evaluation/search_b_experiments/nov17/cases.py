"""Question loading, document-set metrics, and contrast cases for the 11/17 experiments."""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bidmate.retrieval import project_finder as finder

TWO_QUESTIONS = re.compile(r"[?？]\s*(?:그리고|또|아울러)\s+", re.IGNORECASE)
SEED = 1711


@dataclass(frozen=True)
class Case:
    question: str
    expected: frozenset[str]
    split: str
    category: str


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def groups_by_agency(projects, seed):
    agencies = sorted({normalized(project.agency_name) for project in projects})
    random.Random(seed).shuffle(agencies)
    a, b = round(len(agencies) * 0.55), round(len(agencies) * 0.75)
    return {
        name: "train" if i < a else "validation" if i < b else "test"
        for i, name in enumerate(agencies)
    }


def load_cases(paths, projects, team, seed):
    folds = groups_by_agency(projects, seed)
    fold_by_id = {p.project_id: folds[normalized(p.agency_name)] for p in projects}
    seen = {normalized(row["question"]) for row in team}
    cases = []
    for path in paths:
        for row in json.loads(path.read_text()):
            key = normalized(row["question"])
            expected = frozenset(row["expected_ids"])
            fold = {fold_by_id[pid] for pid in expected}
            fold.update(folds[normalized(a)] for a in row["agency_groups"])
            if len(fold) != 1 or key in seen:
                continue
            seen.add(key)
            cases.append(Case(row["question"], expected, fold.pop(), row["category"]))

    for split in ("train", "validation", "test"):
        descriptions = [
            case for case in cases if case.split == split and case.category == "summary_paraphrase"
        ]
        random.Random(seed).shuffle(descriptions)
        for first, second in zip(descriptions[::2], descriptions[1::2]):
            first_question = first.question.rstrip(" ?.")
            question = f"{first_question}? 그리고 {second.question}"
            cases.append(
                Case(question, first.expected | second.expected, split, "two_descriptions")
            )
    return cases, fold_by_id


def evaluate(cases, predictions):
    stats = {}
    for case, answers in zip(cases, predictions, strict=True):
        base = answers["current"]
        for variant, predicted in answers.items():
            for category in ("all", case.category):
                group = stats.setdefault(category, {}).setdefault(variant, Counter())
                group.update(
                    n=1,
                    exact=predicted == case.expected,
                    questions_with_extra=bool(predicted - case.expected),
                    questions_with_missing=bool(case.expected - predicted),
                    wrong_no_match=bool(predicted and not case.expected),
                    wins=base != case.expected and predicted == case.expected,
                    losses=base == case.expected and predicted != case.expected,
                )
    return stats


def split_requests(question: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in TWO_QUESTIONS.split(question))
    if len(parts) == 2 and all(parts):
        return parts
    return (question,)


def contrast_cases(cases, projects, folds):
    rng = random.Random(SEED)
    single = [c for c in cases if c.split == "test" and c.category == "summary_paraphrase"]
    rng.shuffle(single)
    for first, second in zip(single[::2], single[1::2]):
        lead = first.question.rstrip(" ?.")
        yield Case(
            f"{lead}? 또 {second.question}",
            first.expected | second.expected,
            "contrast",
            "two_new_combinations",
        )
    negatives = [c for c in cases if c.split == "test" and c.category == "wrong_business"]
    for first, second in zip(single, negatives):
        lead = first.question.rstrip(" ?.")
        yield Case(f"{lead}? 아울러 {second.question}", first.expected, "contrast", "one_unlisted")
    for project in projects:
        if folds[project.project_id] != "test":
            continue
        title = normalized(project.project_name)
        pattern, bad = (
            (finder.PROJECT_YEAR, "2099년")
            if finder.PROJECT_YEAR.search(title)
            else (finder.PROJECT_ROUND, "98차")
        )
        if len(set(pattern.findall(title))) != 1:
            continue
        invalid = pattern.sub(bad, title)
        q = f"{project.agency_name} {invalid} 예산"
        yield Case(q, frozenset(), "contrast", "unknown_edition")
