"""Audit selection failures on prior cases and controlled wording changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from bidmate.retrieval import project_finder as finder
from evaluation.score_project_finder import score
from evaluation.search_b_experiments.nov17.cases import contrast_cases, csv_rows, load_cases
from evaluation.search_b_experiments.nov17.revisions import from_git
from scripts.evaluate_project_finder import load_projects

SOURCE = "src/bidmate/retrieval/project_finder.py"


def classify(expected, chosen):
    if expected == chosen:
        return "exact"
    if not expected:
        return "false_match"
    if not chosen:
        return "empty"
    if expected.isdisjoint(chosen):
        return "wrong"
    if not expected <= chosen and not chosen <= expected:
        return "wrong_and_missing"
    return "missing" if chosen < expected else "extra"


def controls(projects):
    seed = random.Random(1717)
    titles = Counter(finder._normalize(p.project_name).strip() for p in projects)
    unique = [p for p in projects if titles[finder._normalize(p.project_name).strip()] == 1]
    seed.shuffle(unique)
    used = set()
    pairs = []
    for first in unique:
        if first.project_id in used:
            continue
        second = next(
            (p for p in unique if p.project_id not in used and p.agency_name != first.agency_name),
            None,
        )
        if second is None:
            break
        pairs.append((first, second))
        used.update((first.project_id, second.project_id))
        if len(pairs) == 12:
            break
    for first, second in pairs:
        expected = {first.project_id, second.project_id}
        a = f"{first.agency_name} {first.project_name} 용역에서 제출 서류는"
        b = f"{second.agency_name} {second.project_name} 사업의 설치 조건은"
        for word in ("그리고", "또", "아울러", "그다음", "추가로"):
            yield f"pair_{word}", f"{a}? {word} {b}?", expected, (a, b)
            yield f"pair_{word}", f"{b}? {word} {a}?", expected, (b, a)
        yield "pair_no_question_mark", f"{a}, 그리고 {b}?", expected, (a, b)
        yield "pair_no_question_mark", f"{b}, 그리고 {a}?", expected, (b, a)
        absent = "발주처: 해누리모의연구원; 수중음성나침반 잠수정 제작 사업 예산은"
        known_agency = f"발주처: {second.agency_name}; 수중음성나침반 잠수정 제작 사업 예산은"
        for category, part in (("unlisted_issuer", absent), ("unlisted_project", known_agency)):
            yield category, f"{part}?", set(), (part,)
            yield category + "_mixed", f"{a}? 또 {part}?", {first.project_id}, (a, part)
            yield category + "_mixed", f"{part}? 또 {a}?", {first.project_id}, (part, a)
    edition_cases = 0
    for p in unique:
        title = p.project_name
        for pattern, unused in ((finder.PROJECT_YEAR, "2099년"), (finder.PROJECT_ROUND, "98차")):
            if len(set(pattern.findall(title))) != 1:
                continue
            altered = pattern.sub(unused, title)
            stems = {finder._normalize(x.project_name).strip() for x in projects}
            if finder._normalize(altered).strip() in stems:
                continue
            q = f"{p.agency_name} {altered} 예산은?"
            yield "unlisted_edition", q, set(), (q,)
            edition_cases += 1
            break
        if edition_cases >= 12:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("catalog", "questions", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--label-audit", type=Path)
    parser.add_argument("--main-revision", default="ee3614e")
    parser.add_argument("--previous-revision", default="56f0cd9")
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    team = csv_rows(args.questions)
    old, groups = load_cases(args.cases, projects, team, 1611)
    contrast = list(contrast_cases(old, projects, groups))
    label_audit = json.loads(args.label_audit.read_text()) if args.label_audit else {"flags": []}
    sources = {
        c.question.rstrip(" ?."): [f"summary-{i + 1}", c.expected]
        for i, c in enumerate(c for c in old if c.category == "summary_paraphrase")
    }
    source_labels = {item["question_fingerprint"] for item in label_audit["flags"]}

    main_selector = from_git(args.main_revision, SOURCE).find_project_ids
    prev = from_git(args.previous_revision, SOURCE).find_project_ids
    modes = {"team_main": main_selector, "previous": prev, "current": finder.find_project_ids}
    overall = {}
    rows = []
    parts_separator = re.compile(r"[?？]\s*(?:그리고|또|아울러)\s+")
    for c in [*old, *contrast]:
        if c.split not in ("test", "contrast"):
            continue
        ps = tuple(parts_separator.split(c.question))
        rows.append((c.split, c.category, c.question, set(c.expected), ps))
    rows += [("new", category, q, expected, ps) for category, q, expected, ps in controls(projects)]
    official = {
        key: {k: result[k] for k in ("correct", "counted", "by_type", "skipped_followups")}
        for key, selector in modes.items()
        for result in [score(team, projects, selector=selector)]
    }
    failures = []
    seen = set()
    for split, category, q, expected, ps in rows:
        key = (split, category, q)
        if key in seen:
            continue
        seen.add(key)
        chosen = {key: set(fn(q, projects)) for key, fn in modes.items()}
        stats = {key: classify(expected, value) for key, value in chosen.items()}
        for key, result in stats.items():
            overall.setdefault(split, {}).setdefault(category, {}).setdefault(key, Counter())[
                result
            ] += 1
        if stats["current"] == "exact":
            continue
        matches = []
        for part in ps:
            stem = part.rstrip(" ?.")
            source = sources.get(stem)
            hashed = hashlib.sha256(part.encode()).hexdigest()[:12]
            if hashed not in source_labels:
                hashed = hashlib.sha256((stem + "?").encode()).hexdigest()[:12]
            matches.append(
                {
                    "source_case": source[0] if source else None,
                    "suspect_label": hashed in source_labels,
                    "expected": sorted(source[1]) if source else None,
                    "current": finder._find_project_ids(part, projects, allow_agency_pool=False),
                }
            )
        failures.append(
            {
                "id": f"A{len(failures) + 1:03d}",
                "split": split,
                "category": category,
                "question": q,
                "expected": sorted(expected),
                "selected": {key: sorted(value) for key, value in chosen.items()},
                "classification": stats,
                "parts": matches,
            }
        )
    args.output.write_text(
        json.dumps(
            {"seed": 1717, "official": official, "metrics": overall, "failures": failures},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
