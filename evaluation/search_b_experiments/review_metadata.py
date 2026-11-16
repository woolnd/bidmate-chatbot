from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from scripts.evaluate_project_finder import load_projects

BUDGET = re.compile(
    r"(?:총사업비|사업예산|배정예산|사업금액|예산액|사업비)"
    r"[:：]?([0-9][0-9,]{2,})(?:원|천원|만원|억원)"
)
DEADLINE = re.compile(
    r"(입찰참여마감(?:일시|일|시간)?|입찰서제출마감(?:일시|일)?|제안서제출마감(?:일시|일)?)"
    r"[:：]?((?:20[0-9]{2})[-./년](?:[01]?[0-9])[-./월](?:[0-3]?[0-9]))"
)


def find_values(text: str, pattern: re.Pattern) -> list[dict]:
    matches = []
    for number, line in enumerate(text.splitlines(), start=1):
        normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", line))
        for match in pattern.finditer(normalized):
            matches.append({"line": number, "excerpt": line[:300], "matched": match.group()})
    return matches


def main():
    args = argparse.ArgumentParser()
    for name in ("catalog", "original", "archive", "corpus", "output"):
        args.add_argument(name, type=Path)
    cfg = args.parse_args()
    current = load_projects(cfg.catalog)
    original = {p.project_id: p for p in load_projects(cfg.original)}
    texts = json.loads(cfg.corpus.read_text())["text"]
    before_after = Counter()
    missing = []
    for p in current:
        baseline = original.get(p.project_id)
        if not baseline:
            before_after["missing_from_original"] += 1
            continue
        for field, pattern in (("budget", BUDGET), ("deadline", DEADLINE)):
            old, new = getattr(baseline, field), getattr(p, field)
            before_after[field + "_changed"] += old != new
            before_after[field + "_populated"] += old is None and new is not None
            before_after[field + "_cleared"] += old is not None and new is None
            if new is not None:
                continue
            matches = find_values(texts[p.project_id], pattern)
            before_after[field + "_empty"] += 1
            before_after[field + "_empty_with_labeled_candidate"] += bool(matches)
            missing.append({"id": p.project_id, "field": field, "candidates": matches})
    with ZipFile(cfg.archive, metadata_encoding="utf-8") as archive:
        files = [
            Path(name).name
            for name in archive.namelist()
            if Path(name).suffix.lower() in {".hwp", ".pdf"}
            and "__MACOSX" not in Path(name).parts
            and not Path(name).name.startswith("._")
        ]
    normalized_names = {unicodedata.normalize("NFC", name) for name in files}
    before_after["archive_matches_exact"] = sum(p.project_id in files for p in current)
    before_after["archive_matches_nfc"] = sum(
        unicodedata.normalize("NFC", p.project_id) in normalized_names for p in current
    )
    cfg.output.write_text(
        json.dumps({"summary": before_after, "needs_review": missing}, ensure_ascii=False, indent=2)
    )
    print(dict(before_after))


if __name__ == "__main__":
    main()
