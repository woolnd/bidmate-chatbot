from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from bidmate.data.chunker import make_info_chunk, split_document
from bidmate.data.loader import load_document
from scripts.evaluate_project_finder import load_projects


def key(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build(csv_path: Path, raw_folder: Path, out: Path) -> dict:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    projects = load_projects(csv_path)
    if len(rows) != len(projects):
        raise ValueError("Some catalog records did not parse.")
    paths = defaultdict(list)
    for file in raw_folder.iterdir():
        if file.is_file() and file.suffix.lower() in {".pdf", ".hwp", ".hwpx"}:
            paths[key(file.name)].append(file)
    duplicate_ids = [
        pid for pid, n in Counter(key(p.project_id) for p in projects).items() if n > 1
    ]
    duplicate_titles = [
        title
        for title, n in Counter(key(p.project_name).casefold() for p in projects).items()
        if n > 1
    ]
    duplicate_hashes = defaultdict(list)
    files, chunks, problems = [], [], []
    raw_texts = {}
    for project in projects:
        file_matches = paths[key(project.project_id)]
        if len(file_matches) != 1:
            problems.append({"kind": "missing_or_ambiguous_file", "id": project.project_id})
            continue
        file = file_matches[0]
        checksum = digest(file)
        duplicate_hashes[checksum].append(project.project_id)
        files.append(
            {
                "id": project.project_id,
                "sha256": checksum,
                "exact_name": file.name == project.project_id,
            }
        )
        try:
            doc = load_document(replace(project, project_id=file.name), raw_folder)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            problems.append({"kind": type(error).__name__, "id": project.project_id})
            continue
        raw_texts[project.project_id] = doc.text
        if not doc.text.strip():
            problems.append({"kind": "empty_body", "id": project.project_id})
        pieces = split_document(doc, project)
        chunks.extend(piece.model_dump() for piece in [make_info_chunk(project), *pieces])
    counts = Counter(c["metadata"]["project_id"] for c in chunks)
    chunk_counts = Counter(c["metadata"]["chunk_id"] for c in chunks)
    body_hash_collisions = [same for same in duplicate_hashes.values() if len(same) > 1]
    state = {
        "catalog_rows": len(rows),
        "raw_files": sum(map(len, paths.values())),
        "matched_files": len(files),
        "documents_loaded": len(raw_texts),
        "normalized_only_names": sum(not record["exact_name"] for record in files),
        "orphans": sum(
            len(paths[name]) for name in set(paths) - {key(p.project_id) for p in projects}
        ),
        "missing_budget": sum(p.budget is None for p in projects),
        "one_won_budget": sum(p.budget == 1 for p in projects),
        "missing_deadline": sum(p.deadline is None for p in projects),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_titles": len(duplicate_titles),
        "duplicate_body_hash_groups": len(body_hash_collisions),
        "sources_with_body_chunks": sum(n > 1 for n in counts.values()),
        "chunks": len(chunks),
        "duplicate_chunk_ids": sum(n > 1 for n in chunk_counts.values()),
        "chunks_with_unknown_ids": len(set(counts) - {p.project_id for p in projects}),
        "problems": len(problems),
    }
    out.write_text(
        json.dumps(
            {
                "catalog_sha256": digest(csv_path),
                "raw_files": files,
                "audit": state,
                "duplicate_content_groups": body_hash_collisions,
                "problems": problems,
                "text": raw_texts,
                "chunks": chunks,
            },
            ensure_ascii=False,
        )
    )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build(args.catalog, args.raw_dir, args.output))
