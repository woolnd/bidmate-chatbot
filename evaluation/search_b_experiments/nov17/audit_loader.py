"""Compare main and PR #29 on the same original documents and keywords."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from bidmate.data import hwp_loader, pdf_loader
from bidmate.data.chunker import split_document
from bidmate.data.loader import clean_text
from bidmate.schemas import LoadedDocument
from evaluation import score_search
from evaluation.search_b_experiments.nov17.revisions import from_git
from scripts.evaluate_project_finder import load_projects


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--pdf":
        key, filename, revision = sys.argv[2:]
        extract = pdf_loader.extract_pdf_text
        if key == "pr29":
            extract = from_git(
                revision,
                "src/bidmate/data/pdf_loader.py",
            ).extract_pdf_text
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            body = clean_text(extract(Path(filename)))
        print(body, end="")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "catalog", "questions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--loader-revision", default="a267dfe")
    args = parser.parse_args()
    projects = load_projects(args.catalog)
    names = {p.project_name: p.project_id for p in projects}
    expectations = defaultdict(list)
    for question in score_search.load_questions(args.questions):
        if question["question_type"] == score_search.FOLLOW_UP_QUESTION_TYPE:
            continue
        project_ids = [
            names[name]
            for name in score_search.split_names(question.get("expected_project_names"))
            if name in names
        ]
        for project_id, words in score_search.match_keywords_to_projects(
            question, project_ids
        ).items():
            if words:
                expectations[project_id].append((question["question_id"], words))

    counts = defaultdict(Counter)
    diffs = []
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.archive) as z:
        root = Path(tmp)
        rev = args.loader_revision
        pr_hwp = from_git(rev, "src/bidmate/data/hwp_loader.py")
        extractors = {"main": hwp_loader.extract_hwp_text, "pr29": pr_hwp.extract_hwp_text}
        archive = {
            unicodedata.normalize("NFC", Path(f.filename).name): f
            for f in z.infolist()
            if not f.is_dir()
        }
        assert all(unicodedata.normalize("NFC", p.project_id) in archive for p in projects)
        for i, p in enumerate(projects):
            extension = Path(p.project_id).suffix.lower()
            path = root / f"doc-{i}{extension}"
            path.write_bytes(z.read(archive[unicodedata.normalize("NFC", p.project_id)]))
            observation = {}
            for key, extract in extractors.items():
                try:
                    if extension == ".pdf":
                        body = subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "evaluation.search_b_experiments.nov17.audit_loader",
                                "--pdf",
                                key,
                                str(path),
                                rev,
                            ],
                            timeout=15,
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout
                    else:
                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            body = clean_text(extract(path))
                    docs = split_document(LoadedDocument(p.project_id, body), p)
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    counts[key][f"error_{type(exc).__name__}"] += 1
                    continue
                counts[key]["documents"] += 1
                counts[key]["empty"] += not body
                counts[key]["chunks"] += len(docs)
                text = [d.page_content.split("\n", 1)[1] for d in docs]
                for question_id, words in expectations[p.project_id]:
                    raw = score_search.keyword_in_document(words, [body])
                    chunk = score_search.keyword_in_document(words, text)
                    counts[key]["keyword_targets"] += 1
                    counts[key]["keyword_found_in_extracted_text"] += raw
                    counts[key]["keyword_found_in_chunk"] += chunk
                    observation.setdefault(question_id, {})[key] = [raw, chunk]
            for question_id, results in observation.items():
                if len(results) == 2 and results["main"] != results["pr29"]:
                    diffs.append({"id": question_id, "extension": extension, "hits": results})
    args.output.write_text(json.dumps({"counts": counts, "changes": diffs}, indent=2))


if __name__ == "__main__":
    main()
