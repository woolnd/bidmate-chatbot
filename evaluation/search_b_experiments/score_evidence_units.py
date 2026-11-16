from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import syhwp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bidmate.config import CHUNK_SIZE, TOP_K
from bidmate.data.loader import clean_text
from evaluation.score_search import check_content_hit
from evaluation.search_b_experiments.compare_chunks import Index, select
from scripts.evaluate_project_finder import load_projects

SEED = 161105
CHILD_SIZE = 320
CHILD_OVERLAP = 60
HOLDOUT_PAIRS = 24


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_chunks(project, raw_dir, original_metadata):
    path = raw_dir / project.project_id
    if path.suffix.lower() != ".hwp":
        return []
    blocks = syhwp.open(str(path)).blocks
    result = []
    for block_number, block in enumerate(blocks):
        if type(block).__name__ != "Table":
            continue
        grid = defaultdict(dict)
        for cell in block.cells:
            text = clean_text((cell.text or "").replace("\n", " ")).strip()
            if text:
                grid[cell.row][cell.col] = text
        headers = grid.get(0, {})
        if not (block.n_rows > 1 and len(headers) == block.n_cols and block.n_cols > 1):
            headers = {}
        preceding = blocks[block_number - 1] if block_number else None
        heading = ""
        if preceding and type(preceding).__name__ == "Paragraph":
            heading = clean_text(preceding.text or "").strip()
            if len(heading) > 80 or "\n" in heading:
                heading = ""
        for row_number, cells in sorted(grid.items()):
            if row_number == 0 and headers:
                continue
            values = [
                f"{headers[column]}: {text}" if column in headers else text
                for column, text in sorted(cells.items())
            ]
            body = "\n".join(filter(None, (heading, " | ".join(values))))
            if len(cells) < 2 or len(body) > CHUNK_SIZE:
                continue
            metadata = {
                **original_metadata,
                "chunk_id": f"{project.project_id}-table-{block_number}-{row_number}",
            }
            result.append(
                Document(page_content=f"[{project.project_name}]\n{body}", metadata=metadata)
            )
    return result


class ChildIndex:
    def __init__(self, pieces):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP
        )
        children = []
        self.parents = []
        self.pieces = pieces
        for i, piece in enumerate(pieces):
            body = piece.page_content.partition("\n")[2] or piece.page_content
            for child in splitter.split_text(body):
                children.append(Document(page_content=child, metadata=piece.metadata))
                self.parents.append(i)
        self.index = Index(children)

    def rank(self, question, project_ids):
        ordering, child_scores = self.index.rank(question, project_ids)
        scores = {}
        for i in ordering:
            scores.setdefault(self.parents[i], child_scores.get(i, 0))
        return list(scores), scores


def sample_new_pairs(projects, seen):
    available = [p for p in projects if p.project_id not in seen and p.budget and p.deadline]
    rng = random.Random(SEED)
    rng.shuffle(available)
    available = available[: HOLDOUT_PAIRS * 2]
    assert len(available) >= 30
    for i, (first, second) in enumerate(zip(available[::2], available[1::2], strict=True)):
        field = "budget" if i % 2 == 0 else "deadline"
        phrase = (
            "배정 금액이 각각 얼마야?"
            if field == "budget"
            else "자료 받는 마지막 날이 각각 언제야?"
        )
        query = f"[{first.project_name}] / [{second.project_name}] {phrase}"
        keys = {
            p.project_id: [f"{p.budget}원" if field == "budget" else p.deadline]
            for p in (first, second)
        }
        ids = [first.project_id, second.project_id]
        yield {"id": f"synthetic-{i + 1:02}", "query": query, "ids": ids, "expected": keys}


def team_query_rows(reference):
    for row in reference["rows"]:
        if row["scorable"] == "scorable":
            yield {
                "id": row["question_id"],
                "query": row["question"],
                "ids": row["selected"],
                "expected": {k: v["keywords"] for k, v in row["evidence"].items()},
            }


def measure(questions, indexes, k):
    answers = []
    for question in questions:
        result = {}
        for method, index in indexes.items():
            order, scores = index.rank(question["query"], question["ids"])
            chunks = select(index, order, scores, question["ids"], "per_document_first", k)
            result[method] = {
                "rank": check_content_hit(chunks, question["expected"]),
                "ids": [doc.metadata["chunk_id"] for doc, _ in chunks],
            }
        answers.append(
            {"case_id": question["id"], "multi": len(question["ids"]) > 1, "runs": result}
        )
    summary = {}
    for group, items in (("all", answers), ("multi", [row for row in answers if row["multi"]])):
        summary[group] = {}
        for method in indexes:
            pairs = [
                (r["runs"]["base"]["rank"] is not None, r["runs"][method]["rank"] is not None)
                for r in items
            ]
            summary[group][method] = {
                "hits": sum(after for _, after in pairs),
                "n": len(pairs),
                "wins": sum(after and not before for before, after in pairs),
                "losses": sum(before and not after for before, after in pairs),
            }
    return summary, answers


def main():
    parser = argparse.ArgumentParser()
    for field in ("corpus", "reference", "catalog", "raw_dir", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    corpus = json.loads(args.corpus.read_text())
    assert reference["corpus"] == file_hash(args.corpus)
    projects = load_projects(args.catalog)
    records = list(team_query_rows(reference))
    seen = {pid for row in reference["rows"] for pid in row["expected"]}
    synthetic = list(sample_new_pairs(projects, seen))
    ids = {pid for row in records + synthetic for pid in row["ids"]}
    pieces = [
        Document.model_validate(c) for c in corpus["chunks"] if c["metadata"]["project_id"] in ids
    ]
    raw_hashes = {record["id"]: record["sha256"] for record in corpus["raw_files"]}
    metadata = {d.metadata["project_id"]: d.metadata for d in pieces}
    rows = []
    for project in projects:
        if project.project_id in ids and Path(project.project_id).suffix.lower() == ".hwp":
            assert file_hash(args.raw_dir / project.project_id) == raw_hashes[project.project_id]
            rows.extend(row_chunks(project, args.raw_dir, metadata[project.project_id]))
    indexes = {
        "base": Index(pieces, omit_header=True),
        "table_rows": Index(pieces + rows, omit_header=True),
        "short_hit_parent_context": ChildIndex(pieces),
    }
    team_metrics, team_details = measure(records, indexes, TOP_K)
    new_metrics, new_details = measure(synthetic, indexes, TOP_K)
    print(
        "baseline chunks",
        len(pieces),
        "table rows",
        len(rows),
        "child segments",
        len(indexes["short_hit_parent_context"].parents),
        flush=True,
    )
    print("team", team_metrics, flush=True)
    print("new catalog synthetic", new_metrics, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "experiment": "bidmate_evidence_units_20261116",
                "seed": SEED,
                "k": TOP_K,
                "child_size": CHILD_SIZE,
                "child_overlap": CHILD_OVERLAP,
                "syhwp": version("syhwp"),
                "corpus_sha256": file_hash(args.corpus),
                "questions_sha256": file_hash(args.reference),
                "chunks": len(pieces),
                "table_rows": len(rows),
                "child_segments": len(indexes["short_hit_parent_context"].parents),
                "team": team_metrics,
                "synthetic": new_metrics,
                "details": {"team": team_details, "synthetic": new_details},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
