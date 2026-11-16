"""Replay team scoring against a fixed local index and probe scorer edge cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from bidmate.data.loader import clean_text
from bidmate.data.metadata import _to_budget, _to_date
from bidmate.retrieval import project_finder as finder
from bidmate.retrieval.vector_store import search_chunks
from bidmate.schemas import ProjectInfo
from evaluation import answer_grader, score_search
from evaluation.score_project_finder import score as document_score
from evaluation.search_b_experiments.build_local_chroma import open_database
from evaluation.search_b_experiments.nov17.cases import csv_rows
from evaluation.search_b_experiments.nov17.revisions import from_git
from evaluation.search_b_experiments.score_chroma_replay import CachedQueries
from scripts.evaluate_project_finder import load_projects


def audit_chunks(corpus, projects, questions):
    ids = {p.project_name: p.project_id for p in projects}
    chunks = defaultdict(list)
    for chunk in corpus["chunks"]:
        chunks[chunk["metadata"]["project_id"]].append(chunk)
    issues = []
    for question in questions:
        if question["question_type"] == score_search.FOLLOW_UP_QUESTION_TYPE:
            continue
        names = score_search.split_names(question.get("expected_project_names"))
        expected = [ids[name] for name in names if name in ids]
        for pid, terms in score_search.match_keywords_to_projects(question, expected).items():
            if not terms:
                continue
            raw = corpus["text"].get(pid)
            if not raw:
                status = "source_missing"
            elif score_search.keyword_in_document(
                terms,
                [
                    x["page_content"].partition("\n")[2]
                    for x in chunks[pid]
                    if not x["metadata"]["chunk_id"].endswith("-info")
                ],
            ):
                status = "in_body_chunk"
            elif score_search.keyword_in_document(terms, [raw]):
                status = "body_lost_after_chunking"
            elif score_search.keyword_in_document(
                terms,
                [
                    x["page_content"]
                    for x in chunks[pid]
                    if x["metadata"]["chunk_id"].endswith("-info")
                ],
            ):
                status = "in_metadata_only"
            else:
                status = "absent_in_snapshot"
            issues.append({"id": question["question_id"], "status": status})
    return {"counts": Counter(row["status"] for row in issues), "rows": issues}


def audit_metadata(path):
    rows = csv_rows(path)
    issuers_by_title = defaultdict(set)
    times = []
    for row in rows:
        issuers_by_title[row["사업명"].strip()].add(row["발주 기관"].strip())
        match = re.search(r"\s(\d{1,2}:\d\d)(?::\d\d)?", row["입찰 참여 마감일"])
        if match:
            times.append(match[1])
    return {
        "projects": len(rows),
        "missing_budget": sum(_to_budget(r["사업 금액"] or None) is None for r in rows),
        "missing_deadline": sum(_to_date(r["입찰 참여 마감일"] or None) is None for r in rows),
        "deadlines_with_time": len(times),
        "non_midnight_deadlines": sum(t not in {"0:00", "00:00"} for t in times),
        "shared_titles_with_different_issuers": sum(len(x) > 1 for x in issuers_by_title.values()),
    }


def controls(split):
    p = ProjectInfo("demo", "가온연구원", "시범 사업", None, None)
    q = {
        "question_type": "한 문서",
        "expected_project_names": [p.project_name],
        "answer_keywords": "3000000",
    }
    chunk = Document(
        page_content="예산 3,000,000원",
        metadata={"project_id": p.project_id, "project_name": p.project_name},
    )
    duration = chunk.model_copy(update={"page_content": "계약 기간 130일"})
    with patch.object(answer_grader, "get_projects", return_value=[p]):
        comma = answer_grader.find_failed_stage(
            q, {"project_ids": ["demo"], "chunks": [(chunk, 0.0)]}
        )
        rewrite = answer_grader.find_failed_stage(
            q, {"rewritten_question": "다른 사업 예산은?", "project_ids": [], "chunks": []}
        )
    other = ProjectInfo("demo-2", "누리연구원", "시범 사업", None, None)
    same_title = finder.find_project_ids("누리연구원 시범 사업 예산은?", [p, other])
    label_rejected = False
    try:
        document_score(
            [
                {
                    "question_id": "fixture",
                    "question_type": "한 문서",
                    "question": "누리연구원 시범 사업 예산은?",
                    "expected_project_names": other.project_name,
                }
            ],
            [p, other],
        )
    except ValueError:
        label_rejected = True
    return {
        "numeric_substring_30_in_130": score_search.keyword_in_document(["30일"], ["130일"]),
        "numeric_substring_false_top5_hit": score_search.check_content_hit(
            [(duration, 0.0)], {"demo": ["30일"]}
        ),
        "grouped_number_search_hit": score_search.check_content_hit(
            [(chunk, 0.0)], {"demo": ["3000000"]}
        ),
        "grouped_number_answer_error_diagnosis": comma,
        "wrong_rewrite_labeled_stage": rewrite,
        "same_title_lookup_finds_named_issuer": same_title == ["demo-2"],
        "same_title_label_rejected": label_rejected,
        "split_k_5_for_6": split.split_k(5, 6),
        "split_k_5_for_2": split.split_k(5, 2),
        "loader_preserves_spaced_letters": "작 성 방 법" in clean_text("작 성 방 법"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "questions", "corpus", "index", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--main-revision", default="ee3614e")
    parser.add_argument("--search-revision", default="65acb49")
    args = parser.parse_args()
    logging.getLogger("bidmate.retrieval.vector_store").setLevel(logging.ERROR)
    questions = score_search.load_questions(args.questions)
    projects = load_projects(args.catalog)
    corpus = json.loads(args.corpus.read_text())
    assert corpus["catalog_sha256"] == hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    result = {
        "metadata": audit_metadata(args.catalog),
        "chunks": audit_chunks(corpus, projects, questions),
        "replay": {},
    }
    result["no_content_routes"] = {}
    for q in questions:
        if "".join(q["question_type"].split()) != "없는내용":
            continue
        ids, stop = finder.select_project_scope(q["question"], projects)
        result["no_content_routes"][q["question_id"]] = {"selected": len(ids), "stop": stop}
    with tempfile.TemporaryDirectory(prefix="bidmate-team-audit-") as tmp:
        base = Path(tmp)
        search_source = "src/bidmate/retrieval/vector_store.py"
        main_finder = from_git(
            args.main_revision,
            "src/bidmate/retrieval/project_finder.py",
        )
        main_search = from_git(args.main_revision, search_source)
        split_search = from_git(args.search_revision, search_source)
        result["controls"] = controls(split_search)
        manifest = json.loads((args.index / "manifest.json").read_text())
        for key, path in (("corpus", args.corpus), ("projects", args.catalog)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["input_sha256"][key]:
                raise ValueError(f"Replay index and {key} do not match")
        shutil.copytree(args.index, base / "chroma")
        store = open_database(base / "chroma", CachedQueries())
        assert store._collection.count() == manifest["chunks"]
        indexed = {d["project_id"] for d in store.get(include=["metadatas"])["metadatas"]}
        expected = {
            project_id
            for row in document_score(questions, projects)["rows"]
            for project_id in row["expected_ids"]
        }
        assert expected <= indexed
        result["index"] = {"documents": len(indexed), "chunks": store._collection.count()}
        variants = [
            (
                "main",
                lambda q, p: (main_finder.find_project_ids(q, p), False),
                main_search.search_chunks,
            ),
            ("pr41", finder.select_project_scope, search_chunks),
            ("pr41_with_pr39_search", finder.select_project_scope, split_search.search_chunks),
        ]
        omit = ("rows", "unknown_project_names", "average_search_seconds", "slowest_search_seconds")
        for name, select, search in variants:
            with (
                patch.object(score_search, "select_project_scope", side_effect=select),
                patch.object(score_search, "search_chunks", side_effect=search),
            ):
                cohort = {}
                for label, oracle in (("actual", False), ("oracle", True)):
                    stats = score_search.score(
                        questions, store, projects, use_expected_projects=oracle
                    )
                    cohort[label] = {k: v for k, v in stats.items() if k not in omit}
                    rows = stats["rows"]
                    cohort[label]["scope_missing_index"] = [
                        r["question_id"]
                        for r in rows
                        if any(
                            pid not in indexed
                            for pid in r["searched_project_ids"].split("|")
                            if pid
                        )
                    ]
                    cohort[label]["content_status"] = {
                        r["question_id"]: r["content_status"] for r in rows
                    }
                result["replay"][name] = cohort
                print("replayed", name, flush=True)
        selected = [q for q in questions if q["question_id"] in {"Q007", "Q011"}]
        result["persistent_rank_check"] = {}
        for k in (5, 10, 20, 50):
            stats = score_search.score(selected, store, projects, use_expected_projects=True, k=k)
            result["persistent_rank_check"][k] = {
                row["question_id"]: {"status": row["content_status"], "rank": row["content_rank"]}
                for row in stats["rows"]
            }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
