from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document

from bidmate.config import TOP_K
from bidmate.retrieval.vector_store import make_project_filter
from evaluation.score_search import check_content_hit, keyword_in_document
from evaluation.search_b_experiments.build_local_chroma import digests, open_database
from evaluation.search_b_experiments.compare_chunks import Index, select
from evaluation.search_b_experiments.score_evidence_units import team_query_rows

WINDOW = 20
RRF_CONSTANT = 60


def by_document(index, sparse, store, question, ids, chunk_indices):
    vector = store.embeddings.embed_query(question)
    rank_sources = {}
    for pid in ids:
        lexical = [i for i in sparse if index.pieces[i].metadata["project_id"] == pid]
        semantic = store.similarity_search_by_vector_with_relevance_scores(
            vector, k=WINDOW, filter=make_project_filter([pid])
        )
        dense = [chunk_indices[chunk.metadata["chunk_id"]] for chunk, _ in semantic]
        rank_sources[pid] = (lexical, dense)
    return rank_sources


def fused_chunks(index, ranks, ids, lexical_weight, dense_weight):
    fused = defaultdict(float)
    for lexical, dense in ranks.values():
        for weight, entries in ((lexical_weight, lexical), (dense_weight, dense)):
            if weight:
                for rank, i in enumerate(entries[:WINDOW], 1):
                    fused[i] += weight / (RRF_CONSTANT + rank)
    order = sorted(fused, key=lambda i: (-fused[i], i))
    return select(index, order, fused, ids, "per_document_first", TOP_K)


def main():
    parser = argparse.ArgumentParser()
    for field in ("database", "corpus", "reference", "catalog", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.database / "manifest.json").read_text())
    assert manifest["input_sha256"] == digests(args.corpus, args.reference, args.catalog)
    corpus = json.loads(args.corpus.read_text())
    refs = json.loads(args.reference.read_text())
    store = open_database(args.database)
    assert len(store.get(include=[])["ids"]) == manifest["chunks"]
    pieces = [Document.model_validate(chunk) for chunk in corpus["chunks"]]
    index = Index(pieces, omit_header=True)
    chunk_indices = {c.metadata["chunk_id"]: i for i, c in enumerate(pieces)}
    details = []
    ranks_of_missing = Counter()
    for row in team_query_rows(refs):
        question, ids, expected = row["query"], row["ids"], row["expected"]
        sparse, scores = index.rank(question, ids)
        baseline = select(index, sparse, scores, ids, "per_document_first", TOP_K)
        ranks = by_document(index, sparse, store, question, ids, chunk_indices)
        results = {"sparse": check_content_hit(baseline, expected)}
        for name, weights in (("rrf_equal", (1, 1)), ("rrf_lexical_weighted", (3, 1))):
            chunks = fused_chunks(index, ranks, ids, *weights)
            results[name] = check_content_hit(chunks, expected)
        details.append({"case": row["id"], "multi": len(ids) > 1, "ranks": results})
        if len(ids) > 1 and results["sparse"] is None:
            for pid, keywords in expected.items():
                if check_content_hit(baseline, {pid: keywords}) is not None:
                    continue
                lexical, dense = ranks[pid]
                lexical_rank = next(
                    (
                        r
                        for r, i in enumerate(lexical, 1)
                        if keyword_in_document(keywords, [pieces[i].page_content])
                    ),
                    None,
                )
                dense_rank = next(
                    (
                        r
                        for r, i in enumerate(dense, 1)
                        if keyword_in_document(keywords, [pieces[i].page_content])
                    ),
                    None,
                )
                ranks_of_missing["missing_document_slots"] += 1
                ranks_of_missing["sparse_top2"] += bool(lexical_rank and lexical_rank <= 2)
                ranks_of_missing["sparse_top5"] += bool(lexical_rank and lexical_rank <= 5)
                ranks_of_missing["sparse_top20"] += bool(lexical_rank and lexical_rank <= WINDOW)
                ranks_of_missing["dense_top20"] += dense_rank is not None
    outcomes = {}
    for name, group in (("all", details), ("multi", [r for r in details if r["multi"]])):
        outcomes[name] = {}
        for mode in ("sparse", "rrf_equal", "rrf_lexical_weighted"):
            paired = [
                (r["ranks"]["sparse"] is not None, r["ranks"][mode] is not None) for r in group
            ]
            outcomes[name][mode] = {
                "hits": sum(final for _, final in paired),
                "n": len(paired),
                "wins": sum(final and not base for base, final in paired),
                "losses": sum(base and not final for base, final in paired),
            }
    print(outcomes, dict(ranks_of_missing), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "experiment": "bidmate_rank_fusion_20261116",
                "window_per_project": WINDOW,
                "rrf_constant": RRF_CONSTANT,
                "k": TOP_K,
                "manifest": manifest,
                "scores": outcomes,
                "ranks_of_missed_multi_evidence": ranks_of_missing,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
