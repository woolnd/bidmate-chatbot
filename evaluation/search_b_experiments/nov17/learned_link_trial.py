"""Compare local E5 and a learned document selector against Search-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from evaluation.score_project_finder import score
from evaluation.search_b_experiments.nov17.cases import (
    Case,
    csv_rows,
    evaluate,
    load_cases,
    normalized,
)
from evaluation.search_b_experiments.nov17.revisions import from_git
from scripts.evaluate_project_finder import load_projects

SEED = 1611
MAX_LENGTH = 256
CANDIDATES = 5
TRAIN_NEGATIVES = 10
VARIANTS = ("current", "e5", "classifier")


def trigrams(text: str) -> set[str]:
    compact = "".join(re.findall(r"[가-힣a-z0-9]+", normalized(text)))
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


class Encoder:
    def __init__(self, model, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(model, local_files_only=True, attn_implementation="eager")
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def encode(self, prefix, texts, batch_size=24):
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                [prefix + text for text in texts[i : i + batch_size]],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"][..., None]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(vectors)


def pair_features(questions, titles, query_vectors, title_vectors, profile_vectors):
    title_scores = query_vectors @ title_vectors.T
    relevance = query_vectors @ profile_vectors.T
    title_grams = [trigrams(title) for title in titles]
    overlaps = np.zeros_like(relevance)
    for i, question in enumerate(questions):
        query_grams = trigrams(question)
        for j, grams in enumerate(title_grams):
            overlaps[i, j] = len(query_grams & grams) / max(1, len(query_grams | grams))
    return np.stack(
        [relevance, title_scores, overlaps, relevance - relevance.max(axis=1, keepdims=True)],
        axis=2,
    )


def train_classifier(cases, features, doc_ids, fold_by_id):
    group = [i for i, doc_id in enumerate(doc_ids) if fold_by_id[doc_id] == "train"]
    vectors, labels = [], []
    for case, row in zip(cases, features, strict=True):
        if case.split != "train":
            continue
        positives = [i for i in group if doc_ids[i] in case.expected]
        negatives = [
            i for i in sorted(group, key=lambda j: -row[j, 0]) if doc_ids[i] not in case.expected
        ][:TRAIN_NEGATIVES]
        vectors.extend(row[i] for i in positives + negatives)
        labels.extend([1] * len(positives) + [0] * len(negatives))
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced"))
    classifier.fit(vectors, labels)
    return classifier, Counter(labels)


def predict(case_scores, doc_ids, baseline, blocked, threshold, margin):
    if baseline or blocked:
        return baseline
    candidates = np.argsort(-case_scores, kind="stable")[:CANDIDATES]
    cutoff = max(threshold, case_scores[candidates[0]] - margin)
    return frozenset(doc_ids[i] for i in candidates if case_scores[i] >= cutoff)


def tune(cases, scores, doc_ids, baselines, blocked):
    best = None
    for threshold in [*np.linspace(0.1, 0.995, 91), 1.01]:
        for margin in (0.0, 0.01, 0.03, 0.08, 0.15, 1.0):
            result = []
            for case, row, base, miss in zip(cases, scores, baselines, blocked, strict=True):
                selected = predict(row, doc_ids, base, miss, threshold, margin)
                result.append((case.expected, base, selected))
            new_extras = sum(bool((p - y) - b) for y, b, p in result)
            exact = sum(y == p for y, _, p in result)
            total_documents = sum(len(p) for _, _, p in result)
            rank = (new_extras == 0, exact, -new_extras, -total_documents, threshold, -margin)
            if best is None or rank > best:
                best = rank
    return float(best[-2]), float(-best[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--team", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path, nargs="+")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--selector-revision", default="56f0cd9")
    args = parser.parse_args()
    select_scope = from_git(
        args.selector_revision, "src/bidmate/retrieval/project_finder.py"
    ).select_project_scope
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    projects = load_projects(args.catalog)
    raw_catalog = {r["파일명"]: r for r in csv_rows(args.catalog)}
    team = csv_rows(args.team)
    team_result = score(team, projects)
    cases, fold_by_id = load_cases(args.cases, projects, team, args.seed)
    by_question_id = {r["question_id"]: r["question"] for r in team}
    cases += [
        Case(by_question_id[r["question_id"]], frozenset(r["expected_ids"]), "team", "team")
        for r in team_result["rows"]
    ]
    doc_ids = [p.project_id for p in projects]
    texts = [case.question for case in cases]
    titles = [p.project_name for p in projects]
    profiles = [
        f"{p.project_name}. {p.agency_name}. {raw_catalog[p.project_id]['사업 요약']}"
        for p in projects
    ]
    print("folds", dict(Counter(case.split for case in cases)), flush=True)
    start = perf_counter()
    encoder = Encoder(args.model, args.device)
    title_vectors = encoder.encode("passage: ", titles)
    profile_vectors = encoder.encode("passage: ", profiles)
    print("catalog vectors ready", round(perf_counter() - start, 2), flush=True)
    query_vectors = encoder.encode("query: ", texts)
    features = pair_features(texts, titles, query_vectors, title_vectors, profile_vectors)
    classifier, labels = train_classifier(cases, features, doc_ids, fold_by_id)
    classifier_scores = classifier.predict_proba(features.reshape(-1, features.shape[2]))[:, 1]
    e5_scores = features[:, :, 0]
    candidate_indices = np.argsort(-e5_scores, axis=1, kind="stable")[:, :CANDIDATES]
    candidates = np.zeros(e5_scores.shape, dtype=bool)
    np.put_along_axis(candidates, candidate_indices, True, axis=1)
    scores = {
        "e5": e5_scores,
        "classifier": np.where(candidates, classifier_scores.reshape(features.shape[:2]), -np.inf),
    }
    scopes = [select_scope(case.question, projects) for case in cases]
    baselines = [frozenset(ids) for ids, _ in scopes]
    blocked = [miss for _, miss in scopes]
    val = [i for i, case in enumerate(cases) if case.split == "validation"]
    parameters = {
        name: tune(
            [cases[i] for i in val],
            matrix[val],
            doc_ids,
            [baselines[i] for i in val],
            [blocked[i] for i in val],
        )
        for name, matrix in scores.items()
    }
    predictions = []
    for i, base in enumerate(baselines):
        predictions.append(
            {"current": base}
            | {
                name: predict(matrix[i], doc_ids, base, blocked[i], *parameters[name])
                for name, matrix in scores.items()
            }
        )
    report = {}
    for split in ("train", "validation", "test", "team"):
        idx = [i for i, case in enumerate(cases) if case.split == split]
        report[split] = evaluate([cases[i] for i in idx], [predictions[i] for i in idx])
        if split in ("test", "team"):
            print(split, report[split]["all"], flush=True)
    retrieval = {}
    for i, case in enumerate(cases):
        if case.split != "test" or not case.expected:
            continue
        group = retrieval.setdefault(case.category, Counter())
        top = [doc_ids[j] for j in candidate_indices[i]]
        group.update(n=1, all_top5=case.expected <= set(top), top1=case.expected == {top[0]})

    latency = {"current": [], "e5": [], "classifier": []}
    for i, case in enumerate(cases):
        if case.split != "test" or case.category not in {"summary_paraphrase", "two_descriptions"}:
            continue
        start = perf_counter()
        ids, miss = select_scope(case.question, projects)
        current_ms = (perf_counter() - start) * 1000
        latency["current"].append(current_ms)
        if ids or miss:
            latency["e5"].append(current_ms)
            latency["classifier"].append(current_ms)
            continue
        start = perf_counter()
        query = encoder.encode("query: ", [case.question])
        row = (query @ profile_vectors.T)[0]
        predict(row, doc_ids, frozenset(), False, *parameters["e5"])
        e5_ms = current_ms + (perf_counter() - start) * 1000
        latency["e5"].append(e5_ms)
        start = perf_counter()
        pair = pair_features([case.question], titles, query, title_vectors, profile_vectors)[0]
        out = np.full(len(projects), -np.inf)
        top = np.argsort(-row, kind="stable")[:CANDIDATES]
        out[top] = classifier.predict_proba(pair[top])[:, 1]
        predict(out, doc_ids, frozenset(), False, *parameters["classifier"])
        latency["classifier"].append(e5_ms + (perf_counter() - start) * 1000)
    data = {
        "model_revision": args.model.name,
        "model": "intfloat/multilingual-e5-small",
        "seed": args.seed,
        "device": args.device,
        "max_length": MAX_LENGTH,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": [
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.catalog, args.team, *args.cases)
        ],
        "thresholds": parameters,
        "training_pairs": labels,
        "candidate_recall_test": retrieval,
        "latency_p95_ms": {k: round(float(np.quantile(v, 0.95)), 2) for k, v in latency.items()},
        "metrics": report,
    }
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print("saved aggregate", flush=True)


if __name__ == "__main__":
    main()
