"""규칙과 NER 연결 방식의 변경 전후를 같은 질문으로 비교한다."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo
from scripts.evaluate_project_finder import load_projects

Selector = Callable[[str, list[ProjectInfo]], list[str]]


def sha256(path: Path) -> str:
    """파일의 SHA-256을 돌려준다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_question(question: str) -> str:
    """중복 검사에 쓸 Unicode, 대소문자와 공백을 통일한다."""
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def validate_cases(
    cases: list[dict[str, Any]],
    known_ids: set[str],
    *,
    training_texts: set[str] | None = None,
) -> None:
    """개발 질문의 중복과 정답 ID를 검사한다. 미검수 최종 평가는 받지 않는다."""
    seen_ids: set[str] = set()
    questions: set[str] = set()
    for case in cases:
        if case.get("split") != "development":
            raise ValueError("이 실행에는 개발 질문만 넣을 수 있다")
        question = normalize_question(case["question"])
        if not question or question in questions or case["case_id"] in seen_ids:
            raise ValueError("빈 질문 또는 중복 질문/ID")
        expected = case["expected_ids"]
        if len(expected) != len(set(expected)) or not set(expected) <= known_ids:
            raise ValueError("중복 또는 목록에 없는 정답 ID")
        questions.add(question)
        seen_ids.add(case["case_id"])
    if not cases:
        raise ValueError("비교할 질문이 없다")
    if questions & {normalize_question(text) for text in training_texts or set()}:
        raise ValueError("학습/검증 문장과 같은 질문이 들어 있다")


def has_entity(question: str, entities: list[dict[str, Any]], threshold: float) -> bool:
    """유효한 기관명 또는 사업명 구간이 임계값을 넘었는지 확인한다."""
    for entity in entities:
        score = float(entity["score"])
        start, end = int(entity["start"]), int(entity["end"])
        if (
            entity["entity_group"] in {"AGENCY", "PROJECT"}
            and math.isfinite(score)
            and threshold <= score <= 1
            and 0 <= start < end <= len(question)
            and question[start:end].strip()
        ):
            return True
    return False


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """질문별 문서 집합 일치와 전체 문서 기준 P/R/F1을 집계한다."""
    tp = fp = fn = exact = duplicates = 0
    for row in rows:
        expected, predicted = set(row["expected_ids"]), set(row["predicted_ids"])
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += expected == predicted
        duplicates += len(row["predicted_ids"]) - len(predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "count": len(rows),
        "exact_set_accuracy": exact / len(rows) if rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicate_ids": duplicates,
        "p50_ms": statistics.median(row["selector_ms"] for row in rows) if rows else None,
    }


def load_selector(path: Path) -> Selector:
    """고정한 변경 전 코드를 별도 모듈로 읽는다."""
    spec = importlib.util.spec_from_file_location("search_b_baseline", path)
    if spec is None or spec.loader is None:
        raise ValueError("변경 전 코드를 읽을 수 없다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_project_ids


def main() -> None:
    """설정 파일을 읽고 추론과 문서 선택의 비교 결과를 로컬에 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 <= config["threshold"] <= 1 or config["repeats"] < 1:
        raise ValueError("임계값과 반복 횟수를 확인해야 한다")
    paths = {
        key: Path(config[key]).resolve()
        for key in ("csv", "cases", "baseline_rules", "checkpoint", "weak_dataset", "output")
    }
    checkpoint_file = paths["checkpoint"] / "model.safetensors"
    fingerprints = {
        key: sha256(paths[key]) for key in ("csv", "cases", "baseline_rules", "weak_dataset")
    }
    fingerprints["checkpoint"] = sha256(checkpoint_file)
    current_path = Path(sys.modules[find_project_ids.__module__].__file__)
    fingerprints["candidate_rules"] = sha256(current_path)
    fingerprints["runner"] = sha256(Path(__file__))
    projects = load_projects(paths["csv"])
    cases = json.loads(paths["cases"].read_text())
    weak = json.loads(paths["weak_dataset"].read_text())
    validate_cases(
        cases,
        {project.project_id for project in projects},
        training_texts={row["text"] for row in weak},
    )
    paths["output"].mkdir(parents=True, exist_ok=False)
    (paths["output"] / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (paths["output"] / "candidate_rules.py").write_bytes(current_path.read_bytes())
    (paths["output"] / "baseline_rules.py").write_bytes(paths["baseline_rules"].read_bytes())
    (paths["output"] / "runner.py").write_bytes(Path(__file__).read_bytes())
    baseline = load_selector(paths["baseline_rules"])

    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(paths["checkpoint"], local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        paths["checkpoint"], local_files_only=True, use_safetensors=True
    )
    extractor = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=config["device"],
    )
    extractor("기관명과 사업명 확인")
    random.Random(config["seed"]).shuffle(cases)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with (paths["output"] / "results.jsonl").open("w") as output:
        for index, case in enumerate(cases, 1):
            entities = extractor(case["question"])
            present = has_entity(case["question"], entities, config["threshold"])
            for name, selector in (("baseline", baseline), ("candidate", find_project_ids)):
                elapsed = []
                predictions = []
                for _ in range(config["repeats"]):
                    before = time.perf_counter()
                    predictions.append(selector(case["question"], projects))
                    elapsed.append((time.perf_counter() - before) * 1000)
                if any(value != predictions[0] for value in predictions):
                    raise ValueError("반복 실행의 반환값이 다르다")
                for method, predicted in (
                    (f"rules/{name}", predictions[0]),
                    (f"ner_gate/{name}", predictions[0] if present else []),
                ):
                    row = dict(
                        case,
                        method=method,
                        predicted_ids=predicted,
                        selector_ms=statistics.median(elapsed),
                        entity_present=present,
                    )
                    if method.startswith("ner_gate/"):
                        row["entity_spans"] = [
                            {
                                "entity_group": entity["entity_group"],
                                "score": float(entity["score"]),
                                "start": int(entity["start"]),
                                "end": int(entity["end"]),
                            }
                            for entity in entities
                        ]
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    results.append(row)
            output.flush()
            if index % 100 == 0:
                print(f"DONE {index}/{len(cases)}", flush=True)
    if sha256(current_path) != fingerprints["candidate_rules"]:
        raise ValueError("실행 중 운영 코드가 바뀌었다")
    if sha256(Path(__file__)) != fingerprints["runner"]:
        raise ValueError("실행 중 비교 코드가 바뀌었다")
    if (
        any(
            sha256(paths[key]) != fingerprints[key]
            for key in ("csv", "cases", "baseline_rules", "weak_dataset")
        )
        or sha256(checkpoint_file) != fingerprints["checkpoint"]
    ):
        raise ValueError("실행 중 입력 파일이 바뀌었다")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[row["method"] + "/all"].append(row)
        groups[row["method"] + "/" + row["category"]].append(row)
    report = {
        "experiment_id": config["experiment_id"],
        "date": datetime.now().astimezone().isoformat(),
        "input_sha256": fingerprints,
        "seed": config["seed"],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "selector": "Same fixed NER span-presence gate, original-question-preserving rule selector",
        "latency_scope": "selector only; model inference latency not compared",
        "label_status": "proposed diagnostic labels, not an independent final evaluation",
        "model_training": False,
        "summary": {key: summarize(value) for key, value in groups.items()},
    }
    (paths["output"] / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report["summary"].items() if k.endswith("/all")}, indent=2))


if __name__ == "__main__":
    main()
