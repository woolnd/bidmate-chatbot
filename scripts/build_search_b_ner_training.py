"""학습 문장에만 질문 표현과 기관 약칭을 추가한다. 검증 문장은 보존한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bidmate.retrieval.project_finder import AGENCY_ALIASES, _compact_agency_name
from scripts.compare_search_b_models import normalize_question


def augment_training(rows: list[dict], excluded_questions: set[str]) -> list[dict]:
    """기관 분리를 유지하면서 학습 문장만 늘린 새 목록을 돌려준다."""
    blocked = {normalize_question(text) for text in excluded_questions}
    blocked.update(normalize_question(row["text"]) for row in rows if row["split"] != "train")
    seen = {normalize_question(row["text"]) for row in rows}
    additional = []

    def add(text: str, spans: list[dict], group: str) -> None:
        normalized = normalize_question(text)
        if normalized in seen or normalized in blocked:
            return
        seen.add(normalized)
        additional.append(
            {
                "text": text,
                "spans": spans,
                "group": group,
                "split": "train",
                "label_status": "weak_metadata_augmented_train_only",
            }
        )

    for row in rows:
        if row["split"] != "train":
            continue
        add(row["text"] + " 제출 서류", row["spans"], row["group"])
        for span in row["spans"]:
            if span["label"] == "PROJECT":
                surface = row["text"][span["start"] : span["end"]]
                add(
                    surface + " 필수 조건",
                    [{"start": 0, "end": len(surface), "label": "PROJECT"}],
                    row["group"],
                )
    train_groups = {row["group"] for row in rows if row["split"] == "train"}
    for group in sorted(train_groups):
        for alias, targets in AGENCY_ALIASES.items():
            if _compact_agency_name(group) not in {_compact_agency_name(v) for v in targets}:
                continue
            for tail in ("지원 조건", "소요 비용"):
                add(alias + " " + tail, [{"start": 0, "end": len(alias), "label": "AGENCY"}], group)
    return rows + additional


def main() -> None:
    """원본 학습 파일과 제외할 평가 질문을 읽고 파생 데이터와 해시를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--exclude-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("기존 데이터 파일을 덮어쓸 수 없다")
    rows = json.loads(args.input.read_text())
    cases = json.loads(args.exclude_cases.read_text())
    augmented = augment_training(rows, {case["question"] for case in cases})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(augmented, ensure_ascii=False, indent=2))
    manifest = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "excluded_cases_sha256": hashlib.sha256(args.exclude_cases.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_count": len(rows),
        "augmented_count": len(augmented),
        "validation_unchanged": [r for r in rows if r["split"] == "validation"]
        == [r for r in augmented if r["split"] == "validation"],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
