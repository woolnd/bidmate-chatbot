"""메타데이터에서 만든 초안 span으로 소형 NER를 학습하고 별도 검증한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path

from scripts.compare_search_b_models import normalize_question

LABELS = ["O", "B-AGENCY", "I-AGENCY", "B-PROJECT", "I-PROJECT"]


def align_labels(offsets: list[tuple[int, int]], spans: list[dict]) -> list[int]:
    """문자 위치의 정답을 토큰 BIO 라벨로 바꾼다. 특수 토큰은 무시한다."""
    ordered = sorted(spans, key=lambda span: span["start"])
    for index, span in enumerate(ordered):
        if span["start"] < 0 or span["end"] <= span["start"]:
            raise ValueError("잘못된 span 범위")
        if span["label"] not in {"AGENCY", "PROJECT"}:
            raise ValueError("알 수 없는 span 라벨")
        if index and span["start"] < ordered[index - 1]["end"]:
            raise ValueError("정답 span이 겹친다")
    result = []
    previous = None
    for start, end in offsets:
        if start == end:
            result.append(-100)
            previous = None
            continue
        target = next(
            (i for i, span in enumerate(ordered) if span["start"] <= start and end <= span["end"]),
            None,
        )
        if target is None:
            # 정답 경계를 가로지르는 토큰은 O 정답으로 강제하지 않는다.
            crosses = any(start < span["end"] and span["start"] < end for span in ordered)
            result.append(-100 if crosses else 0)
            previous = None
        else:
            label = ordered[target]["label"]
            result.append(LABELS.index(("I-" if previous == target else "B-") + label))
            previous = target
    return result


def decode_spans(offsets: list[tuple[int, int]], labels: list[int]) -> list[tuple[int, int, str]]:
    """토큰 BIO 예측을 문자 span으로 합친다. 단독 I는 새 span으로 읽는다."""
    spans = []
    active = False
    for (start, end), label_id in zip(offsets, labels):
        if start == end or label_id <= 0:
            active = False
            continue
        prefix, kind = LABELS[label_id].split("-", 1)
        if prefix == "I" and active and spans and spans[-1][2] == kind:
            spans[-1] = (spans[-1][0], end, kind)
        else:
            spans.append((start, end, kind))
        active = True
    return spans


def span_metrics(expected: list[list[tuple]], predicted: list[list[tuple]]) -> dict:
    """질문별 span의 라벨과 시작/끝 위치가 모두 같은 경우만 정답으로 센다."""
    if len(expected) != len(predicted):
        raise ValueError("예측과 정답의 질문 수가 다르다")
    tp = fp = fn = 0
    for truth, guess in zip(expected, predicted):
        left, right = set(truth), set(guess)
        tp += len(left & right)
        fp += len(right - left)
        fn += len(left - right)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    )


def main() -> None:
    """고정한 학습/검증 파일을 읽고 검증 F1이 높은 checkpoint를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    trainer_source = Path(__file__).read_bytes()
    trainer_sha256 = hashlib.sha256(trainer_source).hexdigest()
    normalization_path = Path(__file__).with_name("compare_search_b_models.py")
    normalization_source = normalization_path.read_bytes()
    class_weight_power = config.get("class_weight_power", 0.5)
    if not 0 <= class_weight_power <= 1:
        raise ValueError("class_weight_power는 0부터 1 사이여야 한다")
    lr_schedule = config.get("lr_schedule", "constant")
    if lr_schedule not in {"constant", "linear_decay"}:
        raise ValueError("알 수 없는 학습률 일정")
    dataset_path = Path(config["dataset"])
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("학습 데이터 해시가 다르다")
    dataset = json.loads(dataset_path.read_text())
    train = [row for row in dataset if row["split"] == "train"]
    validation = [row for row in dataset if row["split"] == "validation"]
    if {r["group"] for r in train} & {r["group"] for r in validation}:
        raise ValueError("기관 그룹이 학습과 검증에 겹친다")
    if {normalize_question(r["text"]) for r in train} & {
        normalize_question(r["text"]) for r in validation
    }:
        raise ValueError("질문이 학습과 검증에 겹친다")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (output / "trainer.py").write_bytes(trainer_source)
    (output / "comparison_helper.py").write_bytes(normalization_source)
    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        config["model"],
        revision=config["revision"],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
        use_safetensors=True,
        local_files_only=True,
        attn_implementation=config["attention_implementation"],
    )
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for layer in model.base_model.encoder.layer[-config["unfreeze_layers"] :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    device = torch.device(config["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = math.ceil(len(train) / config["batch_size"]) * config["epochs"]
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / total_steps))
        if lr_schedule == "linear_decay"
        else None
    )

    def encode(rows):
        encoded = []
        for row in rows:
            tokens = tokenizer(
                row["text"], return_offsets_mapping=True, truncation=False, add_special_tokens=True
            )
            if len(tokens["input_ids"]) > config["max_length"]:
                raise ValueError("정답 span이 잘리지 않도록 max_length를 늘려야 한다")
            offsets = tokens.pop("offset_mapping")
            labels = align_labels(offsets, row["spans"])
            encoded.append(dict(row=row, inputs=tokens, offsets=offsets, labels=labels))
        return encoded

    train_encoded, val_encoded = encode(train), encode(validation)
    counts = Counter(label for row in train_encoded for label in row["labels"] if label >= 0)
    weights = torch.tensor(
        [
            (sum(counts.values()) / max(counts[i], 1)) ** class_weight_power
            for i in range(len(LABELS))
        ],
        device=device,
    )
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    def batch(items):
        inputs = tokenizer.pad([item["inputs"] for item in items], return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        labels = torch.tensor(
            [item["labels"] + [-100] * (length - len(item["labels"])) for item in items],
            device=device,
        )
        return {key: value.to(device) for key, value in inputs.items()}, labels

    def evaluate():
        model.eval()
        truth = []
        predicted = []
        with torch.inference_mode():
            for start in range(0, len(val_encoded), config["batch_size"]):
                items = val_encoded[start : start + config["batch_size"]]
                inputs, _ = batch(items)
                guesses = model(**inputs).logits.argmax(-1).cpu().tolist()
                for item, labels in zip(items, guesses):
                    truth.append([(s["start"], s["end"], s["label"]) for s in item["row"]["spans"]])
                    predicted.append(decode_spans(item["offsets"], labels[: len(item["offsets"])]))
        return span_metrics(truth, predicted)

    best = -1.0
    history = []
    started = time.perf_counter()
    print(
        f"TRAIN {len(train)} VALIDATION {len(validation)} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}",
        flush=True,
    )
    for epoch in range(config["epochs"]):
        model.train()
        order = list(range(len(train_encoded)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), config["batch_size"]):
            items = [train_encoded[i] for i in order[start : start + config["batch_size"]]]
            inputs, labels = batch(items)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits.reshape(-1, len(LABELS)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
        metrics = evaluate()
        record = dict(epoch=epoch + 1, loss=sum(losses) / len(losses), validation=metrics)
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["f1"] > best:
            best = metrics["f1"]
            model.save_pretrained(output / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "checkpoint")
    if config["device"] == "mps":
        torch.mps.synchronize()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != trainer_sha256:
        raise ValueError("실행 중 학습 코드가 바뀌었다")
    if normalization_path.read_bytes() != normalization_source:
        raise ValueError("실행 중 정규화 코드가 바뀌었다")
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("실행 중 학습 데이터가 바뀌었다")
    summary = dict(
        history=history,
        best_validation_f1=best,
        elapsed_seconds=time.perf_counter() - started,
        train_examples=len(train),
        validation_examples=len(validation),
        seed=config["seed"],
        label_source="metadata synthetic weak labels; not human reviewed",
        label_counts=dict(counts),
        training="last encoder layers + classifier",
        final_holdout_used=False,
        script_sha256=trainer_sha256,
        class_weight_power=class_weight_power,
        lr_schedule=lr_schedule,
        normalization_sha256=hashlib.sha256(normalization_source).hexdigest(),
        versions={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
