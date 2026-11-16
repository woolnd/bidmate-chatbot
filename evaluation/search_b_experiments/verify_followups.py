from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZipFile

import syhwp
from langchain_core.documents import Document
from mecab import MeCab
from rapidfuzz.distance import DamerauLevenshtein

from bidmate.retrieval import project_finder
from bidmate.retrieval.project_finder import find_project_ids
from evaluation.score_project_finder import score
from evaluation.score_search import check_content_hit
from evaluation.search_b_experiments.compare_chunks import Index, select
from evaluation.search_b_experiments.text_trials import (
    bounded_catalog,
    compact,
    extract_subject,
    ranking,
    strict_single_edit_distractors,
    top,
)
from scripts.evaluate_project_finder import load_projects

FIELDS = {
    "budget": r"예산|사업비|금액|얼마|비용",
    "deadline": r"마감|언제까지|제출기한|입찰일시|마지막\s*날",
    "duration": r"사업\s*기간|계약\s*기간|용역\s*기간|수행\s*기간|납기",
    "eligibility": r"자격|참가\s*조건",
    "weight": r"배점|점수|평가\s*비율|평가\s*기준",
    "requirement": r"요구사항|과업|수행\s*범위|기능|조건",
}
LABELS = {
    "budget": r"총\s*사업비|사업\s*예산|배정\s*예산|사업\s*금액|추정\s*금액|기초\s*금액|예산",
    "deadline": (
        r"입찰\s*마감일|(?:입찰|제안|전자입찰)서?\s*(?:제출|접수)\s*(?:마감|기한|일시)"
        r"|제출\s*마감|접수\s*마감"
    ),
    "duration": r"사업\s*기간|계약\s*기간|용역\s*기간|수행\s*기간|납품\s*기한",
    "eligibility": r"(?:입찰|참가|참여)\s*자격|자격\s*요건|참가\s*조건",
    "weight": r"배점|평가\s*항목|평가\s*기준|기술\s*평가|가격\s*평가",
    "requirement": r"요구사항|과업\s*범위|수행\s*범위|요구\s*기능",
}
MONEY = r"(?P<amount>\d{1,3}(?:,\d{3})+|\d{1,16})\s*(?P<unit>억원|만원|천원|원)"
DAY = r"(?P<y>20\d{2})\s*[-./년]\s*(?P<m>\d{1,2})\s*[-./월]\s*(?P<d>\d{1,2})(?!\d)"
CUES = {
    "budget": MONEY,
    "deadline": DAY,
    "duration": r"\d+\s*(?:개월|일간|일\b|년간|주간)",
    "eligibility": r"자격|등록|허가|신고|업종|중소기업",
    "weight": r"\d+(?:\.\d+)?\s*(?:점|%|퍼센트)",
    "requirement": r"하여야|해야|지원|제공|구현|요구사항",
}
UNITS = {"억원": 100_000_000, "만원": 10_000, "천원": 1_000, "원": 1}
HANGUL = re.compile(r"[가-힣]+")
NOUN_TAGS = {"NNG", "NNP", "XR", "VV"}


def normalized(text):
    return unicodedata.normalize("NFKC", text).casefold()


def field_signals(text, fields):
    body = normalized(text.partition("\n")[2] or text)
    return sum(
        (
            2
            if re.search(rf"(?:{LABELS[field]})[^\n]{{0,80}}(?:{CUES[field]})", body)
            else 1
            if re.search(LABELS[field], body)
            else 0
        )
        for field in fields
    )


def rerank(index, order, question):
    fields = [field for field, pattern in FIELDS.items() if re.search(pattern, question)]
    per_project = Counter()
    candidates = []
    for position, number in enumerate(order):
        project = index.pieces[number].metadata["project_id"]
        per_project[project] += 1
        if per_project[project] <= 20:
            candidates.append(
                (-field_signals(index.pieces[number].page_content, fields), position, number)
            )
    return [number for _, _, number in sorted(candidates)]


def rank_trial(corpus, reference):
    pieces = [Document.model_validate(piece) for piece in corpus["chunks"]]
    index = Index(pieces, omit_header=True)
    metrics = defaultdict(Counter)
    rows = []
    for row in reference["rows"]:
        if row["scorable"] != "scorable":
            continue
        question, projects = row["question"], row["selected"]
        keywords = {pid: evidence["keywords"] for pid, evidence in row["evidence"].items()}
        order, scores = index.rank(question, projects)
        start = perf_counter_ns()
        new_order = rerank(index, order, question)
        rerank_ms = (perf_counter_ns() - start) / 1e6
        baseline = select(index, order, scores, projects, "per_document_first", 5)
        proposed = select(index, new_order, scores, projects, "per_document_first", 5)
        assert {doc.metadata["project_id"] for doc, _ in proposed} <= set(projects)
        before = check_content_hit(baseline, keywords) is not None
        after = check_content_hit(proposed, keywords) is not None
        for group in ("all", "multi" if len(projects) > 1 else "single"):
            stat = metrics[group]
            stat.update(
                n=1,
                before=before,
                after=after,
                wins=after and not before,
                losses=before and not after,
            )
        rows.append(
            {
                "question_id": row["question_id"],
                "before": before,
                "after": after,
                "rerank_ms": rerank_ms,
            }
        )
    return {"metrics": metrics, "rows": rows}


def is_lexical_word(mecab, word):
    parsed = mecab.pos(word)
    return len(parsed) == 1 and parsed[0][0] == word and parsed[0][1].split("+")[0] in NOUN_TAGS


def lexical_conflict(question, projects, mecab):
    context = extract_subject(question, projects)
    if context is None:
        return False
    agency, subject = context
    key = compact(subject)
    candidates = bounded_catalog(projects, agency, subject)
    if any(compact(doc.project_name) == key for doc in candidates):
        return False
    supplied = HANGUL.findall(subject)
    for doc in candidates:
        title = normalized(doc.project_name)
        if DamerauLevenshtein.distance(key, compact(title)) != 1:
            continue
        original = HANGUL.findall(title)
        if len(original) != len(supplied):
            continue
        differences = [
            (left, right) for left, right in zip(supplied, original, strict=True) if left != right
        ]
        if len(differences) == 1 and all(is_lexical_word(mecab, word) for word in differences[0]):
            return True
    return False


def substitution_conflict(question, projects):
    context = extract_subject(question, projects)
    if context is None:
        return False
    agency, subject = context
    key = compact(subject)
    names = {
        compact(project.project_name) for project in bounded_catalog(projects, agency, subject)
    }
    return key not in names and any(
        len(name) == len(key) and sum(a != b for a, b in zip(name, key, strict=True)) == 1
        for name in names
    )


def substitution_typos(projects):
    titles = {compact(project.project_name) for project in projects}
    for project in projects:
        match = next(
            (token for token in HANGUL.finditer(project.project_name) if len(token[0]) >= 4), None
        )
        if match is None:
            continue
        index = match.start() + 2
        old = ord(project.project_name[index]) - 0xAC00
        initial, tail = divmod(old, 588)
        replacement = chr(0xAC00 + ((initial + 1) % 19) * 588 + tail)
        title = project.project_name[:index] + replacement + project.project_name[index + 1 :]
        if compact(title) not in titles:
            yield {
                "question": f"{project.agency_name} {title} 사업 마감일?",
                "expected": [project.project_id],
            }


def typo_trial(catalog, questions, title_cases):
    from csv import DictReader

    projects = load_projects(catalog)
    mecab = MeCab()

    def guarded(question, docs):
        if lexical_conflict(question, docs, mecab):
            return []
        return find_project_ids(question, docs) or top(ranking(question, docs, "damerau"), 1)

    def strict(question, docs):
        return [] if substitution_conflict(question, docs) else guarded(question, docs)

    with questions.open(encoding="utf-8-sig", newline="") as source:
        question_rows = list(DictReader(source))
    baseline = score(question_rows, projects)
    result = score(question_rows, projects, selector=guarded)
    strict_team = score(question_rows, projects, selector=strict)
    groups = defaultdict(Counter)
    grouped_rows = {
        **title_cases["rows"],
        "different_project": strict_single_edit_distractors(projects),
        "substituted_typos": list(substitution_typos(projects)),
    }
    for label, row in ((label, row) for label, rows in grouped_rows.items() for row in rows):
        expected = row["expected"]
        question = row["question"]
        before = sorted(find_project_ids(question, projects))
        after = sorted(guarded(question, projects))
        strict_result = sorted(strict(question, projects))
        groups[label].update(
            n=1,
            before=before == expected,
            after=after == expected,
            extra_before=bool(set(before) - set(expected)),
            extra_after=bool(set(after) - set(expected)),
            strict_after=strict_result == expected,
        )
    return {
        "team": [baseline["correct"], result["correct"], baseline["counted"]],
        "strict_team": strict_team["correct"],
        "groups": groups,
    }


def candidates(text, field):
    value_pattern = MONEY if field == "budget" else DAY
    pattern = re.compile(
        rf"(?P<label>{LABELS[field]})[ ]*(?:[(:：=|·ㆍ-][ ]*)?(?:\n[ ]*)?{value_pattern}"
    )
    output = []
    for match in pattern.finditer(normalized(text)):
        value = match.groupdict()
        if field == "budget":
            amount = int(value["amount"].replace(",", "")) * UNITS[value["unit"]]
            if amount < 1000:
                continue
        else:
            try:
                amount = date(int(value["y"]), int(value["m"]), int(value["d"])).isoformat()
            except ValueError:
                continue
        output.append(
            {
                "value": amount,
                "evidence": match.group(),
                "line": text.count("\n", 0, match.start()) + 1,
            }
        )
    return output


def original_table_candidates(blob, field):
    with TemporaryDirectory(prefix="bidmate-section5-") as directory:
        path = Path(directory) / "source.hwp"
        path.write_bytes(blob)
        document = syhwp.open(str(path))
        hits = []
        for number, block in enumerate(document.blocks):
            if type(block).__name__ != "Table":
                continue
            cells = {
                (cell.row, cell.col): normalized(cell.text or "").strip() for cell in block.cells
            }
            for (row, column), label in cells.items():
                if not re.fullmatch(rf"(?:{LABELS[field]})(?:[ :：일시]+)?", label):
                    continue
                for adjacent in ((row, column + 1), (row + 1, column)):
                    value = cells.get(adjacent, "")
                    for match in candidates(f"{label}: {value}", field):
                        match.pop("line")
                        hits.append(
                            {
                                **match,
                                "table": number,
                                "label_cell": (row, column),
                                "value_cell": adjacent,
                            }
                        )
        return hits


def metadata_trial(catalog, archive, corpus):
    projects = load_projects(catalog)
    by_name = defaultdict(list)
    with ZipFile(archive) as source:
        for info in source.infolist():
            path = Path(info.filename)
            if "__MACOSX" not in path.parts and path.suffix.lower() in {".hwp", ".pdf"}:
                by_name[unicodedata.normalize("NFC", path.name)].append(info)
        linked = sum(
            len(by_name[unicodedata.normalize("NFC", doc.project_id)]) == 1 for doc in projects
        )
        same = sum(
            len(by_name[unicodedata.normalize("NFC", doc.project_id)]) == 1
            and sha256(
                source.read(by_name[unicodedata.normalize("NFC", doc.project_id)][0])
            ).hexdigest()
            == entry["sha256"]
            for doc, entry in zip(projects, corpus["raw_files"], strict=True)
            if doc.project_id == entry["id"]
        )
    rows = []
    summary = defaultdict(Counter)
    with ZipFile(archive) as source:
        table_hits = {
            (doc.project_id, field): original_table_candidates(
                source.read(by_name[unicodedata.normalize("NFC", doc.project_id)][0]), field
            )
            for doc in projects
            if doc.project_id.lower().endswith(".hwp")
            and len(by_name[unicodedata.normalize("NFC", doc.project_id)]) == 1
            for field in ("budget", "deadline")
            if getattr(doc, field) is None
        }
    for doc in projects:
        for field in ("budget", "deadline"):
            if getattr(doc, field) is not None:
                continue
            hits = candidates(corpus["text"][doc.project_id], field)
            hits.extend(table_hits.get((doc.project_id, field), ()))
            distinct = {hit["value"] for hit in hits}
            status = (
                "needs_source"
                if not distinct
                else "review_one"
                if len(distinct) == 1
                else "conflict"
            )
            rows.append(
                {"project_id": doc.project_id, "field": field, "status": status, "candidates": hits}
            )
            summary[field].update(missing=1, **{status: 1})
    return {"unique_file_link": linked, "same_content": same, "fields": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    for name in ("catalog", "questions", "archive", "corpus", "reference", "title_cases", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    reference = json.loads(args.reference.read_text())
    title_cases = json.loads(args.title_cases.read_text())
    result = {
        "ranking": rank_trial(corpus, reference),
        "typos": typo_trial(args.catalog, args.questions, title_cases),
        "metadata": metadata_trial(args.catalog, args.archive, corpus),
    }
    result["experiment"] = {
        "id": "search-b-followup-verification",
        "selector_sha256": sha256(Path(project_finder.__file__).read_bytes()).hexdigest(),
        "ranking_pool_per_document": 20,
        "top_k": 5,
        "maximum_damerau_distance": 1,
        "training": False,
    }
    result["input_sha256"] = {
        name: sha256(getattr(args, name).read_bytes()).hexdigest()
        for name in ("catalog", "questions", "archive", "corpus", "reference", "title_cases")
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "ranking": result["ranking"]["metrics"],
                "typos": result["typos"],
                "metadata": {
                    key: value for key, value in result["metadata"].items() if key != "rows"
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
