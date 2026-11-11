"""Search-B A/B 실험 및 오검색·검색 누락 자동 평가.

A: 현재 운영 중인 find_project_ids 사용
B: 기관명 식별 + 기관 후보 확장 + 사업명 일치도 평가

운영 코드(project_finder.py)는 수정하지 않는다.
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from bidmate.retrieval.project_finder import find_project_ids  # noqa: E402

try:
    from scripts.evaluate_project_finder import load_projects  # noqa: E402
except ImportError:
    from evaluate_project_finder import load_projects  # noqa: E402


CSV_PATH = ROOT_DIR.parent / "data_list.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "search_b_experiment"

# 검색에 사용되더라도 사업 식별력이 낮은 일반 단어
STOP_WORDS = {
    "사업", "사업명", "사업비", "사업금액", "기간", "예산",
    "마감일", "계약", "방식", "시스템", "용역", "운영",
    "구축", "및", "등", "관련", "대한", "통한", "위한",
    "긴급", "재공고", "입찰", "입찰공고", "협상",
    "제안서", "제출", "안내", "공고", "정보",
}

# 별칭은 현재 데이터 100건에서 유일한 기관을 가리키는지 확인하면서 사용한다.
AGENCY_ALIASES = {
    "가스공사": "한국가스공사",
    "철도": "한국철도공사",
    "국민연금": "국민연금공단",
    "고려대": "고려대학교",
    "수자원공사": "한국수자원공사",
    "한의학연구원": "한국한의학연구원",
}


# 정답은 검색 결과를 보기 전에 고정한다.
# expected_rules에 여러 항목이 있으면 복수 정답을 의미한다.
# project_contains는 사업명에 포함되어야 하는 구체적인 문자열이다.
BENCHMARK = [
    {
        "id": "Q01",
        "question": "철도 예약발매 시스템 개량",
        "expected_rules": [
            {"agency_contains": "한국철도공사", "project_contains": "예약발매시스템 개량"}
        ],
        "reason": "기관 별칭 및 복합어 띄어쓰기",
    },
    {
        "id": "Q02",
        "question": "예약발매시스템 개량",
        "expected_rules": [
            {"agency_contains": "한국철도공사", "project_contains": "예약발매시스템 개량"}
        ],
        "reason": "사업명 중심 검색",
    },
    {
        "id": "Q03",
        "question": "한국가스공사 차세대 통합정보시스템 ERP 구축",
        "expected_rules": [
            {"agency_contains": "한국가스공사", "project_contains": "ERP"}
        ],
        "reason": "정식 기관명 및 사업 식별",
    },
    {
        "id": "Q04",
        "question": "가스공사 ERP 구축",
        "expected_rules": [
            {"agency_contains": "한국가스공사", "project_contains": "ERP"}
        ],
        "reason": "기관 약칭",
    },
    {
        "id": "Q05",
        "question": "인천공항운영서비스 차세대 ERP 시스템 구축",
        "expected_rules": [
            {"agency_contains": "인천공항운영서비스", "project_contains": "ERP"}
        ],
        "reason": "기관 불일치 오검색 차단",
    },
    {
        "id": "Q06",
        "question": "통합정보시스템 고도화 용역",
        "expected_rules": [
            {
                "agency_contains": "국가과학기술지식정보서비스",
                "project_contains": "통합정보시스템 고도화 용역",
            },
            {
                "agency_contains": "한국한의학연구원",
                "project_contains": "통합정보시스템 고도화 용역",
            },
        ],
        "reason": "동일·유사 사업명 복수 정답",
    },
    {
        "id": "Q07",
        "question": "한국한의학연구원 통합정보시스템 고도화",
        "expected_rules": [
            {
                "agency_contains": "한국한의학연구원",
                "project_contains": "통합정보시스템 고도화",
            }
        ],
        "reason": "기관명으로 유사 사업 구분",
    },
    {
        "id": "Q08",
        "question": "의료기기산업 종합정보시스템 기능개선",
        "expected_rules": [
            {
                "agency_contains": "한국보건산업진흥원",
                "project_contains": "의료기기산업 종합정보시스템",
            },
            {
                "agency_contains": "BioIN",
                "project_contains": "의료기기산업 종합정보시스템",
            },
        ],
        "reason": "유사 사업명 복수 후보",
    },
    {
        "id": "Q09",
        "question": "BioIN 의료기기산업 종합정보시스템 기능개선",
        "expected_rules": [
            {
                "agency_contains": "BioIN",
                "project_contains": "의료기기산업 종합정보시스템",
            }
        ],
        "reason": "기관명으로 유사 사업 구분",
    },
    {
        "id": "Q10",
        "question": "한국수자원공사 시스템 구축",
        "evaluation_type": "agency_pool",
        "expected_agency": "한국수자원공사",
        "reason": "사업명 불명확 시 기관 후보 확장",
    },
    {
        "id": "Q11",
        "question": "광주과학기술원 연구비관리시스템 연계 모듈 변경",
        "expected_rules": [
            {
                "agency_contains": "광주과학기술원",
                "project_contains": "연계 모듈 변경",
            }
        ],
        "reason": "사업 핵심 표현 일치",
    },
    {
        "id": "Q12",
        "question": "고려대학교 포털 학사정보시스템",
        "expected_rules": [
            {
                "agency_contains": "고려대학교",
                "project_contains": "포털",
            }
        ],
        "reason": "기관명 및 포털 표현",
    },
    {
        "id": "Q13",
        "question": "삼성전자 스마트폰 개발 사업",
        "expected_rules": [],
        "reason": "데이터에 정답이 없는 음성 사례",
    },
    {
        "id": "Q14",
        "question": "국민연금공단 이러닝 사업 기간은?",
        "expected_rules": [
            {
                "agency_contains": "국민연금공단",
                "project_contains": "이러닝",
            }
        ],
        "reason": "실제 사용자 질문 - 기간",
    },
    {
        "id": "Q15",
        "question": "국민연금공단 이러닝 사업 예산은?",
        "expected_rules": [
            {
                "agency_contains": "국민연금공단",
                "project_contains": "이러닝",
            }
        ],
        "reason": "실제 사용자 질문 - 예산",
    },
    {
        "id": "Q16",
        "question": "예약발매시스템 개량 사업비는?",
        "expected_rules": [
            {
                "agency_contains": "한국철도공사",
                "project_contains": "예약발매시스템 개량",
            }
        ],
        "reason": "실제 사용자 질문 - 사업비",
    },
    {
        "id": "Q17",
        "question": "모바일오피스 고도화 사업 예산은?",
        "expected_rules": [
            {
                "agency_contains": "한국철도공사",
                "project_contains": "모바일오피스",
            }
        ],
        "reason": "실제 사용자 질문 - 예산",
    },
    {
        "id": "Q18",
        "question": "운행정보기록 자동분석시스템 계약 방식은?",
        "expected_rules": [
            {
                "agency_contains": "한국철도공사",
                "project_contains": "운행정보기록 자동분석시스템",
            }
        ],
        "reason": "실제 사용자 질문 - 계약 방식",
    },
    {
        "id": "Q19",
        "question": "고려대 포털 학사 시스템 예산과 마감일은?",
        "expected_rules": [
            {
                "agency_contains": "고려대학교",
                "project_contains": "포털",
            }
        ],
        "reason": "실제 사용자 질문 - 예산 및 마감일",
    },
    {
        "id": "Q20",
        "question": "삼성전자 스마트폰 개발 사업 예산",
        "expected_rules": [],
        "reason": "실제 사용자 질문 - 정답 없는 질문",
    },
]


def compact(value: str) -> str:
    """유니코드와 법인 표기를 정규화한 비교 문자열."""
    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    ).casefold().strip()

    # 법인 표기는 기관 식별에 영향을 주지 않도록 제거한다.
    # NFKC 정규화 후 ㈜는 (주), ㈔는 (사)로 통일된다.
    legal_suffixes = (
        "(주)",
        "(사)",
        "주식회사",
        "유한회사",
    )

    for suffix in legal_suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
            break

    # 공백·괄호·기호 제거
    return re.sub(r"[^가-힣a-z0-9]", "", normalized)



def tokenize(value: str) -> set[str]:
    """한국어·영문·숫자 기반 핵심 토큰."""
    tokens = re.findall(r"[가-힣a-z0-9]+", str(value).casefold())
    return {
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    }


def get_agency_matches(question: str, projects: list[Any]) -> list[str]:
    """질문에 명시된 기관을 데이터의 실제 기관명과 연결한다."""
    q = compact(question)
    agencies = sorted(
        {p.agency_name for p in projects if p.agency_name},
        key=lambda value: len(compact(value)),
        reverse=True,
    )

    matched: list[str] = []

    # 데이터에 존재하는 정식 기관명 우선 탐색
    for agency in agencies:
        normalized = compact(agency)
        if len(normalized) >= 4 and normalized in q:
            matched.append(agency)

    # 별칭은 실제 데이터에 대상 기관이 존재할 때만 적용
    for alias, canonical in AGENCY_ALIASES.items():
        alias_compact = compact(alias)
        canonical_compact = compact(canonical)

        if alias_compact not in q:
            continue

        target = next(
            (
                agency
                for agency in agencies
                if compact(agency) == canonical_compact
                or canonical_compact in compact(agency)
            ),
            None,
        )
        if target and target not in matched:
            matched.append(target)

    return matched


def project_match_score(question: str, project_name: str) -> float:
    """질문 핵심 토큰이 사업명에 얼마나 포함되는지 계산한다."""
    question_tokens = tokenize(question)
    project_tokens = tokenize(project_name)

    if not question_tokens:
        return 0.0

    matched = 0
    project_compact = compact(project_name)

    for token in question_tokens:
        if token in project_tokens or token in project_compact:
            matched += 1
        elif any(token in candidate or candidate in token for candidate in project_tokens):
            matched += 1

    return matched / len(question_tokens)


def run_experiment_b(
    question: str,
    projects: list[Any],
    baseline_ids: list[str],
) -> tuple[list[str], str]:
    """기관 후보 확장 실험.

    - 기관을 찾지 못하면 A 결과를 유지한다.
    - 기관을 찾으면 다른 기관의 문서를 후보에서 제외한다.
    - 사업명 근거가 충분하면 관련 사업만 반환한다.
    - 사업명 근거가 약하면 해당 기관의 전체 사업을 후보로 반환한다.
    """
    agencies = get_agency_matches(question, projects)

    if not agencies:
        return baseline_ids, "NO_AGENCY_USE_A"

    agency_projects = [
        project
        for project in projects
        if project.agency_name in agencies
    ]

    if not agency_projects:
        return [], "AGENCY_NOT_FOUND"

    # 질문에서 기관명 표현을 제거해 사업명 근거를 평가한다.
    project_query = question
    for agency in agencies:
        project_query = re.sub(
            re.escape(agency),
            " ",
            project_query,
            flags=re.IGNORECASE,
        )

    for alias, canonical in AGENCY_ALIASES.items():
        if canonical in agencies:
            project_query = re.sub(
                re.escape(alias),
                " ",
                project_query,
                flags=re.IGNORECASE,
            )

    scores = [
        (project, project_match_score(project_query, project.project_name))
        for project in agency_projects
    ]

    # 식별력 있는 사업 표현이 확인된 경우 관련 후보만 반환
    matched = [
        (project, score)
        for project, score in scores
        if score >= 0.50
    ]

    if matched:
        matched.sort(key=lambda item: item[1], reverse=True)
        return [project.project_id for project, _ in matched], "AGENCY_AND_PROJECT"

    # 기관은 명확하지만 사업명이 불명확한 경우 전체 기관 후보 반환
    return (
        [project.project_id for project in agency_projects],
        "AGENCY_CANDIDATE_EXPANSION",
    )


def resolve_expected_ids(
    case: dict[str, Any],
    projects: list[Any],
) -> tuple[set[str], str]:
    """사전에 정의한 정답 규칙을 실제 원본 project_id로 변환한다."""
    expected: set[str] = set()
    unresolved: list[str] = []

    for rule in case.get("expected_rules", []):
        agency_key = compact(rule["agency_contains"])
        project_key = compact(rule["project_contains"])

        matches = [
            project
            for project in projects
            if agency_key in compact(project.agency_name)
            and project_key in compact(project.project_name)
        ]

        if not matches:
            unresolved.append(
                f"{rule['agency_contains']} / {rule['project_contains']}"
            )

        expected.update(project.project_id for project in matches)

    if unresolved:
        return expected, "UNRESOLVED: " + " | ".join(unresolved)

    return expected, "OK"



def calculate_metrics(
    expected: set[str],
    returned: list[str],
    k: int = 3,
) -> dict[str, Any]:
    """정답 문서와 검색 결과를 비교한다."""

    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")

    returned_set = set(returned)
    top_k = returned[:k]

    true_positive = expected & returned_set
    false_positive = returned_set - expected
    false_negative = expected - returned_set

    if expected:
        recall = len(true_positive) / len(expected)
        top1_correct = bool(returned) and returned[0] in expected
    else:
        # 정답이 없는 질문은 아무것도 반환하지 않아야 정답이다.
        recall = 0.0
        top1_correct = not returned

    precision_at_k = len(set(top_k) & expected) / k

    return {
        "top1_correct": top1_correct,
        "precision_at_k": precision_at_k,
        "recall": recall,
        "true_positive_ids": sorted(true_positive),
        "false_positive_ids": sorted(false_positive),
        "false_negative_ids": sorted(false_negative),
    }


def classify_issue(
    expected: set[str],
    returned: list[str],
    resolved: bool,
    evaluation_type: str,
) -> str:
    if not resolved:
        return "GROUND_TRUTH_UNRESOLVED"

    if evaluation_type == "agency_pool":
        return "AGENCY_POOL_DIAGNOSTIC"

    if not expected and not returned:
        return "TRUE_NEGATIVE"

    if not expected and returned:
        return "FALSE_POSITIVE"

    if expected and not returned:
        return "FALSE_NEGATIVE"

    if expected and set(returned) == expected:
        return "EXACT_MATCH"

    if expected & set(returned):
        if expected - set(returned):
            return "PARTIAL_RECALL"
        return "MATCH_WITH_EXTRA_CANDIDATES"

    return "WRONG_DOCUMENT"


def evaluate(projects: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for case in BENCHMARK:
        question = case["question"]
        evaluation_type = case.get("evaluation_type", "supervised")

        expected, resolution = resolve_expected_ids(case, projects)
        resolved = resolution == "OK"

        # 기관 후보 확장 자체를 평가하는 별도 케이스
        if evaluation_type == "agency_pool":
            agency_key = compact(case["expected_agency"])
            agency_ids = {
                p.project_id
                for p in projects
                if agency_key in compact(p.agency_name)
            }
            expected = agency_ids
            resolved = bool(agency_ids)
            resolution = (
                f"AGENCY_POOL_SIZE={len(agency_ids)}"
                if agency_ids
                else "UNRESOLVED_AGENCY"
            )

        # A: 기존 운영 코드
        a_ids = find_project_ids(question, projects)

        # B: 실험용 기관 후보 확장
        b_ids, b_strategy = run_experiment_b(question, projects, a_ids)

        a_metrics = calculate_metrics(expected, a_ids)
        b_metrics = calculate_metrics(expected, b_ids)

        a_issue = classify_issue(
            expected, a_ids, resolved, evaluation_type
        )
        b_issue = classify_issue(
            expected, b_ids, resolved, evaluation_type
        )

        rows.append({
            "case_id": case["id"],
            "question": question,
            "reason": case["reason"],
            "evaluation_type": evaluation_type,
            "ground_truth_status": resolution,
            "expected_ids": sorted(expected),
            "a_ids": a_ids,
            "b_ids": b_ids,
            "b_strategy": b_strategy,
            "a_top1_correct": a_metrics["top1_correct"],
            "b_top1_correct": b_metrics["top1_correct"],
            "a_precision_at_3": a_metrics["precision_at_k"],
            "b_precision_at_3": b_metrics["precision_at_k"],
            "a_recall": a_metrics["recall"],
            "b_recall": b_metrics["recall"],
            "a_false_positive_ids": a_metrics["false_positive_ids"],
            "b_false_positive_ids": b_metrics["false_positive_ids"],
            "a_false_negative_ids": a_metrics["false_negative_ids"],
            "b_false_negative_ids": b_metrics["false_negative_ids"],
            "a_issue": a_issue,
            "b_issue": b_issue,
        })

    return rows



def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """긍정 질문과 부정 질문을 분리해 평가 지표를 계산한다."""

    supervised = [
        row for row in rows
        if row["evaluation_type"] == "supervised"
        and row["ground_truth_status"] == "OK"
    ]

    positive_rows = [
        row for row in supervised
        if row["expected_ids"]
    ]

    negative_rows = [
        row for row in supervised
        if not row["expected_ids"]
    ]

    tp = fp = fn = 0
    top1_correct_count = 0
    precision_sum = 0.0
    recall_sum = 0.0

    negative_true_negative = 0
    negative_false_positive = 0

    for row in supervised:
        expected = set(row["expected_ids"])
        returned = row[f"{prefix}_ids"]

        metrics = calculate_metrics(expected, returned)

        tp += len(metrics["true_positive_ids"])
        fp += len(metrics["false_positive_ids"])
        fn += len(metrics["false_negative_ids"])

        top1_correct_count += int(metrics["top1_correct"])

        if expected:
            precision_sum += metrics["precision_at_k"]
            recall_sum += metrics["recall"]

        else:
            if not returned:
                negative_true_negative += 1
            else:
                negative_false_positive += 1

    total_questions = len(supervised)
    positive_count = len(positive_rows)
    negative_count = len(negative_rows)

    return {
        "evaluated_questions": total_questions,
        "positive_questions": positive_count,
        "negative_questions": negative_count,

        "top1_accuracy": (
            top1_correct_count / total_questions
            if total_questions else 0.0
        ),

        "micro_precision": (
            tp / (tp + fp)
            if tp + fp else 0.0
        ),

        "micro_recall": (
            tp / (tp + fn)
            if tp + fn else 0.0
        ),

        "micro_f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else 0.0
        ),

        # P@3와 Recall 평균은 정답이 있는 질문만 대상으로 한다.
        "mean_precision_at_3": (
            precision_sum / positive_count
            if positive_count else 0.0
        ),

        "mean_recall": (
            recall_sum / positive_count
            if positive_count else 0.0
        ),

        # 정답이 없는 질문에 대한 별도 평가
        "negative_accuracy": (
            negative_true_negative / negative_count
            if negative_count else 0.0
        ),

        # 오검색 결과가 하나라도 나온 부정 질문의 개수
        "negative_false_positive": negative_false_positive,

        "negative_true_negative": negative_true_negative,

        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "case_id", "question", "reason", "evaluation_type",
        "ground_truth_status", "expected_ids",
        "a_ids", "b_ids", "b_strategy",
        "a_top1_correct", "b_top1_correct",
        "a_precision_at_3", "b_precision_at_3",
        "a_recall", "b_recall",
        "a_false_positive_ids", "b_false_positive_ids",
        "a_false_negative_ids", "b_false_negative_ids",
        "a_issue", "b_issue",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            output = dict(row)
            for key, value in output.items():
                if isinstance(value, list):
                    output[key] = " ; ".join(value)
            writer.writerow({key: output.get(key, "") for key in columns})


def write_report(
    rows: list[dict[str, Any]],
    path: Path,
    project_count: int,
) -> None:
    a = summarize(rows, "a")
    b = summarize(rows, "b")

    lines = [
        "# Search-B A/B 실험 결과",
        "",
        f"- 실행 시각: {datetime.now().isoformat(timespec='seconds')}",
        f"- 전체 프로젝트 수: {project_count}",
        f"- 전체 벤치마크 질문 수: {len(rows)}",
        f"- 평가 가능한 정답 질문 수: {a['evaluated_questions']}",
        "",
        "## 1. 핵심 성능 비교",
        "",
        "| 지표 | A: 기존 검색 | B: 기관 후보 확장 |",
        "|---|---:|---:|",
        f"| Top-1 Accuracy | {a['top1_accuracy']:.3f} | {b['top1_accuracy']:.3f} |",
        f"| Micro Precision | {a['micro_precision']:.3f} | {b['micro_precision']:.3f} |",
        f"| Micro Recall | {a['micro_recall']:.3f} | {b['micro_recall']:.3f} |",
        f"| Micro F1 | {a['micro_f1']:.3f} | {b['micro_f1']:.3f} |",
        f"| Mean Precision@3 | {a['mean_precision_at_3']:.3f} | {b['mean_precision_at_3']:.3f} |",
        f"| Mean Recall | {a['mean_recall']:.3f} | {b['mean_recall']:.3f} |",
        f"| Negative Accuracy | {a['negative_accuracy']:.3f} | {b['negative_accuracy']:.3f} |",
        f"| Negative FP Cases | {a['negative_false_positive']} | {b['negative_false_positive']} |",
        (
            f"| Positive / Negative Questions | "
            f"{a['positive_questions']} / "
            f"{a['negative_questions']} | "
            f"{b['positive_questions']} / "
            f"{b['negative_questions']} |"
        ),
        f"| TP / FP / FN | {a['tp']} / {a['fp']} / {a['fn']} | {b['tp']} / {b['fp']} / {b['fn']} |",
        "",
        "## 2. 질문별 오검색 및 검색 누락",
        "",
        "| ID | 질문 | A 결과 | B 결과 | B 전략 |",
        "|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['case_id']} "
            f"| {row['question']} "
            f"| {row['a_issue']} "
            f"| {row['b_issue']} "
            f"| {row['b_strategy']} |"
        )

    lines.extend([
        "",
        "## 3. 오검색(False Positive)",
        "",
    ])

    for row in rows:
        if row["evaluation_type"] != "supervised":
            continue
        if row["a_false_positive_ids"] or row["b_false_positive_ids"]:
            lines.append(f"### {row['case_id']} — {row['question']}")
            lines.append(f"- A 오검색 ID: {row['a_false_positive_ids']}")
            lines.append(f"- B 오검색 ID: {row['b_false_positive_ids']}")
            lines.append("")

    lines.extend([
        "## 4. 검색 누락(False Negative)",
        "",
    ])

    for row in rows:
        if row["evaluation_type"] != "supervised":
            continue
        if row["a_false_negative_ids"] or row["b_false_negative_ids"]:
            lines.append(f"### {row['case_id']} — {row['question']}")
            lines.append(f"- A 누락 ID: {row['a_false_negative_ids']}")
            lines.append(f"- B 누락 ID: {row['b_false_negative_ids']}")
            lines.append("")

    lines.extend([
        "## 5. 정답 데이터 검증 필요 항목",
        "",
    ])

    unresolved = [
        row for row in rows
        if row["ground_truth_status"] != "OK"
        and row["evaluation_type"] == "supervised"
    ]

    if unresolved:
        for row in unresolved:
            lines.append(
                f"- {row['case_id']}: {row['ground_truth_status']}"
            )
    else:
        lines.append("- 미해결 정답 규칙 없음")

    lines.extend([
        "",
        "## 6. 해석 시 주의사항",
        "",
        (
            "- 기관 후보 확장 케이스는 실제 정답을 확정한 평가가 아니라 "
            "후보 확보 여부를 확인하는 진단이다."
        ),
        "- 정답이 여러 개인 질문은 사전에 지정한 복수 정답을 기준으로 평가한다.",
        (
            "- 이 벤치마크의 점수는 등록된 질문에 대한 결과이며 "
            "전체 자연어 질문 성능을 보장하지 않는다."
        ),
        "- B의 후보 개수가 많아졌다고 정확도가 높아진 것은 아니다. FP와 FN을 함께 비교해야 한다.",
        "- 정답 규칙이 데이터에서 해결되지 않은 질문은 성능 계산에서 제외한다.",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"데이터 CSV를 찾을 수 없습니다: {CSV_PATH}\n"
            "CSV_PATH를 실제 data_list.csv 위치로 수정하세요."
        )

    projects = load_projects(CSV_PATH)

    if not projects:
        raise RuntimeError("CSV에서 프로젝트를 불러오지 못했습니다.")

    rows = evaluate(projects)

    csv_path = OUTPUT_DIR / "results.csv"
    report_path = OUTPUT_DIR / "report.md"

    write_csv(rows, csv_path)
    write_report(rows, report_path, len(projects))

    a = summarize(rows, "a")
    b = summarize(rows, "b")

    print("\n" + "=" * 65)
    print("SEARCH-B A/B 실험 완료")
    print("=" * 65)
    print(f"프로젝트 수: {len(projects)}")
    print(f"질문 수: {len(rows)}")
    print(f"평가 가능 질문 수: {a['evaluated_questions']}")
    print()
    print("[A] 기존 검색")
    print(f"Top-1 Accuracy : {a['top1_accuracy']:.3f}")
    print(f"Precision      : {a['micro_precision']:.3f}")
    print(f"Recall         : {a['micro_recall']:.3f}")
    print(f"F1             : {a['micro_f1']:.3f}")
    print(f"TP / FP / FN   : {a['tp']} / {a['fp']} / {a['fn']}")
    print()
    print("[B] 기관 후보 확장")
    print(f"Top-1 Accuracy : {b['top1_accuracy']:.3f}")
    print(f"Precision      : {b['micro_precision']:.3f}")
    print(f"Recall         : {b['micro_recall']:.3f}")
    print(f"F1             : {b['micro_f1']:.3f}")
    print(f"TP / FP / FN   : {b['tp']} / {b['fp']} / {b['fn']}")
    print()
    print(f"CSV 보고서: {csv_path}")
    print(f"상세 보고서: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
