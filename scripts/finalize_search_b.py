from __future__ import annotations

import csv
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REPORT_DIR = ROOT / "reports" / "search_b_experiment"

EVALUATOR = SCRIPTS / "evaluate_search_b_experiments.py"
VISUALIZER = SCRIPTS / "visualize_search_b_results.py"
TEAM_BUILDER = SCRIPTS / "build_search_b_team_report.py"
README_BUILDER = SCRIPTS / "build_search_b_readme.py"

REPORT = REPORT_DIR / "report.md"
CSV_FILE = REPORT_DIR / "results.csv"
GRAPH = REPORT_DIR / "search_b_comparison.png"
SHORT_README = REPORT_DIR / "README.md"
TEAM_REPORT = REPORT_DIR / "TEAM_REPORT.md"
DASHBOARD = REPORT_DIR / "team_dashboard.html"
ROOT_README = ROOT / "README.md"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def run(command: list[str], description: str, required: bool = True) -> bool:
    print(f"\n{'=' * 65}")
    print(description)
    print("=" * 65)

    result = subprocess.run(command, cwd=ROOT)

    if result.returncode != 0:
        print(f"[실패] {description}: 종료 코드 {result.returncode}")
        if required:
            raise RuntimeError(description)
        return False

    print(f"[성공] {description}")
    return True


def backup_files(paths: list[Path]) -> None:
    print("\n기존 파일 백업")

    for path in paths:
        if not path.exists() or not path.is_file():
            continue

        backup = path.with_name(f"{path.name}.before_finalize_{STAMP}.bak")

        shutil.copy2(path, backup)
        print(f"백업: {backup.relative_to(ROOT)}")


def read_metrics() -> dict[str, tuple[float, float]]:
    if not REPORT.exists():
        raise FileNotFoundError(f"보고서가 없습니다: {REPORT}")

    content = REPORT.read_text(encoding="utf-8")

    names = [
        "Top-1 Accuracy",
        "Micro Precision",
        "Micro Recall",
        "Micro F1",
        "Mean Precision@3",
        "Mean Recall",
        "Negative Accuracy",
    ]

    metrics: dict[str, tuple[float, float]] = {}

    for name in names:
        pattern = r"^\|\s*" + re.escape(name) + r"\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"

        match = re.search(pattern, content, re.MULTILINE)

        if not match:
            raise ValueError(f"report.md에서 지표를 찾을 수 없습니다: {name}")

        metrics[name] = (
            float(match.group(1)),
            float(match.group(2)),
        )

    return metrics


def read_confusion_counts() -> dict[str, tuple[int, int]]:
    content = REPORT.read_text(encoding="utf-8")

    match = re.search(
        r"^\|\s*TP\s*/\s*FP\s*/\s*FN\s*\|\s*"
        r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|\s*"
        r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|",
        content,
        re.MULTILINE,
    )

    if not match:
        raise ValueError("report.md에서 TP / FP / FN을 읽을 수 없습니다.")

    values = [int(value) for value in match.groups()]

    return {
        "A": (values[0], values[1], values[2]),
        "B": (values[3], values[4], values[5]),
    }


def read_csv_summary() -> tuple[int, int]:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"결과 CSV가 없습니다: {CSV_FILE}")

    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    return len(rows), len(rows)


def make_graph(
    metrics: dict[str, tuple[float, float]],
    counts: dict[str, tuple[int, int, int]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [
        "Top-1 Accuracy",
        "Micro Precision",
        "Micro Recall",
        "Micro F1",
        "Mean Precision@3",
        "Mean Recall",
    ]

    a_values = [metrics[label][0] for label in labels]
    b_values = [metrics[label][1] for label in labels]

    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(x - width / 2, a_values, width, label="Search-A")
    axes[0].bar(x + width / 2, b_values, width, label="Search-B")

    axes[0].set_title("Search-A vs Search-B")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    categories = ["TP", "FP", "FN"]
    a_counts = counts["A"]
    b_counts = counts["B"]

    x2 = np.arange(len(categories))

    axes[1].bar(
        x2 - width / 2,
        a_counts,
        width,
        label="Search-A",
    )
    axes[1].bar(
        x2 + width / 2,
        b_counts,
        width,
        label="Search-B",
    )

    axes[1].set_title("Document-level TP / FP / FN")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Search-B Experiment Results", fontsize=15)
    fig.tight_layout()
    fig.savefig(GRAPH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"그래프 생성: {GRAPH.relative_to(ROOT)}")


def make_experiment_readme(
    metrics: dict[str, tuple[float, float]],
    counts: dict[str, tuple[int, int, int]],
    total_rows: int,
) -> None:
    a = counts["A"]
    b = counts["B"]

    lines = [
        "# Search-B 실험 결과",
        "",
        "## 1. 실험 개요",
        "",
        f"- 전체 프로젝트 수: {100}",
        f"- 벤치마크 CSV 행 수: {total_rows}",
        "- 평가 대상: supervised 및 ground truth가 확인된 질문",
        "- 비교 대상: Search-A와 Search-B",
        "",
        "## 2. A/B 성능 비교",
        "",
        "| 지표 | Search-A | Search-B |",
        "|---|---:|---:|",
    ]

    for name, (a_value, b_value) in metrics.items():
        lines.append(f"| {name} | {a_value:.3f} | {b_value:.3f} |")

    lines.extend(
        [
            "",
            "## 3. 문서 단위 오류 집계",
            "",
            "| 항목 | Search-A | Search-B |",
            "|---|---:|---:|",
            f"| TP | {a[0]} | {b[0]} |",
            f"| FP | {a[1]} | {b[1]} |",
            f"| FN | {a[2]} | {b[2]} |",
            "",
            "## 4. 지표 해석",
            "",
            "- Top-1 Accuracy: 전체 평가 질문 중 첫 번째 결과가 정답인 비율. "
            "부정 질문은 결과가 없을 때 정답으로 처리한다.",
            "- Micro Precision: 전체 문서 단위 TP / (TP + FP).",
            "- Micro Recall: 전체 문서 단위 TP / (TP + FN).",
            "- Micro F1: 문서 단위 TP, FP, FN을 합산해 계산한 F1.",
            "- Mean Precision@3: 정답이 있는 질문별 Precision@3의 평균.",
            "- Mean Recall: 정답이 있는 질문별 Recall의 평균.",
            "- Negative Accuracy: 부정 질문 중 검색 결과가 비어 있는 비율.",
            "",
            "TP, FP, FN은 질문 개수가 아니라 문서 ID 집합을 기준으로 집계한다.",
            "부정 질문은 Negative Accuracy로 별도 평가한다.",
            "",
            "## 5. Q05 오류 분석",
            "",
            "질문: 인천공항운영서비스 차세대 ERP 시스템 구축",
            "",
            "- 원인: 법인 표기 및 유니코드 표현 차이로 기관 후보 추출 실패",
            "- 기존 문제: 기관 후보가 추출되지 않아 기존 검색 결과로 fallback",
            "- 개선: NFKC 정규화 및 법인 접미사 제거를 적용",
            "- 개선 결과: Search-B가 인천공항운영서비스 ERP 문서를 정확히 검색",
            "",
            "## 6. 재현 방법",
            "",
            "프로젝트 루트에서 다음 명령을 실행한다.",
            "",
            "```bash",
            "uv run python scripts/finalize_search_b.py",
            "```",
            "",
            "## 7. 산출물",
            "",
            "- `report.md`: 상세 평가 보고서",
            "- `results.csv`: 질문별 검색 결과",
            "- `search_b_comparison.png`: A/B 성능 비교 그래프",
            "- `TEAM_REPORT.md`: 팀 공유 보고서",
            "- `team_dashboard.html`: HTML 대시보드",
            "",
            "## 8. 평가 범위 및 한계",
            "",
            "실험 결과는 현재 벤치마크 데이터에 대한 관측값이다. "
            "실제 운영 환경의 모든 질문에 대한 정확도를 보장하지 않는다.",
            "더 큰 평가 데이터, 기관명 표기 변형, 부정 질문, 순위 평가, "
            "운영 성능 측정으로 추가 검증해야 한다.",
            "",
            f"<sub>Generated by `scripts/finalize_search_b.py` · Snapshot: {STAMP}</sub>",
        ]
    )

    SHORT_README.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"실험 README 생성: {SHORT_README.relative_to(ROOT)}")


def normalize_root_readme() -> None:
    if not ROOT_README.exists():
        raise FileNotFoundError("프로젝트 루트 README.md가 없습니다.")

    content = ROOT_README.read_text(encoding="utf-8")

    content = re.sub(
        r"(?m)^(\|\s*)Precision(\s*\|)",
        r"\1Micro Precision\2",
        content,
    )
    content = re.sub(
        r"(?m)^(\|\s*)Recall(\s*\|)",
        r"\1Micro Recall\2",
        content,
    )
    content = re.sub(
        r"(?m)^(\|\s*)F1(\s*\|)",
        r"\1Micro F1\2",
        content,
    )

    ROOT_README.write_text(content, encoding="utf-8")
    print("루트 README 지표 명칭 확인 및 정리 완료")


def replace_visualizer_with_wrapper() -> None:
    wrapper = '''"""최신 Search-B 산출물을 생성하는 호환 실행 진입점."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "scripts" / "finalize_search_b.py"

raise SystemExit(
    subprocess.call(
        [sys.executable, str(FINALIZER)],
        cwd=ROOT,
    )
)
'''

    VISUALIZER.write_text(wrapper, encoding="utf-8")
    print("기존 시각화 스크립트를 최종 생성 스크립트 연결용으로 변경")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        EVALUATOR,
        VISUALIZER,
        TEAM_BUILDER,
        README_BUILDER,
        ROOT_README,
        REPORT,
        CSV_FILE,
        GRAPH,
        SHORT_README,
        TEAM_REPORT,
        DASHBOARD,
    ]

    backup_files(targets)

    required_files = [
        EVALUATOR,
        VISUALIZER,
        TEAM_BUILDER,
        README_BUILDER,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"필수 파일이 없습니다: {path}")

    run(
        ["uv", "run", "python", str(EVALUATOR)],
        "1. A/B 평가 재실행",
    )

    metrics = read_metrics()
    counts = read_confusion_counts()
    total_rows, _ = read_csv_summary()

    print("\n최신 report.md에서 읽은 지표")
    for name, values in metrics.items():
        print(f"{name}: A={values[0]:.3f}, B={values[1]:.3f}")

    print(f"TP/FP/FN A: {counts['A']}")
    print(f"TP/FP/FN B: {counts['B']}")
    print(f"CSV 행 수: {total_rows}")

    make_graph(metrics, counts)
    make_experiment_readme(metrics, counts, total_rows)

    run(
        ["uv", "run", "python", str(TEAM_BUILDER)],
        "2. 팀 보고서 및 HTML 대시보드 생성",
    )

    run(
        ["uv", "run", "python", str(README_BUILDER)],
        "3. 프로젝트 루트 README 생성",
    )

    normalize_root_readme()
    replace_visualizer_with_wrapper()

    print("\n최종 테스트 실행")

    tests_ok = run(
        ["uv", "run", "pytest", "-q"],
        "4. 전체 pytest",
        required=False,
    )

    ruff_ok = run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "scripts",
            "src/bidmate/retrieval/project_finder.py",
            "tests",
        ],
        "5. Ruff 전체 검사",
        required=False,
    )

    print("\n" + "=" * 65)
    print("최종 산출물 확인")
    print("=" * 65)

    outputs = [
        ROOT_README,
        REPORT,
        CSV_FILE,
        GRAPH,
        SHORT_README,
        TEAM_REPORT,
        DASHBOARD,
    ]

    all_exist = True

    for path in outputs:
        exists = path.exists()
        all_exist = all_exist and exists
        status = "존재" if exists else "누락"
        print(f"[{status}] {path.relative_to(ROOT)}")

    print("\n검증 결과")
    print(f"전체 테스트: {'통과' if tests_ok else '확인 필요'}")
    print(f"Ruff: {'통과' if ruff_ok else '확인 필요'}")
    print(f"산출물: {'전체 존재' if all_exist else '일부 누락'}")

    print("\nGit 상태")
    subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=False,
    )

    if not (tests_ok and ruff_ok and all_exist):
        raise SystemExit("일부 검증이 통과하지 못했습니다. 결과를 확인하세요.")

    print("\n최종 생성 및 검증 완료.")
    print("Git commit 및 push는 수행하지 않았습니다.")


if __name__ == "__main__":
    main()
