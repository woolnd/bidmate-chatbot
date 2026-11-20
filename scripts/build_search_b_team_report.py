# ruff: noqa: E501
"""Search-B 실험 결과를 팀 공유용 HTML 대시보드와 Markdown 보고서로 생성한다."""

from __future__ import annotations

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "search_b_experiment"
CSV_PATH = REPORT_DIR / "results.csv"
HTML_PATH = REPORT_DIR / "team_dashboard.html"
MD_PATH = REPORT_DIR / "TEAM_REPORT.md"

# 2026-11-11 최신 실험 결과
METRICS = [
    ("Top-1 Accuracy", 0.842, 0.947, "질문별 최상위 검색 결과의 정답률"),
    ("Precision", 0.941, 0.947, "검색 결과 중 정답 문서의 비율"),
    ("Recall", 0.842, 0.947, "찾아야 할 정답 문서를 찾아낸 비율"),
    ("F1", 0.889, 0.947, "Precision과 Recall의 조화 평균"),
    ("Precision@3", 0.314, 0.353, "상위 3개 결과의 평균 정밀도"),
    ("Mean Recall", 0.824, 0.941, "정답이 있는 질문별 Recall 평균"),
]

COUNTS = {
    "TP": (16, 18),
    "FP": (1, 1),
    "FN": (3, 1),
}

CHECKS = [
    ("A/B 실험 결과 CSV", "완료", "results.csv"),
    ("상세 평가 보고서", "완료", "report.md"),
    ("성능 비교 그래프", "완료", "search_b_comparison.png"),
    ("공유용 README 초안", "완료", "README.md"),
    ("신규 평가 테스트", "8 passed", "tests/test_search_b_experiments.py"),
    ("기존 Search-B 테스트", "9 passed", "tests/test_project_finder.py"),
    ("Ruff 코드 검사", "통과", "All checks passed"),
]


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def read_results() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"실험 CSV를 찾을 수 없습니다: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def clean_ids(value: str | None) -> str:
    if not value:
        return "-"
    value = value.strip()
    return value if value else "-"


def metric_cards() -> str:
    cards = []
    for name, a, b, description in METRICS:
        delta = (b - a) * 100
        sign = "+" if delta >= 0 else ""
        cards.append(
            f"""
            <article class="metric-card">
              <div class="metric-name">{esc(name)}</div>
              <div class="metric-values">
                <span class="a-value">{a:.3f}</span>
                <span class="arrow">→</span>
                <span class="b-value">{b:.3f}</span>
              </div>
              <div class="metric-delta">{sign}{delta:.1f}%p</div>
              <p>{esc(description)}</p>
            </article>
            """
        )
    return "\n".join(cards)


def comparison_chart() -> str:
    rows = []
    for name, a, b, _ in METRICS:
        width_a = a * 100
        width_b = b * 100
        rows.append(
            f"""
            <div class="chart-row">
              <div class="chart-label">{esc(name)}</div>
              <div class="bar-line">
                <span class="bar-name">A</span>
                <div class="bar-track">
                  <div class="bar a-bar" style="width:{width_a:.1f}%"></div>
                </div>
                <b>{a:.3f}</b>
              </div>
              <div class="bar-line">
                <span class="bar-name">B</span>
                <div class="bar-track">
                  <div class="bar b-bar" style="width:{width_b:.1f}%"></div>
                </div>
                <b>{b:.3f}</b>
              </div>
            </div>
            """
        )
    return "\n".join(rows)


def count_chart() -> str:
    max_value = max(max(pair) for pair in COUNTS.values()) or 1
    blocks = []
    for label, (a, b) in COUNTS.items():
        width_a = a / max_value * 100
        width_b = b / max_value * 100
        blocks.append(
            f"""
            <div class="count-row">
              <strong>{label}</strong>
              <div class="count-bars">
                <div class="count-line">
                  <span>A</span>
                  <div class="count-track">
                    <div class="count-fill a-bar" style="width:{width_a}%"></div>
                  </div>
                  <b>{a}</b>
                </div>
                <div class="count-line">
                  <span>B</span>
                  <div class="count-track">
                    <div class="count-fill b-bar" style="width:{width_b}%"></div>
                  </div>
                  <b>{b}</b>
                </div>
              </div>
            </div>
            """
        )
    return "\n".join(blocks)


def result_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        case_id = row.get("case_id", "")
        question = row.get("question", "")
        evaluation_type = row.get("evaluation_type", "")
        strategy = row.get("b_strategy", "")
        a_ids = clean_ids(row.get("a_ids"))
        b_ids = clean_ids(row.get("b_ids"))
        a_fn = clean_ids(row.get("a_false_negative_ids"))
        b_fn = clean_ids(row.get("b_false_negative_ids"))
        a_fp = clean_ids(row.get("a_false_positive_ids"))
        b_fp = clean_ids(row.get("b_false_positive_ids"))

        issue = row.get("b_issue") or row.get("a_issue") or ""
        is_q05 = case_id == "Q05"
        has_error = any(value != "-" for value in (a_fn, b_fn, a_fp, b_fp))
        row_class = "critical-row" if is_q05 else ("issue-row" if has_error else "")

        body.append(
            f"""
            <tr class="{row_class}">
              <td><strong>{esc(case_id)}</strong></td>
              <td class="question-cell">{esc(question)}</td>
              <td>{esc(evaluation_type)}</td>
              <td>{esc(strategy)}</td>
              <td class="id-cell">{esc(a_ids)}</td>
              <td class="id-cell">{esc(b_ids)}</td>
              <td class="id-cell">{esc(b_fn)}</td>
              <td class="id-cell">{esc(b_fp)}</td>
              <td>{esc(issue)}</td>
            </tr>
            """
        )

    return "\n".join(body)


def check_table() -> str:
    return "\n".join(
        f"""
        <tr>
          <td>{esc(name)}</td>
          <td><span class="status">{esc(status)}</span></td>
          <td><code>{esc(detail)}</code></td>
        </tr>
        """
        for name, status, detail in CHECKS
    )


def build_html(rows: list[dict[str, str]]) -> str:
    evaluated = [row for row in rows if row.get("evaluation_type") != "AGENCY_POOL_DIAGNOSTIC"]
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Search-B A/B 실험 결과</title>
<style>
:root {{
  --bg:#f4f6fa; --panel:#fff; --text:#202938; --muted:#667085;
  --line:#e5e9f0; --a:#8492a6; --b:#3269d5;
  --green:#16845b; --red:#c34c4c; --soft:#edf3ff;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",
  "Malgun Gothic",sans-serif; line-height:1.6;
}}
main {{ max-width:1200px; margin:0 auto; padding:36px 22px 70px; }}
header {{
  background:#17243a; color:white; padding:34px;
  border-radius:18px; margin-bottom:24px;
}}
header h1 {{ margin:0 0 8px; font-size:30px; }}
header p {{ margin:0; color:#d1d9e6; }}
.meta {{ margin-top:18px; font-size:13px; color:#c2ccda; }}
section {{
  background:var(--panel); border:1px solid var(--line);
  border-radius:15px; padding:24px; margin:20px 0;
}}
h2 {{ margin:0 0 8px; font-size:21px; }}
.subtitle {{ margin:0 0 20px; color:var(--muted); font-size:14px; }}
.metric-grid {{
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px;
}}
.metric-card {{
  border:1px solid var(--line); border-radius:12px; padding:18px;
  background:#fff;
}}
.metric-name {{ font-weight:700; font-size:14px; }}
.metric-values {{ display:flex; align-items:center; gap:9px; margin-top:8px; }}
.metric-values span {{ font-size:25px; font-weight:750; }}
.a-value {{ color:var(--a); }}
.b-value {{ color:var(--b); }}
.arrow {{ color:#98a2b3; }}
.metric-delta {{ color:var(--green); font-weight:700; font-size:14px; }}
.metric-card p {{ color:var(--muted); font-size:12px; margin:7px 0 0; }}
.legend {{ display:flex; gap:18px; margin:12px 0 20px; font-size:13px; }}
.legend span::before {{
  content:""; display:inline-block; width:10px; height:10px;
  border-radius:3px; margin-right:7px; background:var(--a);
}}
.legend .legend-b::before {{ background:var(--b); }}
.chart-row {{ padding:12px 0; border-bottom:1px solid var(--line); }}
.chart-label {{ font-weight:700; margin-bottom:7px; }}
.bar-line {{ display:grid; grid-template-columns:20px 1fr 48px; gap:10px;
  align-items:center; margin:5px 0; font-size:13px; }}
.bar-name {{ color:var(--muted); }}
.bar-track,.count-track {{ height:13px; background:#edf0f4; border-radius:20px; overflow:hidden; }}
.bar {{ height:100%; border-radius:20px; }}
.a-bar {{ background:var(--a); }}
.b-bar {{ background:var(--b); }}
.bar-line b {{ text-align:right; font-variant-numeric:tabular-nums; }}
.count-row {{
  display:grid; grid-template-columns:50px 1fr; gap:14px;
  padding:12px 0; border-bottom:1px solid var(--line);
}}
.count-bars {{ display:grid; gap:7px; }}
.count-line {{ display:grid; grid-template-columns:20px 1fr 25px;
  align-items:center; gap:9px; font-size:13px; }}
.count-fill {{ height:100%; border-radius:20px; }}
.callout {{
  border-left:4px solid var(--red); background:#fff5f4;
  padding:17px 19px; border-radius:8px; margin-top:14px;
}}
.callout strong {{ color:#a83232; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:11px 10px; border-bottom:1px solid var(--line);
  text-align:left; vertical-align:top; }}
th {{ background:#f7f8fb; white-space:nowrap; }}
.question-cell {{ min-width:230px; }}
.id-cell {{ min-width:150px; overflow-wrap:anywhere; }}
.critical-row {{ background:#fff4f2; }}
.issue-row {{ background:#fffaf0; }}
.status {{
  display:inline-block; padding:3px 9px; border-radius:20px;
  background:#e8f6ef; color:#14764f; font-weight:700; white-space:nowrap;
}}
code {{ color:#475467; overflow-wrap:anywhere; }}
.steps {{ padding-left:20px; }}
.steps li {{ margin:8px 0; }}
footer {{ color:var(--muted); font-size:12px; text-align:center; padding:20px; }}
@media(max-width:760px) {{
  main {{ padding:16px 12px 40px; }}
  header {{ padding:24px 20px; }}
  header h1 {{ font-size:24px; }}
  .metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  section {{ padding:18px 14px; }}
}}
@media(max-width:440px) {{
  .metric-grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>Search-B A/B 실험 결과</h1>
  <p>기존 검색 방식(A)과 기관 후보 확장 방식(B)의 검색 성능 비교</p>
  <div class="meta">실험 기준일: 2026-11-11 · 프로젝트 100개 · 벤치마크 질문 20개</div>
</header>

<section>
  <h2>1. 핵심 성과</h2>
  <p class="subtitle">A는 기존 검색, B는 기관 후보 확장 전략을 적용한 검색 결과입니다.</p>
  <div class="metric-grid">{metric_cards()}</div>
  <p class="subtitle" style="margin-top:18px">
    평가 가능 질문 {len(evaluated)}개 기준 · 전체 20개 중 진단 전용 질문 1개 제외
  </p>
</section>

<section>
  <h2>2. A/B 성능 비교</h2>
  <p class="subtitle">막대 길이는 각 지표의 0~1 범위를 나타냅니다.</p>
  <div class="legend"><span>A 기존 검색</span><span class="legend-b">B 기관 후보 확장</span></div>
  {comparison_chart()}
</section>

<section>
  <h2>3. 정답·오답 건수 비교</h2>
  <p class="subtitle">TP는 정답 검색, FP는 오검색, FN은 정답 문서 누락입니다.</p>
  {count_chart()}
  <div class="callout">
    <strong>핵심 해석</strong><br>
    B는 A보다 TP가 16건에서 18건으로 증가했고, FN은 3건에서 1건으로 감소했습니다.
    FP는 1건으로 동일합니다. 후보 수 증가 자체가 아니라 정답 회수와 오검색을 함께 봐야 합니다.
  </div>
</section>

<section>
  <h2>4. 질문별 검색 결과</h2>
  <p class="subtitle">오답이 있는 행은 별도 색상으로 표시했습니다. Q05는 현재 우선 분석 대상입니다.</p>
  <div class="table-wrap">
  <table>
    <thead><tr>
      <th>질문</th><th>질문 내용</th><th>평가 유형</th><th>B 전략</th>
      <th>A 결과 ID</th><th>B 결과 ID</th><th>B 누락(FN)</th><th>B 오검색(FP)</th><th>이슈</th>
    </tr></thead>
    <tbody>{result_table(rows)}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>5. 현재 핵심 개선 과제</h2>
  <div class="callout">
    <strong>Q05 — 인천공항운영서비스 차세대 ERP 시스템 구축</strong><br>
    A와 B 모두 한국가스공사의 차세대 ERP 구축 문서를 반환하고,
    실제 인천공항운영서비스 문서를 누락했습니다.
    기관명 인식, 별칭 매칭, 기관 필터링, A fallback 경로를 순서대로 확인해야 합니다.
  </div>
  <ul>
    <li>Q01 철도 예약발매 시스템 개량: B에서 정답 검색 성공</li>
    <li>Q04 가스공사 ERP 구축: B에서 정답 검색 성공</li>
    <li>Q13, Q20 삼성전자 스마트폰 개발 사업: 정답 문서가 없는 질문이며 결과 미반환</li>
    <li>Q10: 기관 후보 풀 진단 전용으로 성능 평가에서 제외</li>
  </ul>
</section>

<section>
  <h2>6. 산출물 및 품질 검증</h2>
  <p class="subtitle">현재까지 생성·검증한 결과물입니다.</p>
  <div class="table-wrap">
  <table>
    <thead><tr><th>항목</th><th>상태</th><th>파일 또는 결과</th></tr></thead>
    <tbody>{check_table()}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>7. 평가 범위와 해석 기준</h2>
  <ul>
    <li>프로젝트 데이터 100개, 질문 20개를 사용했습니다.</li>
    <li>평가 가능 질문은 19개이며, positive 17개와 negative 2개로 구성됩니다.</li>
    <li>Q10은 기관 후보 풀 진단 전용으로 평가에서 제외했습니다.</li>
    <li>Q13과 Q20은 정답 문서가 없는 negative 질문입니다.</li>
    <li>Micro Recall과 Mean Recall은 계산 단위가 다르므로 서로 같은 지표로 해석하면 안 됩니다.</li>
    <li>Precision@3 평균은 정답 문서가 있는 positive 질문을 대상으로 계산합니다.</li>
  </ul>
</section>

<section>
  <h2>8. 다음 작업 순서</h2>
  <ol class="steps">
    <li>Q05 기관명 추출 및 별칭 매칭 실패 원인 분석</li>
    <li>기관명이 명시된 질문에서 타 기관 문서가 반환되지 않도록 검색 로직 개선</li>
    <li>Q06, Q08 등 유사 사업명 질문의 오검색 가능성 점검</li>
    <li>전체 벤치마크 재실행 및 A/B 지표 재계산</li>
    <li>그래프와 팀 공유 보고서 갱신</li>
    <li>최종 테스트 후 코드와 문서를 한 번에 GitHub 공유</li>
  </ol>
</section>

<footer>Search-B Experiment Report · Generated from results.csv</footer>
</main>
</body>
</html>
"""


def build_markdown(rows: list[dict[str, str]]) -> str:
    evaluated = [row for row in rows if row.get("evaluation_type") != "AGENCY_POOL_DIAGNOSTIC"]
    lines = [
        "# Search-B A/B 실험 결과",
        "",
        "> 기존 검색(A)과 기관 후보 확장 검색(B)의 성능 비교",
        "",
        "## 1. 실험 개요",
        "",
        "| 항목 | 내용 |",
        "|---|---:|",
        "| 프로젝트 데이터 | 100개 |",
        "| 전체 벤치마크 질문 | 20개 |",
        f"| 평가 가능 질문 | {len(evaluated)}개 |",
        "| Positive / Negative | 17개 / 2개 |",
        "| 진단 전용 질문 | Q10 1개 |",
        "",
        "## 2. 핵심 성능 비교",
        "",
        "| 지표 | A 기존 검색 | B 기관 후보 확장 | 변화 |",
        "|---|---:|---:|---:|",
    ]

    for name, a, b, _ in METRICS:
        delta = (b - a) * 100
        lines.append(f"| {name} | {a:.3f} | {b:.3f} | {delta:+.1f}%p |")

    lines.extend(
        [
            "",
            "## 3. TP / FP / FN 비교",
            "",
            "| 항목 | A | B | 해석 |",
            "|---|---:|---:|---|",
            "| TP | 16 | 18 | 정답 검색 증가 |",
            "| FP | 1 | 1 | 오검색 건수 동일 |",
            "| FN | 3 | 1 | 정답 문서 누락 감소 |",
            "",
            "## 4. 주요 성공 및 실패 사례",
            "",
            "| 질문 | A | B | 결과 |",
            "|---|---|---|---|",
            "| Q01 철도 예약발매 시스템 개량 | 실패 | 성공 | B에서 정답 검색 |",
            "| Q04 가스공사 ERP 구축 | 실패 | 성공 | B에서 정답 검색 |",
            "| Q05 인천공항운영서비스 차세대 ERP | 오검색 | 오검색 | 최우선 개선 대상 |",
            "| Q13/Q20 삼성전자 스마트폰 개발 | 미반환 | 미반환 | Negative 정답 |",
            "",
            "## 5. 현재 확인된 핵심 이슈",
            "",
            "- Q05에서 한국가스공사 ERP 문서가 반환되고 실제 인천공항운영서비스 문서가 누락됨.",
            "- 기관명 인식, 별칭 매칭, 기관 필터링, A fallback 경로를 확인해야 함.",
            "- 후보 수 증가만으로 품질을 판단하지 않고 FP와 FN을 함께 줄여야 함.",
            "",
            "## 6. 산출물 및 검증 현황",
            "",
            "| 항목 | 상태 |",
            "|---|---|",
            "| A/B 실험 결과 CSV | 생성 완료 |",
            "| 상세 평가 보고서 | 생성 완료 |",
            "| 비교 그래프 PNG | 생성 완료 |",
            "| 공유용 README 초안 | 생성 완료 |",
            "| 신규 평가 테스트 | 8 passed |",
            "| 기존 Search-B 테스트 | 9 passed |",
            "| Ruff | 통과 |",
            "",
            "## 7. 평가 시 주의사항",
            "",
            "- 평가 가능 질문은 19개이며 Q10은 진단 전용으로 제외함.",
            "- Q13/Q20은 정답 문서가 없는 Negative 질문임.",
            "- Micro Recall은 문서 단위, Mean Recall은 질문별 평균이므로 정의가 다름.",
            "- Precision@3는 Positive 질문을 대상으로 평균 계산함.",
            "",
            "## 8. 다음 작업",
            "",
            "1. Q05 기관명 매칭 실패 원인 분석",
            "2. 기관명이 명시된 질문의 타 기관 문서 오검색 방지",
            "3. Q06, Q08 유사 사업명 사례 점검",
            "4. 전체 벤치마크 재실행",
            "5. 그래프와 보고서 갱신",
            "6. 최종 테스트 후 GitHub 일괄 공유",
            "",
            "## 9. 재현 방법",
            "",
            "```bash",
            "uv run python scripts/evaluate_search_b_experiments.py",
            "uv run python scripts/build_search_b_team_report.py",
            "open reports/search_b_experiment/team_dashboard.html",
            "```",
            "",
            "> 본 문서는 2026-11-11 실험 결과를 기준으로 작성한 공유용 초안입니다.",
            "> Q05 개선 및 전체 재평가가 끝나기 전까지 최종 성능으로 확정하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = read_results()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    HTML_PATH.write_text(build_html(rows), encoding="utf-8")
    MD_PATH.write_text(build_markdown(rows), encoding="utf-8")

    print("\nSearch-B 팀 공유 자료 생성 완료")
    print(f"- HTML 대시보드: {HTML_PATH}")
    print(f"- Markdown 보고서: {MD_PATH}")
    print(f"- CSV 질문 행 수: {len(rows)}")
    print("\n기존 README.md, report.md, CSV, PNG는 수정하지 않았습니다.")


if __name__ == "__main__":
    main()
