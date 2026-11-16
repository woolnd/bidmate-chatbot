from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bidmate.retrieval.project_finder import find_project_ids
from bidmate.schemas import ProjectInfo
from evaluation.search_b_experiments.build_local_chroma import MODEL, LocalEmbeddings

GROUPS = [
    (
        "dev",
        "가람가상공단",
        [
            (
                "a",
                "전자기록 이관",
                "종이 결재철을 스캔하고 보존소로 옮긴다.",
                "서류 결재철을 전산화해서 보존소에 넘기는 일",
            ),
            (
                "b",
                "공공요금 정산",
                "납부 금액을 계산하고 과납한 돈을 환급한다.",
                "더 낸 요금을 되돌려받도록 계산하는 업무",
            ),
            (
                "c",
                "콜센터 품질 평가",
                "상담 통화를 듣고 응대 품질을 검사한다.",
                "전화 상담이 잘 됐는지 녹취를 점검하는 사업",
            ),
        ],
        ["청사 외벽 도색", "식당 집기 교체"],
    ),
    (
        "dev",
        "푸름가상재단",
        [
            (
                "a",
                "비상 전력 공급",
                "정전 시 서버를 동작시키는 축전지와 전원 장치를 점검한다.",
                "전기가 끊겼을 때 서버가 켜져 있게 하는 배터리 점검",
            ),
            (
                "b",
                "통합 좌석 예약",
                "공유 사무실 책상의 사용 시간을 신청한다.",
                "사무실 빈 책상을 원하는 시간에 잡는 서비스",
            ),
            (
                "c",
                "자동 안내 전화",
                "전화를 받으면 음성으로 메뉴를 알려 준다.",
                "사람 대신 통화에서 선택지를 읽어 주는 일",
            ),
        ],
        ["서버실 바닥 청소", "외부 방문자 주차권"],
    ),
    (
        "test",
        "다솔가상공단",
        [
            (
                "a",
                "통합 민원 창구",
                "시민이 접수한 불편 신고를 담당 부서로 전달한다.",
                "주민 불편을 신고하면 담당자에게 보내주는 사업",
            ),
            (
                "b",
                "자료 보존 환경 개선",
                "문서를 두는 서고의 온도와 습도를 맞춘다.",
                "서류 보관실에 습기가 차지 않게 하는 일",
            ),
            (
                "c",
                "영상 관제 설비 교체",
                "주차장에 설치된 감시 카메라를 바꾼다.",
                "지하주차장 CCTV 바꾸는 입찰",
            ),
        ],
        ["경비원 유니폼 구입", "방문자 접수대 도장"],
    ),
    (
        "test",
        "나래가상기관",
        [
            (
                "a",
                "급여 검증",
                "명세서에서 야간근무 수당 계산 내역을 대조한다.",
                "밤에 일한 직원에게 줄 돈이 맞는지 보는 것",
            ),
            (
                "b",
                "전기차 충전 구역 정비",
                "배터리를 채우는 설비가 있는 주차면을 수리한다.",
                "전기 자동차에 전기를 넣는 주차 공간을 고치는 일",
            ),
            (
                "c",
                "통신 회선 이중화",
                "인터넷이 끊길 때 예비 연결을 마련한다.",
                "접속이 멈춰도 다른 네트워크 선으로 이어 주는 사업",
            ),
        ],
        ["인건비 지급 계좌 변경", "자동차 보험 갱신"],
    ),
    (
        "test",
        "여울가상재단",
        [
            (
                "a",
                "시설 출입 보안",
                "보안 카드로 출입문을 통과한 내역을 관리한다.",
                "카드 찍고 문을 들어간 사람을 확인하는 시스템",
            ),
            (
                "b",
                "지역 공연 홍보",
                "인근 주민에게 연극과 음악회 일정을 알린다.",
                "동네 사람들에게 무대 행사 날짜를 안내하는 사업",
            ),
            (
                "c",
                "문서 이력 비교",
                "공문 수정본의 바뀐 내용과 수정 시점을 확인한다.",
                "공식 서류의 예전 판과 최근 판에서 무엇이 바뀌었는지 보여 줘",
            ),
        ],
        ["출입증 인쇄기 구매", "공연장 무대 바닥 교체"],
    ),
    (
        "test",
        "해든가상연구소",
        [
            (
                "a",
                "하천 수질 점검",
                "강물에서 오염 물질을 채취하고 농도를 검사한다.",
                "강에서 떠온 물에 나쁜 성분이 얼마나 있는지 재는 일",
            ),
            (
                "b",
                "연구비 집행 검토",
                "실험에 쓴 돈과 영수증을 대조한다.",
                "실험 결제 내역이 영수증과 맞는지 확인하는 사업",
            ),
            (
                "c",
                "연구 자료 공개 포털",
                "실험에서 생산한 데이터를 주민이 다운로드할 수 있게 한다.",
                "누구나 실험 결과 파일을 내려받을 수 있는 웹사이트",
            ),
        ],
        ["하천 주변 조명 교체", "포털 회원 비밀번호 변경"],
    ),
]


def normalize(vectors):
    matrix = np.array(vectors, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def measure(rows, mode, threshold, margin):
    records = []
    for row in rows:
        ranked = row[mode]
        predicted = row["base"]
        if not predicted and ranked[0][0] >= threshold and ranked[0][0] - ranked[1][0] >= margin:
            predicted = [ranked[0][1]]
        records.append(
            {
                "question": row["question"],
                "expected": row["expected"],
                "base": row["base"],
                "found": predicted,
            }
        )
    return {
        "n": len(records),
        "base": sum(r["base"] == r["expected"] for r in records),
        "score": sum(r["found"] == r["expected"] for r in records),
        "wrong_empty": sum(not r["expected"] and bool(r["found"]) for r in records),
        "new_false_match": sum(
            bool(set(r["found"]) - set(r["expected"])) and r["base"] == r["expected"]
            for r in records
        ),
        "rows": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    docs = []
    questions = []
    for group, (fold, agency, entries, negatives) in enumerate(GROUPS):
        for slug, title, description, paraphrase in entries:
            pid = f"{group}-{slug}"
            docs.append((pid, agency, title, description))
            questions.append((fold, agency, f"{agency} {paraphrase} 예산", [pid]))
        for q in negatives:
            questions.append((fold, agency, f"{agency} {q} 예산", []))
    model = LocalEmbeddings()
    titles = normalize(model.embed_documents([title for _, _, title, _ in docs]))
    descriptions = normalize(
        model.embed_documents([f"{title}\n{desc}" for _, _, title, desc in docs])
    )
    vectors = normalize(
        model._embed(
            [f"search_query: {q.removeprefix(agency).strip()}" for _, agency, q, _ in questions]
        )
    )
    projects = [ProjectInfo(pid, agency, title, None, None) for pid, agency, title, _ in docs]
    rows = {"dev": [], "test": []}
    for vec, (fold, agency, question, expected) in zip(vectors, questions, strict=True):
        row = {
            "question": question,
            "expected": expected,
            "base": sorted(find_project_ids(question, projects)),
        }
        for mode, matrix in (("title", titles), ("title_and_description", descriptions)):
            values = matrix @ vec
            row[mode] = sorted(
                (
                    (float(values[i]), pid)
                    for i, (pid, name, _, _) in enumerate(docs)
                    if name == agency
                ),
                reverse=True,
            )
        rows[fold].append(row)
    report = {}
    for mode in ("title", "title_and_description"):
        selection = []
        for threshold in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
            for margin in (0.02, 0.05, 0.08, 0.12):
                m = measure(rows["dev"], mode, threshold, margin)
                selection.append(
                    (m["new_false_match"] == 0, m["score"], -m["wrong_empty"], threshold, margin)
                )
        _, _, _, threshold, margin = max(selection)
        report[mode] = {
            "threshold": threshold,
            "margin": margin,
            **{fold: measure(group, mode, threshold, margin) for fold, group in rows.items()},
        }
        print(
            mode,
            threshold,
            margin,
            {k: v for k, v in report[mode]["test"].items() if k != "rows"},
            flush=True,
        )
    args.output.write_text(
        json.dumps(
            {
                "experiment": "dense_catalog_link",
                "model": MODEL,
                "groups_dev": 2,
                "groups_test": 4,
                "cases": "manually written synthetic descriptions",
                "results": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
