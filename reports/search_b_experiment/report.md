# Search-B A/B 실험 결과

- 실행 시각: 2026-11-11T14:23:08
- 전체 프로젝트 수: 100
- 전체 벤치마크 질문 수: 20
- 평가 가능한 정답 질문 수: 19

## 1. 핵심 성능 비교

| 지표 | A: 기존 검색 | B: 기관 후보 확장 |
|---|---:|---:|
| Top-1 Accuracy | 0.842 | 1.000 |
| Micro Precision | 0.941 | 1.000 |
| Micro Recall | 0.842 | 1.000 |
| Micro F1 | 0.889 | 1.000 |
| Mean Precision@3 | 0.314 | 0.373 |
| Mean Recall | 0.824 | 1.000 |
| Negative Accuracy | 1.000 | 1.000 |
| Negative FP Cases | 0 | 0 |
| Positive / Negative Questions | 17 / 2 | 17 / 2 |
| TP / FP / FN | 16 / 1 / 3 | 19 / 0 / 0 |

## 2. 질문별 오검색 및 검색 누락

| ID | 질문 | A 결과 | B 결과 | B 전략 |
|---|---|---|---|---|
| Q01 | 철도 예약발매 시스템 개량 | FALSE_NEGATIVE | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q02 | 예약발매시스템 개량 | EXACT_MATCH | EXACT_MATCH | NO_AGENCY_USE_A |
| Q03 | 한국가스공사 차세대 통합정보시스템 ERP 구축 | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q04 | 가스공사 ERP 구축 | FALSE_NEGATIVE | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q05 | 인천공항운영서비스 차세대 ERP 시스템 구축 | WRONG_DOCUMENT | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q06 | 통합정보시스템 고도화 용역 | EXACT_MATCH | EXACT_MATCH | NO_AGENCY_USE_A |
| Q07 | 한국한의학연구원 통합정보시스템 고도화 | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q08 | 의료기기산업 종합정보시스템 기능개선 | EXACT_MATCH | EXACT_MATCH | NO_AGENCY_USE_A |
| Q09 | BioIN 의료기기산업 종합정보시스템 기능개선 | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q10 | 한국수자원공사 시스템 구축 | AGENCY_POOL_DIAGNOSTIC | AGENCY_POOL_DIAGNOSTIC | AGENCY_CANDIDATE_EXPANSION |
| Q11 | 광주과학기술원 연구비관리시스템 연계 모듈 변경 | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q12 | 고려대학교 포털 학사정보시스템 | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q13 | 삼성전자 스마트폰 개발 사업 | TRUE_NEGATIVE | TRUE_NEGATIVE | NO_AGENCY_USE_A |
| Q14 | 국민연금공단 이러닝 사업 기간은? | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q15 | 국민연금공단 이러닝 사업 예산은? | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q16 | 예약발매시스템 개량 사업비는? | EXACT_MATCH | EXACT_MATCH | NO_AGENCY_USE_A |
| Q17 | 모바일오피스 고도화 사업 예산은? | EXACT_MATCH | EXACT_MATCH | NO_AGENCY_USE_A |
| Q18 | 운행정보기록 자동분석시스템 계약 방식은? | EXACT_MATCH | EXACT_MATCH | NO_AGENCY_USE_A |
| Q19 | 고려대 포털 학사 시스템 예산과 마감일은? | EXACT_MATCH | EXACT_MATCH | AGENCY_AND_PROJECT |
| Q20 | 삼성전자 스마트폰 개발 사업 예산 | TRUE_NEGATIVE | TRUE_NEGATIVE | NO_AGENCY_USE_A |

## 3. 오검색(False Positive)

### Q05 — 인천공항운영서비스 차세대 ERP 시스템 구축
- A 오검색 ID: ['한국가스공사_[재공고]차세대 통합정보시스템(ERP) 구축.hwp']
- B 오검색 ID: []

## 4. 검색 누락(False Negative)

### Q01 — 철도 예약발매 시스템 개량
- A 누락 ID: ['한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp']
- B 누락 ID: []

### Q04 — 가스공사 ERP 구축
- A 누락 ID: ['한국가스공사_[재공고]차세대 통합정보시스템(ERP) 구축.hwp']
- B 누락 ID: []

### Q05 — 인천공항운영서비스 차세대 ERP 시스템 구축
- A 누락 ID: ['인천공항운영서비스(주)_인천공항운영서비스㈜ 차세대 ERP시스템 구축 .hwp']
- B 누락 ID: []

## 5. 정답 데이터 검증 필요 항목

- 미해결 정답 규칙 없음

## 6. 해석 시 주의사항

- 기관 후보 확장 케이스는 실제 정답을 확정한 평가가 아니라 후보 확보 여부를 확인하는 진단이다.
- 정답이 여러 개인 질문은 사전에 지정한 복수 정답을 기준으로 평가한다.
- 이 벤치마크의 점수는 등록된 질문에 대한 결과이며 전체 자연어 질문 성능을 보장하지 않는다.
- B의 후보 개수가 많아졌다고 정확도가 높아진 것은 아니다. FP와 FN을 함께 비교해야 한다.
- 정답 규칙이 데이터에서 해결되지 않은 질문은 성능 계산에서 제외한다.
