"""4단계: 찾은 조각만 보고 답변을 만든다. 담당: 재웅

답변은 AnswerResult 형식(답변 / 출처 사업명 / 근거 있음 여부)으로 받는다.
형식이 고정돼 있어서 화면 표시와 평가 자동 채점에 그대로 쓸 수 있다.
조각에 근거가 없으면 모른다고 답한다 (has_evidence=False).
gpt-5-mini는 temperature를 바꿀 수 없어서 최대 800토큰과 reasoning_effort로 조절한다.

받는 것: 다듬은 질문, 찾은 조각 (Document, 점수) 목록, 대화 기록
넘기는 것: AnswerResult (답변 / 출처 사업명 / 근거 있음 여부)
"""
