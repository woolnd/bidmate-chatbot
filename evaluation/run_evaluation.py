"""평가셋으로 챗봇 점수를 낸다. 담당: 재웅(채점 코드), 데이터-3(평가셋, 점수 정리)

실행: uv run python evaluation/run_evaluation.py evaluation/questions.csv
평가셋 파일(questions.csv)은 원본 RFP 내용이 들어 있어서 Git에 올리지 않는다.
양식은 evaluation/questions_template.csv 를 복사해서 쓴다.

평가셋 칸:
    question_id             질문 번호 (예: Q001)
    question_type           한 문서 / 여러 문서 / 후속 질문 / 없는 내용 / 기본 정보
    question                질문
    previous_question       후속 질문일 때 바로 앞 질문 (아니면 비움)
    expected_answer         정답 (원본 RFP를 직접 보고 적는다)
    expected_project_names  정답 문서 사업명. 여러 개면 | 로 구분. 없는 내용 질문이면 비움
"""
