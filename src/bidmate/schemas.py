"""파트 간 데이터 약속.

각 단계가 무엇을 받고 무엇을 넘기는지를 여기서 정한다.
필드를 바꾸거나 지우면 다른 파트 코드가 깨지므로, 바꾸기 전에 PM과 상의한다.

정할 약속:
    ProjectInfo     data_list.csv 한 줄 (project_id, 기관명, 사업명, 예산, 마감일, 파일명)
    LoadedDocument  원본 파일에서 뽑은 글자 (project_id, 파일명, 본문)
    DocumentChunk   검색용 조각 (chunk_id, project_id, 기관명, 사업명, 머리말 붙은 본문)
    RetrievedChunk  검색 결과 한 개 (조각, 점수)
    AnswerResult    답변 (answer, source_project_names, has_evidence)
    ChatState       4단계가 주고받는 상태 (질문, 대화 기록, 단계별 결과)
"""
