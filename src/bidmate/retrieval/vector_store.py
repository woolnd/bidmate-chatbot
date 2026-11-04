"""조각 저장과 3단계(문서에서 관련 부분 찾기). 담당: 검색 파트

저장 (미리 한 번, scripts/build_index.py가 부른다):
    조각(Document)을 임베딩(숫자 목록)으로 바꿔 Chroma에 저장한다.
    metadata(chunk_id, project_id, agency_name, project_name)도 같이 저장된다.
    chunk_id를 저장 id로 써서 같은 조각이 두 번 들어가지 않게 한다.
    임베딩은 돈이 드니까 한 번 만든 것은 다시 만들지 않는다.

찾기 (질문마다):
    project_id 목록이 있으면 metadata로 걸러서 고른 문서 안에서만 찾는다.
    빈 목록이면 전체에서 찾는다.
    k(찾아올 조각 수)는 config.py에서 가져온다. 2주차에 3/5/10을 점수로 비교한다.

받는 것: (저장) Document 목록 / (찾기) 다듬은 질문, project_id 목록
넘기는 것: (찾기) 관련 조각 k개, (Document, 점수) 목록
"""
