"""문서를 검색용 조각으로 자르기. 담당: 데이터 파트

조각 = 이름표(metadata) + 내용(page_content). LangChain Document로 만든다.
    page_content  "[기관명 / 사업명]" 머리말 + 본문
    metadata      chunk_id, project_id, agency_name, project_name

규칙:
    - 1000자씩 자르고 앞 조각과 200자 겹친다 (시작값. 2주차에 500/1000/1500 비교)
    - chunk_id = "파일이름-0001" 부터 순서대로
    - 문서마다 기본 정보 조각 1개 (chunk_id = "파일이름-info").
      목록표 한 줄(기관명·사업명·예산·마감일)로 만든다. 예산·마감일 질문용

받는 것: LoadedDocument, ProjectInfo
넘기는 것: Document 목록
"""
