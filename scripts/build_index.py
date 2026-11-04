"""미리 한 번 해 두는 일: 목록표 읽기 → 문서 읽기 → 조각 자르기 → Chroma 저장

순서:
    1. data/metadata.py      목록표 읽기 → ProjectInfo 목록
    2. data/loader.py        문서마다 글자 뽑기 → LoadedDocument
    3. data/chunker.py       조각 자르기 + 기본 정보 조각 → Document 목록
    4. retrieval/vector_store.py   Chroma에 저장

실행: uv run python scripts/build_index.py
문서나 조각 자르기 방식이 바뀌었을 때만 다시 돌린다 (임베딩은 돈이 든다).
"""
