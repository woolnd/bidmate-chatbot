"""파트 간 약속을 지키는지 확인하는 테스트.

각자 자기 함수를 진짜로 바꾼 뒤에도 이 테스트가 통과해야 한다.
OpenAI API를 부르지 않으므로 CI에서 돌려도 비용이 들지 않는다.
"""

from bidmate.schemas import CHUNK_METADATA_KEYS
from tests.fake_chunks import make_fake_chunks


def test_fake_chunks_follow_chunk_rules() -> None:
    """지어낸 조각이 조각 규칙(이름표 4칸, 머리말, 조각 번호)을 지키는지 확인한다."""
    chunks = make_fake_chunks()
    assert len(chunks) == 10
    for chunk in chunks:
        metadata = chunk.metadata
        assert tuple(metadata) == CHUNK_METADATA_KEYS
        assert chunk.page_content.startswith(
            f"[{metadata['agency_name']} / {metadata['project_name']}]\n"
        )
        assert metadata["chunk_id"].startswith(metadata["project_id"] + "-")
