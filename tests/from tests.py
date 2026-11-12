from tests.fake_chunks import make_fake_chunks
chunks = make_fake_chunks()

print(len(chunks))                # 10이 나오는지 확인
print(chunks[0].page_content)     # 첫 번째 조각의 머리말과 내용 확인
print(chunks[0].metadata)         # 이름표 키 4개가 다 있는지 확인