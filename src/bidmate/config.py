"""설정값을 한곳에서 관리한다.

모델 이름, 경로, 숫자를 코드 곳곳에 직접 쓰지 말고 여기서 가져다 쓴다.
비밀값(API 키, 비밀번호)은 .env 파일에서 읽는다.

넣을 값 (기획서 기준 시작값):
    답변 모델          gpt-5-mini (시나리오 A는 Ollama 모델, 11/17부터 바꾸기 설정)
    임베딩 모델        text-embedding-3-small
    답변 최대 길이     800토큰, reasoning_effort
    조각 길이 / 겹침   1000자 / 200자 (2주차에 500·1000·1500 비교)
    k                  찾아올 조각 수 (2주차에 3·5·10 비교)
    대화 기록          최근 3번의 질문·답변
    경로               원본 문서, 목록표, Chroma 저장 폴더, 로그 폴더
    기능 켜기/끄기     1단계, 2단계, 조각 앞 기관명 (11/23~24 기능 끄기 실험용)
"""

from pathlib import Path

# 경로
PROJECT_ROOT = Path(file).resolve().parents[2]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"  # 원본 RFP 파일 (Git에 안 올라간다)
RAW_DATA_DIR = RAW_FOLDER  # metadata.py가 쓰는 이름 (RAW_FOLDER와 같은 경로)
PROJECT_LIST_PATH = RAW_FOLDER / "data_list.csv"  # 목록표

# 조각 자르기
CHUNK_SIZE = 1000  # 조각 하나의 글자 수
CHUNK_OVERLAP = 200  # 앞 조각과 겹치는 글자 수

# 검색
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHROMA_FOLDER = PROJECT_ROOT / "data" / "chroma"  # Chroma 저장 폴더 (Git에 안 올라간다)
# 영어·숫자·. _ - 로 3글자 이상. 조각 크기를 바꾸면 이름도 바꿔서 옛 조각과 섞이지 않게 한다
COLLECTION_NAME = f"rfp_chunks_{CHUNK_SIZE}"
TOP_K = 5  # 찾아올 조각 수 (2주차에 3·5·10 비교)