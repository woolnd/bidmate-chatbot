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

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env의 OPENAI_API_KEY 등을 읽는다

# 경로
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"  # 원본 RFP 파일 (Git에 안 올라간다)
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

# 생성 (1·4단계)
ANSWER_MODEL_NAME = "gpt-5-mini"  # 1단계 질문 다듬기와 4단계 답변이 같이 쓰는 모델
MAX_HISTORY_TURNS = 3  # AI에게 넘길 대화 기록 수 (최근 질문·답변 3번)

# 1단계 질문 다듬기
REWRITE_ANSWER_PREVIEW_LENGTH = 200  # 대화 기록의 답변은 앞 200자만 넣는다 (기관명·사업명만 필요)
REWRITE_REASONING_EFFORT = "minimal"  # 간단한 일이라 생각을 최소로 해서 빠르고 싸게
REWRITE_MAX_TOKENS = 300  # 다듬은 질문 한 줄이면 충분. 생각에 쓴 토큰도 여기에 포함된다

# 4단계 답변 만들기
ANSWER_REASONING_EFFORT = "low"  # 조각 여러 개에서 맞는 내용을 골라야 해서 minimal보다 한 단계 위
ANSWER_MAX_TOKENS = 800  # 생각에 쓴 토큰도 포함된다. 답이 잘리면 늘린다

# 기능 켜기/끄기 (11/23~24 기능 끄기 실험용)
USE_QUESTION_REWRITE = True  # 1단계 질문 다듬기. 끄면 질문을 그대로 넘긴다

# 웹 서버 (FastAPI)
# 화면 비밀번호. .env에서 읽는다. 비어 있으면 비밀번호 없이 열린다 (개발 모드)
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
