"""설정값을 한곳에서 관리한다.

모델 이름, 경로, 숫자를 코드 곳곳에 직접 쓰지 말고 여기서 가져다 쓴다.
비밀값(API 키, 비밀번호)은 .env 파일에서 읽는다.

넣을 값 (기획서 기준 시작값):
    답변 모델          gpt-5-mini (시나리오 A는 Ollama 모델, 11/17부터 바꾸기 설정)
    임베딩 모델        text-embedding-3-small
    답변 최대 길이     2000토큰, reasoning_effort
    조각 길이 / 겹침   1000자 / 200자 (2주차에 500·1000·1500 비교)
    k                  찾아올 조각 수 (2주차에 3·5·10 비교)
    대화 기록          최근 5번의 질문·답변
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
FEEDBACK_LOG_FOLDER = (
    PROJECT_ROOT / "data" / "logs" / "feedback"
)  # 좋아요/별로예요 기록 (Git에 안 올라간다)

# 조각 자르기
CHUNK_SIZE = 1000  # 조각 하나의 글자 수
CHUNK_OVERLAP = 200  # 앞 조각과 겹치는 글자 수

# 검색
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHROMA_FOLDER = PROJECT_ROOT / "data" / "chroma"  # Chroma 저장 폴더 (Git에 안 올라간다)
# 영어·숫자·. _ - 로 3글자 이상. 조각 크기를 바꾸면 이름도 바꿔서 옛 조각과 섞이지 않게 한다
COLLECTION_NAME = f"rfp_chunks_{CHUNK_SIZE}"
TOP_K = 5  # 찾아올 조각 수 (2주차에 3·5·10 비교)
RETRY_TOP_K = 15  # 답에 근거가 없을 때 한 번 더 찾는 조각 수 (에이전트 다시 찾기)

# 3단계 리랭커 (후보를 넉넉히 찾고 질문과 한 쌍씩 다시 채점해서 k개를 고른다)
# .env의 USE_RERANKER=true일 때만 켠다. GPU가 있는 VM은 켜고, 로컬 Mac은 끈다 (Mac은 질문당 약 13초)
# 켜기 전에 선택 패키지를 설치한다: uv sync --extra rerank
USE_RERANKER = os.getenv("USE_RERANKER", "").strip().lower() == "true"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # 한국어를 지원하는 리랭커 (약 2.3GB)
RERANK_CANDIDATE_K = 20  # 리랭커에게 넘길 후보 조각 수 (문서마다)
RERANKER_MAX_LENGTH = 512  # 질문 + 조각을 이 토큰 수까지만 읽는다 (길수록 느리다)

# 3단계 BM25 (질문 단어가 글자 그대로 많이 든 조각을 리랭커 후보에 더한다)
# .env의 USE_BM25=true일 때, 리랭커(USE_RERANKER=true)와 같이 켤 때만 쓴다
# 켜기 전에 선택 패키지를 설치한다: uv sync --extra rerank
USE_BM25 = os.getenv("USE_BM25", "").strip().lower() == "true"
BM25_CANDIDATE_K = 5  # 리랭커 후보에 더할 BM25 조각 수 (문서마다. 실험에서 10개와 점수가 같았다)

# 생성 (1·4단계)
ANSWER_MODEL_NAME = "gpt-5-mini"  # 1단계 질문 다듬기와 4단계 답변이 같이 쓰는 모델
MAX_HISTORY_TURNS = 5  # AI에게 넘길 대화 기록 수 (최근 질문·답변 5번)
# OpenAI 응답을 기다리는 최대 시간(초). 없으면 연결이 멈췄을 때 끝없이 기다린다
# 넘으면 재시도(2번) 뒤 에러. 응답 목표 10초보다 넉넉하게
OPENAI_TIMEOUT_SECONDS = 30

# 1단계 질문 다듬기
REWRITE_ANSWER_PREVIEW_LENGTH = 200  # 대화 기록의 답변은 앞 200자만 넣는다 (기관명·사업명만 필요)
REWRITE_REASONING_EFFORT = "minimal"  # 간단한 일이라 생각을 최소로 해서 빠르고 싸게
REWRITE_MAX_TOKENS = 300  # 다듬은 질문 한 줄이면 충분. 생각에 쓴 토큰도 여기에 포함된다

# 4단계 답변 만들기
ANSWER_REASONING_EFFORT = "low"  # 조각 여러 개에서 맞는 내용을 골라야 해서 minimal보다 한 단계 위
ANSWER_MAX_TOKENS = 2000  # 생각에 쓴 토큰도 포함된다. 다음 질문·다시 찾기(조각 15개)로 늘림

# 4단계 답변 모델 고르기 (시나리오 B: openai / 시나리오 A: ollama). .env에서 바꾼다
# 1단계 질문 다듬기, 3단계 임베딩, 채점 AI는 이 설정과 상관없이 항상 OpenAI
ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "openai")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:9b")  # 10초 목표에 맞게 작은 판부터
OLLAMA_TEMPERATURE = 0  # 같은 질문에 답이 흔들리지 않게 고정

# 기능 켜기/끄기 (11/23~24 기능 끄기 실험용)
USE_QUESTION_REWRITE = True  # 1단계 질문 다듬기. 끄면 질문을 그대로 넘긴다

# 웹 서버 (FastAPI)
# 화면 비밀번호. .env에서 읽는다. 비어 있으면 비밀번호 없이 열린다 (개발 모드)
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# 평가 채점 (evaluation/answer_grader.py)
JUDGE_MODEL_NAME = "gpt-5-mini"  # 챗봇 답이 모범 답과 맞는지 판단하는 채점 AI
JUDGE_REASONING_EFFORT = "low"  # 두 답의 핵심 내용을 비교해야 해서 minimal보다 한 단계 위
# 맞음/틀림 + 짧은 이유. 생각에 쓴 토큰도 포함된다 (400은 생각만 하다 끝나 채점 실패가 났다)
JUDGE_MAX_TOKENS = 800
RESPONSE_TIME_LIMIT_SECONDS = 10  # 질문 1개 응답 목표. 넘는 질문 수를 평가 때 같이 센다
