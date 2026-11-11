"""챗봇 웹 서버 (FastAPI). 담당: 재웅

화면(static/index.html)을 내려주고, 화면이 부르는 API로 질문·답을 주고받는다.
API 약속은 index.html 맨 위 주석과 아래 pydantic 모델이 같다. 한쪽을 바꾸면 다른 쪽도 바꾼다.

    GET  /                화면
    GET  /api/health      서버 살아 있는지 (VM 배포 확인용)
    POST /api/login       비밀번호 → 로그인 쿠키
    GET  /api/projects    사업 목록 (사이드바·추천 질문)
    POST /api/chat        질문 → 네 단계(run_chatbot) → 답변·출처·조각

비밀번호(.env의 APP_PASSWORD)가 비어 있으면 로그인 없이 열린다 (개발 모드).

실행: uv run uvicorn app.server:app --reload   → http://localhost:8000 (API 문서: /docs)
"""

import hashlib
import hmac
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bidmate.config import APP_PASSWORD, PROJECT_LIST_PATH
from bidmate.pipeline.graph import get_projects, run_chatbot

INDEX_PATH = Path(__file__).parent / "static" / "index.html"
SESSION_COOKIE_NAME = "bidmate_session"
CHUNK_PREVIEW_LENGTH = 600  # 근거 원문 보기에 보여 줄 조각 글자 수

app = FastAPI(title="BidMate RFP 챗봇")


# ── 주고받는 모양 (index.html의 API 약속과 같다) ──


class LoginRequest(BaseModel):
    password: str


class ProjectView(BaseModel):
    """사이드바 사업 목록 한 줄."""

    project_id: str
    agency_name: str
    project_name: str


class ChatRequest(BaseModel):
    """화면이 보내는 질문. chat_history는 화면이 모아 둔 [[질문, 답변], ...]."""

    question: str = Field(min_length=1, max_length=1000)
    chat_history: list[tuple[str, str]] = []


class ChunkView(BaseModel):
    """근거 원문 보기에 보여 줄 조각 하나."""

    chunk_id: str
    project_name: str
    text: str
    score: float  # Chroma 거리. 작을수록 질문과 가깝다


class ChatResponse(BaseModel):
    """화면이 받는 답. v0.2 칸(next_questions 등)은 아직 없다."""

    answer: str
    source_project_names: list[str]
    has_evidence: bool
    rewritten_question: str
    chunks: list[ChunkView]
    seconds: float


# ── 로그인 ──


def make_session_token() -> str:
    """로그인 쿠키에 넣을 값. 비밀번호로 만든 지문이라 서버에 따로 저장할 필요가 없다.

    받는 것: 없음 (APP_PASSWORD를 쓴다)
    넘기는 것: 64글자 문자열. 비밀번호가 같으면 항상 같은 값
    """
    # TODO hmac.new(키, 메시지, hashlib.sha256).hexdigest()
    #   키 = APP_PASSWORD.encode(), 메시지 = b"bidmate-session"
    return ""


def require_login(request: Request) -> None:
    """로그인이 필요한 API 앞에 붙이는 문지기 (Depends로 쓴다). 통과 못 하면 401.

    받는 것: 요청 (쿠키를 본다)
    넘기는 것: 없음. 로그인 안 됐으면 HTTPException(401)을 던진다
    """
    # TODO 1. APP_PASSWORD가 비어 있으면 개발 모드라 그냥 통과 (return)
    # TODO 2. request.cookies.get(SESSION_COOKIE_NAME)로 쿠키 값 꺼내기
    # TODO 3. 쿠키 값이 make_session_token()과 같지 않으면 401
    #   비교는 hmac.compare_digest(쿠키 값, 토큰) (비교 시간 차이로 값을 알아내지 못하게)
    #   raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    return None


@app.post("/api/login")
def login(login_request: LoginRequest, response: Response) -> dict:
    """비밀번호가 맞으면 로그인 쿠키를 준다. 틀리면 401."""
    # TODO 1. hmac.compare_digest(login_request.password, APP_PASSWORD)가 아니면 401
    # TODO 2. response.set_cookie(SESSION_COOKIE_NAME, make_session_token(),
    #           httponly=True, samesite="lax")
    #   httponly: 브라우저 JS가 쿠키를 못 읽게 한다 (쿠키 훔치기 막기)
    # TODO 3. {"ok": True} 돌려주기
    raise HTTPException(status_code=501, detail="재웅이 채운다")


# ── 화면과 상태 확인 ──


@app.get("/")
def index() -> FileResponse:
    """화면 파일을 내려준다. 로그인은 화면 안에서 처리한다."""
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health() -> dict:
    """서버가 살아 있는지. VM 배포 뒤 curl로 확인한다."""
    return {"status": "ok"}


# ── 챗봇 API (로그인 필요) ──


@app.get("/api/projects", dependencies=[Depends(require_login)])
def projects() -> list[ProjectView]:
    """사업 목록. 목록표 파일이 없으면(CI, 데이터 받기 전) 빈 목록."""
    # TODO 1. PROJECT_LIST_PATH.exists()가 아니면 [] 돌려주기
    # TODO 2. get_projects()의 ProjectInfo마다 ProjectView 만들어 목록으로 돌려주기
    #   ProjectView(project_id=…, agency_name=…, project_name=…)
    return []


@app.post("/api/chat", dependencies=[Depends(require_login)])
def chat(chat_request: ChatRequest) -> ChatResponse:
    """질문 하나를 네 단계에 통과시켜 답을 돌려준다."""
    # TODO 1. start_time = time.time()
    # TODO 2. state = run_chatbot(chat_request.question, chat_request.chat_history)
    # TODO 3. seconds = time.time() - start_time
    # TODO 4. state["chunks"]의 (document, score)마다 ChunkView 만들기
    #   chunk_id     = document.metadata.get("chunk_id", "")
    #   project_name = document.metadata.get("project_name", "")
    #   text         = document.page_content[:CHUNK_PREVIEW_LENGTH]
    #   score        = float(score)
    # TODO 5. answer_result = state["answer_result"]로 ChatResponse 만들어 돌려주기
    #   answer, source_project_names, has_evidence는 answer_result에서
    #   rewritten_question은 state["rewritten_question"]
    return ChatResponse(
        answer="찾은 문서에 없습니다.",
        source_project_names=[],
        has_evidence=False,
        rewritten_question=chat_request.question,
        chunks=[],
        seconds=0.0,
    )
